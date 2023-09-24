# -*- coding: utf-8 -*-


def preprocess(subject_ids, client):
    """
    The preprocessing pipeline adapted from the notebook
    """
    import numpy as np
    import pandas as pd
    import gc
    from google.cloud import bigquery
    import os
    min_los = 24
    min_target_onset = 2 * 24

    file_path = os.path.dirname(os.path.realpath(__file__))

    """## Notes
        Extracts two textual features - 'nursing_notes' contains the latest nursing notes and all_notes contains all nursing and physician notes.
        """

    # @title Extract Notes

    def load_notes():
        notes_query = """
            SELECT noteevents.text, noteevents.hadm_id, noteevents.charttime,
                   noteevents.category, noteevents.subject_id, admissions.admittime
            FROM `physionet-data.mimiciii_notes.noteevents` noteevents
            INNER JOIN `physionet-data.mimiciii_clinical.patients` patients
                ON noteevents.subject_id = patients.subject_id
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
                ON noteevents.hadm_id = admissions.hadm_id
            ORDER BY noteevents.subject_id, noteevents.hadm_id, noteevents.charttime;
            """

        notes1 = client.query(notes_query).result().to_dataframe().rename(str.lower, axis='columns')

        # initial_cohort = pd.read_csv(file_path + '/meta_data/initial_cohort.csv')
        notes1 = pd.merge(notes1, subject_ids.copy(), how='inner', on='subject_id')

        return notes1

    gc.collect()
    notes = load_notes()
    gc.collect()

    # @title Exclusion

    # Filter rows where charttime is more than 42 hours after admittime
    mask = (notes['charttime'] - notes['admittime']) <= pd.Timedelta(hours=42)
    filtered_notes = notes[mask][['hadm_id', 'category', 'text', 'charttime']]

    del notes
    gc.collect()

    """### Only Nursing"""

    # Get only the two latest nursings types, and for every patient with two of them, concat
    first_notes = filtered_notes.loc[filtered_notes.groupby(['hadm_id', 'category'])['charttime'].idxmax()]
    first_notes = first_notes[first_notes['category'].isin(['Nursing', 'Nursing/other'])]

    # Concat
    first_notes = first_notes.groupby(['hadm_id'])['text'].apply(' '.join).reset_index()
    first_notes.columns = ['hadm_id', 'first_notes']

    """### All notes"""

    # Keep only the 3 relevant categories
    all_notes = filtered_notes[filtered_notes['category'].isin(['Nursing', 'Nursing/other', 'Physician'])]

    # Concat
    all_notes = all_notes.groupby(['hadm_id'])['text'].apply(' '.join).reset_index()
    all_notes.columns = ['hadm_id', 'all_notes']

    # # @title all_notes Distribution
    # fig, ax = plt.subplots(figsize=(5, 2.5))  # 50% of default (10, 5)

    # # Tokenize
    # batch_size = 1000
    # all_encodings = []

    # for i in range(0, len(all_notes['all_notes']), batch_size):
    #     batch = all_notes['all_notes'].iloc[i:i+batch_size].to_list()
    #     encodings = tokenizer.batch_encode_plus(batch, truncation=False, padding=False)['input_ids']
    #     all_encodings.extend(encodings)

    # lengths = [len(encoding) for encoding in all_encodings]

    # # Convert lengths to pandas Series for histogram
    # pd.Series(lengths).hist(bins=50, edgecolor='black', ax=ax)
    # ax.set_title("Distribution of Description Lengths")
    # ax.set_xlabel("Length")
    # ax.set_ylabel("Frequency")
    # plt.show()

    # del all_encodings
    # del lengths
    # del encodings
    # gc.collect()

    """## Tabular - Demographic, Vital and Lab
        
        ### Demograhic
        """

    def load_hosps():
        hospquery = \
            """
                SELECT admissions.subject_id, admissions.hadm_id
                , admissions.admittime, admissions.dischtime
                , admissions.ethnicity
                , patients.gender, patients.dob, patients.dod
                FROM `physionet-data.mimiciii_clinical.admissions` admissions
                INNER JOIN `physionet-data.mimiciii_clinical.patients` patients
                    ON admissions.subject_id = patients.subject_id
                WHERE admissions.has_chartevents_data = 1
                ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
                """
        hosps1 = client.query(hospquery).result().to_dataframe().rename(str.lower, axis='columns')

        #  initial_cohort = pd.read_csv(file_path + '/meta_data/initial_cohort.csv')
        hosps1 = pd.merge(hosps1, subject_ids.copy(), how='inner', on='subject_id')
        return hosps1

    hosps = load_hosps()

    # @title Feature Extraction

    # Generate feature columns for los, age and mortality
    def age(admittime, dob):
        if admittime < dob:
            return 0
        return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))

    hosps['age'] = hosps.apply(lambda row: age(row['admittime'], row['dob']), axis=1)
    hosps['los_hosp_hr'] = (hosps.dischtime - hosps.admittime).astype('timedelta64[h]')
    hosps['mort'] = np.where(~np.isnat(hosps.dod), 1, 0)

    # Ethnicity - one hot encoding
    hosps.ethnicity = hosps.ethnicity.str.lower()
    hosps.loc[(hosps.ethnicity.str.contains('^white')), 'ethnicity'] = 'white'
    hosps.loc[(hosps.ethnicity.str.contains('^black')), 'ethnicity'] = 'black'
    hosps.loc[
        (hosps.ethnicity.str.contains('^hisp')) | (hosps.ethnicity.str.contains('^latin')), 'ethnicity'] = 'hispanic'
    hosps.loc[(hosps.ethnicity.str.contains('^asia')), 'ethnicity'] = 'asian'
    hosps.loc[~(hosps.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian']))), 'ethnicity'] = 'other'
    hosps = pd.concat([hosps, pd.get_dummies(hosps['ethnicity'], prefix='eth')], axis=1)

    # Gender to binary
    hosps['gender'] = np.where(hosps['gender'] == "M", 1, 0)

    # @title Exclusion

    # First admission for each patient
    hosps = hosps.sort_values('admittime').groupby('subject_id').first().reset_index()
    #print(f"1. Include only first admissions: N={hosps.shape[0]}")

    # Exclude patients with less than min_los hours.
    hosps = hosps[(hosps['dischtime'] - hosps['admittime']).dt.total_seconds() / 60 / 60 > min_los]
    #print(f"3. Include only patients who admitted for at least {min_los} hours: N={hosps.shape[0]}")

    # hosps = hosps[hosps.age.between(17,300,inclusive='neither')]
    # print(f"2. Exclusion by ages: N={hosps.shape[0]}")

    # <implement here> Exclude by death time
    hosps['time_diff'] = (hosps['dod'] - hosps['admittime']).dt.total_seconds() / 60 / 60  # in hours
    hosps = hosps[~(hosps['time_diff'] <= min_target_onset)]
    hosps = hosps.drop(['time_diff'], axis='columns')

    #print(f"4. Exclude patients who died within {min_target_onset}-hours of admission: N={hosps.shape[0]}")

    """### Lab"""

    # @title Query DB
    labquery = \
        """--sql
              SELECT labevents.subject_id ,labevents.hadm_id ,labevents.charttime
              , labevents.itemid, labevents.valuenum
              , admissions.admittime
              FROM `physionet-data.mimiciii_clinical.labevents` labevents
                INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
                ON labevents.subject_id = admissions.subject_id
                AND labevents.hadm_id = admissions.hadm_id
                AND labevents.charttime >=(admissions.admittime)
                AND itemid in UNNEST(@itemids)
            """
    lavbevent_meatdata = pd.read_csv(file_path + '/meta_data/labs_metadata.csv')
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("itemids", "INTEGER", lavbevent_meatdata['itemid'].tolist()),
        ]
    )
    labs = client.query(labquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')

    #  initial_cohort = pd.read_csv(file_path + '/meta_data/initial_cohort.csv')
    labs = pd.merge(labs, subject_ids.copy(), how='inner', on='subject_id')

    # @title Filter Invalid Mesurments
    labs = labs[labs['hadm_id'].isin(hosps['hadm_id'])]
    labs = pd.merge(labs, lavbevent_meatdata, on='itemid')
    labs = labs[labs['valuenum'].between(labs['min'], labs['max'], inclusive='both')]
    #print(labs)

    # @title Exclusion
    labs = labs[labs['charttime'] - labs['admittime'] <= pd.Timedelta(hours=42)]
    labs = labs[['hadm_id', 'charttime', 'feature name', 'valuenum']]

    # @title Pivot & Find First Lab Tests

    # Groupby admission id and feature name, and keep only the row with the earliest charttime
    temp = labs.sort_values(['charttime'], ascending=True).groupby(['hadm_id', 'feature name']).first().reset_index()
    # Pivot the dataframe
    final_labs = temp.pivot(index='hadm_id', columns='feature name', values='valuenum').reset_index()
    final_labs.columns = final_labs.columns.str.lower()

    final_labs.head()

    del temp

    final_labs

    """### Vital"""

    # @title Extract
    vitquery = \
        """--sql
              SELECT chartevents.subject_id ,chartevents.hadm_id ,chartevents.charttime
              , chartevents.itemid, chartevents.valuenum
              , admissions.admittime
              FROM `physionet-data.mimiciii_clinical.chartevents` chartevents
              INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
              ON chartevents.subject_id = admissions.subject_id
              AND chartevents.hadm_id = admissions.hadm_id
              AND chartevents.charttime between
              (admissions.admittime)
              AND (admissions.admittime + interval '48' hour)
              AND itemid in UNNEST(@itemids)
              -- exclude rows marked as error
              AND chartevents.error IS DISTINCT FROM 1
            """
    vital_meatdata = pd.read_csv(file_path + '/meta_data/vital_metadata.csv')
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("itemids", "INTEGER", vital_meatdata['itemid'].tolist()),
        ]
    )
    vits = client.query(vitquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')

    #   initial_cohort = pd.read_csv(file_path + '/meta_data/initial_cohort.csv')
    vits = pd.merge(vits, subject_ids.copy(), how='inner', on='subject_id')

    # @title Filter Invalid Measurement

    # Merge the data with the vital_meatdata
    merged = vits.merge(vital_meatdata, on='itemid')
    # Lowercase
    merged['feature name'] = merged['feature name'].str.lower()

    # Keep only values in range
    filtered = merged[(merged['max'] >= merged['valuenum']) & (merged['valuenum'] >= merged['min'])]

    # Drop unrelevant columns
    vits = filtered.drop(['min', 'max', 'itemid'], axis='columns')

    vits.head()

    # @title Units Conversion

    def fahrenheit_to_celsius(f):
        return (f - 32) * 5 / 9

    # Locate the rows with 'tempF' feature
    tempF_rows = vits['feature name'] == 'tempf'

    # Convert the 'valuenum' for 'tempF' rows to Celsius
    vits.loc[tempF_rows, 'valuenum'] = vits.loc[tempF_rows, 'valuenum'].apply(fahrenheit_to_celsius)

    # Change the feature name from 'tempF' to 'tempC'
    vits.loc[tempF_rows, 'feature name'] = 'tempc'

    # @title Exclusion
    vits = vits[vits['charttime'] - vits['admittime'] <= pd.Timedelta(hours=42)]
    vits = vits[['hadm_id', 'charttime', 'feature name', 'valuenum']]

    # @title Calculate Measurement Statistics in First 42 Hours

    # Group by hadm_id and feature_name (except for weight), and calculate min, max, and mean
    vit_without_weights = vits[vits['feature name'] != 'weight']
    result = vit_without_weights.groupby(['hadm_id', 'feature name']).agg(
        {'valuenum': ['min', 'max', 'mean']}).reset_index()
    #print(result)
    # Create separate pivot tables for min, max, and median
    pivot_min = result.pivot(index='hadm_id', columns='feature name', values=('valuenum', 'min')).add_suffix(
        '_min').reset_index()
    pivot_max = result.pivot(index='hadm_id', columns='feature name', values=('valuenum', 'max')).add_suffix(
        '_max').reset_index()
    pivot_median = result.pivot(index='hadm_id', columns='feature name', values=('valuenum', 'mean')).add_suffix(
        '_mean').reset_index()

    # Merge the three pivot tables into a single table on hadm_id
    result = pivot_min.merge(pivot_max, on='hadm_id').merge(pivot_median, on='hadm_id')
    result.columns = result.columns.str.lower()
    # Re-arrange
    features = ['heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'spo2', 'glucose', 'tempc']
    column_order = ['hadm_id'] + [f'{feature}_{stat}' for feature in features for stat in ['min', 'max', 'mean']]
    temp_vits1 = result[column_order]

    # Keep only the first weight of patient
    temp_vits2 = vits[vits['feature name'] == 'weight'].copy()

    # Groupby admission id and feature name, and keep only the row with the earliest charttime
    temp_vits2 = temp_vits2.sort_values(['charttime'], ascending=True).groupby(
        ['hadm_id', 'feature name']).first().reset_index()
    temp_vits2 = temp_vits2.filter(['subject_id', 'hadm_id', 'valuenum']).rename(columns={'valuenum': 'weight'})

    final_vits = temp_vits1.merge(temp_vits2, on=['hadm_id'], how='outer')

    del vits
    del temp_vits2, temp_vits1

    final_vits

    """### Medication
        Extracts two features from the medication table - the amount of medications and the amount of unique medications taken.
        """

    def load_meds():
        med_query = \
            """
                SELECT admissions.hadm_id, admissions.subject_id, admissions.admittime, drug, startdate, enddate
                FROM `physionet-data.mimiciii_clinical.prescriptions` prescriptions
                INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
                    ON prescriptions.hadm_id = admissions.hadm_id
                ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
                """

        meds = client.query(med_query).result().to_dataframe().rename(str.lower, axis='columns')

        #  initial_cohort = pd.read_csv(file_path + '/meta_data/initial_cohort.csv')
        meds = pd.merge(meds, subject_ids.copy(), how='inner', on='subject_id')
        return meds

    meds = load_meds()

    # @title Exclusion
    print(meds.shape)
    meds = meds[meds['startdate'] - meds['admittime'] <= pd.Timedelta(hours=42)]
    meds = meds[['subject_id', 'hadm_id', 'drug']]

    meds_count = meds.groupby('hadm_id').size().rename('meds_count', inplace=True)
    meds_unique_count = meds.groupby('hadm_id')['drug'].nunique().rename('meds_unique_count', inplace=True)

    final_meds = pd.merge(meds_unique_count, meds_count, on='hadm_id')
    gc.collect()

    final_meds

    """## Final Preprocessing
        Combines demographic, vital and lab dfs and adds prediction objectives
        """

    # @title Combine

    # Merge tabular
    combined_df = pd.merge(final_vits, final_labs, on='hadm_id')
    combined_df = pd.merge(hosps, combined_df, on='hadm_id', how='left')
    combined_df = pd.merge(combined_df, final_meds, on='hadm_id', how='left')

    # Merge text
    combined_df = combined_df.merge(first_notes, how='left', on='hadm_id')
    del first_notes
    gc.collect()
    combined_df = combined_df.merge(all_notes, how='left', on='hadm_id')
    del all_notes
    gc.collect()

    del hosps
    del final_vits
    del final_labs
    del final_meds
    gc.collect()

    combined_df

    # @title Keep rows with prediction objectives
    bool1 = (combined_df['dischtime'] - combined_df['admittime']) >= pd.Timedelta(hours=48)
    bool2 = (combined_df['dod'] - combined_df['admittime']) >= pd.Timedelta(hours=48)

    # Either died or discharged after 48 hours
    combined_df = combined_df[bool1 | bool2]

    combined_df = combined_df[(combined_df['age'] > 18 & (combined_df['age'] < 90))]

    original_hosps = load_hosps()

    # Get the second admission for each patient
    original_hosps['rank'] = original_hosps.groupby('subject_id')['admittime'].rank(method="first")
    second_admissions = original_hosps[original_hosps['rank'] == 2].reset_index(drop=True)
    del original_hosps['rank']

    second_admissions = second_admissions[['subject_id', 'admittime']]
    second_admissions.rename(columns={'admittime': 'second_admittime'})
    #print(second_admissions)

    # If readmitted within 30 days after discharge
    combined_df = pd.merge(combined_df, second_admissions, on='subject_id', how='left', suffixes=('', '_second'))
    print(combined_df.columns)
    combined_df['readmission_target'] = (
            combined_df['admittime_second'] - combined_df['dischtime'] <= pd.Timedelta(days=30))

    # If died during hospitalization or within 30 days after discharge
    mask1 = (combined_df['dod'] - combined_df['dischtime'] <= pd.Timedelta(days=30))
    mask2 = (combined_df['dod'] < combined_df['dischtime'])
    combined_df['mort_target'] = mask1 | mask2

    # If hospitalization longer than 7 days
    combined_df['prolonged_stay_target'] = combined_df['los_hosp_hr'] >= 7 * 24

    combined_df[['readmission_target', 'mort_target', 'prolonged_stay_target']] = combined_df[
        ['readmission_target', 'mort_target', 'prolonged_stay_target']].astype(int)

    final_df = combined_df.drop(
        ['hadm_id', 'mort', 'admittime', 'dischtime', 'subject_id', 'admittime_second', 'dod', 'dob', 'ethnicity',
         'los_hosp_hr'], axis='columns')
    final_df.to_csv('final_df.csv', index=False)
    print(final_df)
    print(final_df.columns)

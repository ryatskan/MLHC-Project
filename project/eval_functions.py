import pandas as pd


def pipeline_transform(final_df, tabular_preprocess, text_preprocess):
    final_df = final_df.copy()
    targets = ['readmission_target', 'mort_target', 'prolonged_stay_target']
    X_test, y_test = tabular_process(final_df, targets, tabular_preprocess)
    X_test = text_preprocess.transform(X_test)
    return X_test, y_test


def pipeline_eval(final_df, model):
    probabilities = model.predict_proba(final_df)
    print(model.classes_)
    return probabilities


def tabular_process(df, targets, tabular_preprocess):
    tabular_features = ['gender', 'age',
                        'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
                        'diasbp_max', 'glucose_max', 'heartrate_max', 'meanbp_max',
                        'resprate_max', 'spo2_max', 'sysbp_max', 'tempc_max', 'diasbp_min',
                        'glucose_min', 'heartrate_min', 'meanbp_min', 'resprate_min',
                        'spo2_min', 'sysbp_min', 'tempc_min', 'diasbp_mean', 'glucose_mean',
                        'heartrate_mean', 'meanbp_mean', 'resprate_mean', 'spo2_mean',
                        'sysbp_mean', 'tempc_mean', 'albumin', 'anion gap', 'bicarbonate',
                        'bilirubin', 'bun', 'chloride', 'creatinine', 'glucose', 'hematocrit',
                        'hemoglobin', 'inr', 'lactate', 'magnesium', 'phosphate', 'platelet',
                        'potassium', 'pt', 'ptt', 'sodium', 'wbc', 'weight', 'meds_count', 'meds_unique_count']

    features = tabular_features.copy()
    text_col = 'all_notes'
    features.append(text_col)
    df = df.copy()
    data = df[features + targets]

    data.loc[:, text_col] = data.loc[:, text_col].fillna("Unknown")

    test_data = pd.DataFrame(tabular_preprocess.transform(data), columns=tabular_features + ['all_notes'] + targets)

    test_data[tabular_features] = test_data[tabular_features].astype(float)
    test_targets = test_data[targets].copy()

    test_data = test_data.drop(columns=targets)
    print(test_data)
    return test_data, test_targets.astype(int)

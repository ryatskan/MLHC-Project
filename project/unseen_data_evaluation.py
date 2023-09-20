def run_pipeline_on_unseen_data(subject_ids, client):
    # from . import eval_functions as eval
    """
  Run your full pipeline, from data loading to prediction.

  :param subject_ids: A list of subject IDs of an unseen test set.
  :type subject_ids: List[int]

  :param client: A BigQuery client object for accessing the MIMIC-III dataset.
  :type client: google.cloud.bigquery.client.Client

  :return: DataFrame with the following columns:
              - subject_id: Subject IDs, which in some cases can be different due to your analysis.
              - mortality_proba: Prediction probabilities for mortality.
              - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
              - readmission_proba: Prediction probabilities for readmission.
  :rtype: pandas.DataFrame
  """
    from .eval_functions import pipeline_eval, pipeline_transform
    from .preprocessing import preprocess

    import pandas as pd
    from joblib import load
    from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                                 recall_score)
    import os
    import warnings
    warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.validation')

    file_path = os.path.dirname(os.path.realpath(__file__))
    subject_ids = pd.Series(subject_ids, name='subject_id')
    # GENERAL PREPROCESS
    preprocess(subject_ids=subject_ids, client=client)
    df = pd.read_csv('final_df.csv')

    # LOAD PREPROCESS
    preprocess_readmission = load(file_path + '/saved_models/preprocess_readmission')
    preprocess_prolonged = load(file_path + '/saved_models/preprocess_prolonged')
    preprocess_mort = load(file_path + '/saved_models/preprocess_mort')

    # LOAD MODELS
    model_readmission = load(file_path + '/saved_models/model_readmission')
    model_prolonged = load(file_path + '/saved_models/model_prolonged')
    model_mort = load(file_path + '/saved_models/model_mort')

    preprocess_tabular = load(file_path + '/saved_models/preprocess_tabular')

    # PIPELINE
    X_readmission, y_readmission = pipeline_transform(df, preprocess_tabular, preprocess_readmission)
    X_prolonged, y_prolonged = pipeline_transform(df, preprocess_tabular, preprocess_prolonged)
    X_mort, y_mort = pipeline_transform(df, preprocess_tabular, preprocess_mort)

    # PREDICT
    proba_mort = pd.DataFrame(pipeline_eval(X_mort, model_mort), columns=['mort_0', 'mort_1'])
    proba_readmission = pd.DataFrame(pipeline_eval(X_readmission, model_readmission),
                                     columns=['readmission_0', 'readmission_1'])
    proba_prolonged = pd.DataFrame(pipeline_eval(X_prolonged, model_prolonged), columns=['prolonged_0', 'prolonged_1'])

    print(proba_mort)
    print(proba_readmission)
    print(proba_prolonged)

    # EVAL
    out = pd.DataFrame(pd.Series(subject_ids, name='subject_id')).join(proba_mort).join(proba_prolonged)
    out = (out.join(proba_readmission)[['subject_id', 'mort_1', 'prolonged_1', 'readmission_1']]
           .rename
           (columns={'subject_id': 'subject_id', 'mort_1': 'mortality_proba', 'prolonged_1': 'prolonged_LOS_proba',
                     'readmission_1': 'readmission_proba'}))

    def calc(predicted, labels):  # BASIC EVALS
        print(f"f1-score: {f1_score(labels, predicted)}")
        print(f"accuracy: {accuracy_score(labels, predicted)}")
        print(f"precision: {precision_score(labels, predicted)}")
        print(f"recall: {recall_score(labels, predicted)}")

    calc(proba_readmission.idxmax(axis=1).replace({'readmission_0': 0, 'readmission_1': 1}),
         y_readmission['readmission_target'])
    calc(proba_prolonged.idxmax(axis=1).replace({'prolonged_0': 0, 'prolonged_1': 1}),
         y_prolonged['prolonged_stay_target'])
    calc(proba_mort.idxmax(axis=1).replace({'mort_0': 0, 'mort_1': 1}), y_mort['mort_target'])
    return out

# MLHC-Project
 
The entire code for analysis is in the `notebook.ipynb` file:

1. The first part is the pre-processing - connecting to MIMIC dataset, adding demographic, lab, vital, notes, and medication processing, and finally merging it all into one final dataframe.
2. The second part is the training and evaluation of the scikit-learn plug-and-play models LR and XGBOOST, and the helper functions for splitting into test and train sets.
3. The last parts are the training of the NLP-specific models - DAN and BART. These models require more complex processing, and in the DAN part, the entire model is designed from scratch using PyTorch and pre-trained embeddings.

The `saved_models` directory contains all the trained models and their preprocessing. One of them is a fine-tuned BART model, and the rest are scikit-learn saved models.

The `meta_data` directory contains two helper csv files for vital and labs pre-processing.

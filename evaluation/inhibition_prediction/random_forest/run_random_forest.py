import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from evaluation.inhibition_prediction.random_forest.scorers import auprc, sensitivity, specificity
from sklearn.metrics import make_scorer, roc_auc_score, auc, precision_recall_curve
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_validate

try:
    print("Loading Data")
    train_df = pd.read_csv('data/inhibition_training_data/tb_inhibition_A&M/final_data/train_df_with_morgan.csv')
except:
    print("No Existing Data, Building Data")
    train_df = pd.read_csv('data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')

    feature_list = []
    column_names = ['SMILES'] + [str(item) for item in np.arange(0,2048,1)] + ['inhibition']

    for idx, row in tqdm(train_df.iterrows(), total = len(train_df)):
        mol = Chem.MolFromSmiles(row['SMILES'])
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        features = [row['SMILES']] + features.tolist() + [row['inhibition']]
        feature_list.append(features)

    train_df = pd.DataFrame(feature_list,
                            columns = column_names)

    train_df.to_csv('evaluation/inhibition_prediction/random_forest/output/train_df_with_morgan.csv',
                    index = False)

X = train_df
y = train_df['inhibition']

cv = 5
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

for i in range(cv):
    
    print(f"Running Fold {i}")

    # Get Splits
    test_SMILES = pd.read_csv(f'data/cv_evaluation/inhibition_dmpnn/fold_{i}/test_pred.csv')['SMILES']
    X_train_with_label = X[~X['SMILES'].isin(test_SMILES)]
    X_train = X_train_with_label.drop(['SMILES','inhibition'], axis=1)
    y_train = X_train_with_label['inhibition']

    X_test_with_label = X[X['SMILES'].isin(test_SMILES)].copy()
    X_test = X_test_with_label.drop(['SMILES','inhibition'], axis=1)
    y_test = X_test_with_label['inhibition']

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = len(X_test)/len(X_train))

    rf = RandomForestClassifier(n_estimators=100,
                                n_jobs=-1, 
                                random_state=0,
                                class_weight='balanced',
                                verbose=0,)

    rf.fit(X_train, y_train)

    ## CV
    test_pred = rf.predict_proba(X_test)[:,1]

    test_auroc = roc_auc_score(y_test, test_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, test_pred)
    test_auprc = auc(recall, precision)
    print("AUROC:", test_auroc, "     AUPRC:", test_auprc)

    X_test_with_label['pred'] = test_pred
    X_test_with_label[['SMILES','inhibition','pred']].to_csv(f'evaluation/inhibition_prediction/random_forest/output/fold{i}_test_predictions.csv',
                                                             index = False)
    
    ## External
    external_dataset = pd.read_csv('data/validation_dataset/outsample_data_morgan_feature.csv')
    external_X = external_dataset.drop(['SMILES','inhibition'],axis=1)
    external_y = external_dataset['inhibition']
    external_pred = rf.predict_proba(external_X)[:,1]

    external_auroc = roc_auc_score(external_y, external_pred)
    precision, recall, thresholds = precision_recall_curve(external_y, external_pred)
    external_auprc = auc(recall, precision)
    print("AUROC:", external_auroc, "     AUPRC:", external_auprc)

    external_dataset['pred'] = external_pred
    external_dataset[['SMILES','inhibition','pred']].to_csv(f'evaluation/inhibition_prediction/random_forest/output/fold{i}_external_predictions.csv',
                                                             index = False)
    
    ## Bench
    external_dataset = pd.read_csv('data/validation_dataset/outsample_data_morgan_feature.csv')
    external_X = external_dataset.drop(['SMILES','inhibition'],axis=1)
    external_y = external_dataset['inhibition']
    external_pred = rf.predict_proba(external_X)[:,1]

    external_auroc = roc_auc_score(external_y, external_pred)
    precision, recall, thresholds = precision_recall_curve(external_y, external_pred)
    external_auprc = auc(recall, precision)
    print("AUROC:", external_auroc, "     AUPRC:", external_auprc)

    external_dataset['pred'] = external_pred
    external_dataset[['SMILES','inhibition','pred']].to_csv(f'evaluation/inhibition_prediction/random_forest/output/fold{i}_bench_predictions.csv',
                                                             index = False)
    
    

## Get Split
# test_SMILES = pd.read_csv('data/cv_evaluation/inhibition_dmpnn/fold_0/test_pred.csv')['SMILES']

# X_train_with_label = X[~X['SMILES'].isin(test_SMILES)]
# X_train = X_train_with_label.drop(['SMILES','inhibition'], axis=1)
# y_train = X_train_with_label['inhibition']

# X_test_with_label = X[X['SMILES'].isin(test_SMILES)].copy()
# X_test = X_test_with_label.drop(['SMILES','inhibition'], axis=1)
# y_test = X_test_with_label['inhibition']

# X_train_with_label, X_test_with_label, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# rf = RandomForestClassifier(n_estimators=100,
#                             n_jobs=-1, 
#                             random_state=0,
#                             class_weight='balanced',
#                             verbose=0,)

# print("Running Cross Validation")
# scoring = {'auroc':'roc_auc',
#            'acc':'accuracy',
#            'precision':'precision',
#            'recall':'recall',
#            'auprc': make_scorer(auprc, greater_is_better=True),
#            'sensitivity': make_scorer(sensitivity, greater_is_better=True),
#            'specificity': make_scorer(specificity, greater_is_better=True)}

# cv_results = cross_validate(rf,
#                          X_train,
#                          y_train,
#                          cv=5,
#                          scoring=scoring,
#                          return_train_score=False,
#                          return_estimator=True,)

# cv_estimators = cv_results['estimator']

# print(cv_results)

# fold = 0
# for estimator in cv_estimators:
#     # CV Performance
#     test_pred = estimator.predict_proba(X_test)[:,1]
#     test_auroc = roc_auc_score(y_test, test_pred)
#     precision, recall, thresholds = precision_recall_curve(y_test, test_pred)
#     test_auprc = auc(recall, precision)
#     print("AUROC:", test_auroc, "     AUPRC:", test_auprc)

#     X_test_with_label['pred'] = test_pred
#     X_test_with_label[['SMILES','inhibition','pred']].to_csv(f'evaluation/inhibition_prediction/random_forest/output/fold{fold}_test_predictions.csv',
#                                                              index = False)
    
#     ## External Performance

#     fold += 1
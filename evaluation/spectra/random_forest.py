import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, auc, precision_recall_curve
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_validate

results = []

for split in tqdm(np.arange(4,20,1)):
    
    print(f"Running Split {split}")

    train_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/{split}.0_data.csv')

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

    X = train_df
    y = train_df['inhibition']

    cv = 5
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    for i in range(cv):
        
        print(f"Running Fold {i}")

        # Get Splits
        # test_SMILES = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/checkpoints/chemprop_spectra_{split}/fold_{i}/test_preds.csv')['SMILES']
        # X_train_with_label = X[~X['SMILES'].isin(test_SMILES)]
        # X_train = X_train_with_label.drop(['SMILES','inhibition'], axis=1)
        # y_train = X_train_with_label['inhibition']

        # X_test_with_label = X[X['SMILES'].isin(test_SMILES)].copy()
        # X_test = X_test_with_label.drop(['SMILES','inhibition'], axis=1)
        # y_test = X_test_with_label['inhibition']

        X_train, X_test, y_train, y_test = train_test_split(X.drop(['SMILES', 'inhibition'], axis=1), 
                                                            y, 
                                                            test_size=test_ratio)

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

        # Save results for this fold
        results.append({
            'split': split,
            'fold': i,
            'AUROC': test_auroc,
            'AUPRC': test_auprc
        })

        # X_test_with_label['pred'] = test_pred
        # X_test_with_label[['SMILES','inhibition','pred']].to_csv(f'/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/random_forest/fold{i}_test_predictions.csv',
        #                                                          index = False)

# Save all results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/rf_test_results.csv', index=False)
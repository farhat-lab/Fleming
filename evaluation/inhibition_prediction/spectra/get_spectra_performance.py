import os
import re
import pandas as pd
from sklearn.metrics import roc_auc_score

for spectra_param in range(4,20):
    folder_path = f'evaluation/spectra/checkpoints/chemprop_spectra_evidential_{spectra_param}.0/'
    auroc = []
    evi_auroc = []
    for folder in range(0,5):
        if spectra_param in [5,7,10,15]:
            pred_df = pd.read_csv(folder_path + f'fold_{folder}/test_preds.csv')
        else:
            pred_df = pd.read_csv(folder_path + f'fold_{folder}/test_preds_evi.csv')
        label_df = pd.read_csv(folder_path + f'fold_{folder}/test_data.csv')

        combined_df = pd.merge(pred_df, 
                               label_df, 
                               on='SMILES', 
                               how='inner')

        auroc = auroc + [roc_auc_score(combined_df['inhibition_y'], combined_df['inhibition_x'])]

        try:
            evi_df = combined_df[combined_df['inhibition_dirichlet_uncal_uncertainty'] < 0.4]
            evi_auroc = evi_auroc + [roc_auc_score(evi_df['inhibition_y'], evi_df['inhibition_x'])]
        except:
            print(f'No Evi AUROC for spectra_param {spectra_param} fold {folder}')
            continue
    
    print(f'Spectra Param {spectra_param} AUROC: {sum(auroc)/len(auroc)} +/- {pd.Series(auroc).std()}')
    print(f'Spectra Param {spectra_param} Evi AUROC: {sum(evi_auroc)/len(evi_auroc)} +/- {pd.Series(evi_auroc).std()}')

    


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

for split in ['5.0','7.0','10.0','15.0']:

    print(f"Split Spectra {split}")

    for fold in range(5):
        true_df = pd.read_csv(f'evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_{split}/fold_{fold}/test_data.csv')
        pred_df = pd.read_csv(f'evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_{split}/fold_{fold}/test_preds.csv')

        hc_pred = pred_df[pred_df['inhibition_dirichlet_uncal_uncertainty'] < 0.4]
        hc_pred = hc_pred.merge(true_df,
                                on = ['SMILES'],
                                how = 'left')

        try:
            print(roc_auc_score(y_true=hc_pred['inhibition_y'],
                                y_score=hc_pred['inhibition_x']))
        except:
            continue
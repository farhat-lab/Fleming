import pandas as pd
import os

for dir in os.listdir('ADMET_predictions'):
    if os.path.exists(f'ADMET_predictions/{dir}/test_scores.csv'):
        try:
            print(f"{dir}\t{pd.read_csv(f'ADMET_predictions/{dir}/test_scores.csv')['Mean auc'].item()}")
        except:
            print(f"{dir}\tNo AUC")
        
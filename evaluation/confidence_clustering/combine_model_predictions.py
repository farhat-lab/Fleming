import pandas as pd
import numpy as np

def ensemble_predictions(prediction_dir1:str,
                         prediction_dir2:str):
    
    prediction_df1 = pd.read_csv(prediction_dir1)
    prediction_df2 = pd.read_csv(prediction_dir2)

    prediction_df = pd.concat([prediction_df1,prediction_df2])
    prediction_df = prediction_df.groupby('SMILES', as_index=False).mean()

    prediction_df.to_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/dragonfold_prediction_dir.csv',
                         index = False)


if __name__ == '__main__':

    ensemble_predictions('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/raw_inhibition_prediction_1.csv',
                         '/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/raw_inhibition_prediction_2.csv')
import pandas as pd

mmv_df = pd.read_csv('evaluation/mmv/MMV HGL4 Library with SMILES.txt', sep='\t')
# mmv_df[['SMILES']].to_csv('evaluation/mmv/mmv_smiles.csv', index=False)
print(len(mmv_df))

train_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')

hits_df = pd.read_excel('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/mmv/Whole Cell Actives from MMV HGL4 Library.xlsx',
                        sheet_name='Data')
hits_df['hit'] = 1

complete_data = hits_df.merge(mmv_df,
                             left_on='LGCY Batch Name',
                             right_on='ENTITY_ID_1',
                             how='right')

complete_data = complete_data[~complete_data['SMILES'].isin(train_df['SMILES'])]

complete_data['hit'] = complete_data['hit'].fillna(0)

pred_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/mmv/mmv_inhibition_prediction.csv')
# pred_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/mmv/mmv_inhibition_prediction_old_models.csv')

complete_data = complete_data.merge(pred_df,
                                    on='SMILES',
                                    how = 'left')

complete_data.to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/mmv/mmv_complete_data_with_predictions.csv',
                     index=False)

# Calculate AUROC
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(complete_data['hit'], 
                      complete_data['inhibition'])

print(f"MMV HGL4 Library AUROC: {auroc}")

confident_hits = complete_data[complete_data['inhibition_dirichlet_uncal_uncertainty'] <= 0.4]

auroc = roc_auc_score(confident_hits['hit'], 
                      confident_hits['inhibition'])

print(auroc)

complete_data['predicted_hit'] = (complete_data['inhibition'] >= 0.7).astype(int)
complete_data['confident_predicted_hit'] = ((complete_data['inhibition'] >= 0.7) & (complete_data['inhibition_dirichlet_uncal_uncertainty'] <= 0.4)).astype(int)

# True Positive Rate
tp = sum((complete_data['predicted_hit'] == 1) & (complete_data['hit'] == 1))
fp = sum((complete_data['predicted_hit'] == 1) & (complete_data['hit'] == 0))
tpr = tp / (tp + fp)
print(tp)
print(fp)
print(f"Threshold = 0.7: {tpr}")

# Confident True Positive Rate
ctp = sum((complete_data['confident_predicted_hit'] == 1) & (complete_data['hit'] == 1))
cfp = sum((complete_data['confident_predicted_hit'] == 1) & (complete_data['hit'] == 0))
ctpr = ctp / (ctp + cfp)
print(ctp)
print(cfp)
print(f"Threshold 0.7 and uncertainty 0.4: {ctpr}")

complete_data[complete_data['confident_predicted_hit'] == 1][['SMILES','LGCY Batch Name','Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation ','inhibition','inhibition_dirichlet_uncal_uncertainty']].to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/mmv/mmv_predicted_hits_confident.csv',
                                                                                                                                                                                                                 index=False)

complete_data[complete_data['hit'] == 1][['SMILES','LGCY Batch Name','Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation ','inhibition','inhibition_dirichlet_uncal_uncertainty']].to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/mmv/mmv_hits_predictions.csv',
                                                                                                                                                index=False)
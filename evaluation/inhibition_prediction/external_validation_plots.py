import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, PrecisionRecallDisplay, precision_recall_curve, roc_auc_score, roc_curve

true_label = pd.read_csv('data/validation_dataset/outsample_data.csv')

print("Positive Rate: ", sum(true_label['inhibition'])/len(true_label))

### AUROC
plt.figure(figsize=(12.8,4.8))
ax = plt.subplot(111)
plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')

## DMPNN
dmpnn_prediction = pd.read_csv('evaluation/inhibition_prediction/output/new_outsample_prediction.csv')

fpr, tpr, threshold = roc_curve(true_label['inhibition'], dmpnn_prediction['inhibition'])

plt.plot(fpr, tpr, 
         label = "DMPNN: AUROC = %0.2f" % roc_auc_score(true_label['inhibition'], dmpnn_prediction['inhibition']))

## EGNN 
egnn_prediction = pd.DataFrame()
for i in range(5):
    egnn_fold_prediction = pd.read_csv(f'data/egnn/evaluation/external_predictions/EGNN_fold_{i}_predictions.csv')
    if i == 0:
        egnn_prediction = egnn_fold_prediction
    else:
        egnn_prediction[f'prediction_{i}'] = egnn_fold_prediction['prediction']

egnn_prediction['avg_prediction'] = egnn_prediction[['prediction','prediction_1','prediction_2','prediction_3','prediction_4']].mean(axis=1)

egnn_prediction = true_label.merge(egnn_prediction,
                                    on = ['SMILES'],
                                    how = 'left')

egnn_prediction = egnn_prediction.fillna(0)
 
fpr, tpr, threshold = roc_curve(egnn_prediction['inhibition'], egnn_prediction['avg_prediction'])

plt.plot(fpr, tpr, 
         label = "EGNN: AUROC = %0.2f" % roc_auc_score(egnn_prediction['inhibition'], egnn_prediction['avg_prediction']))

## FFNN
ffnn_prediction = pd.DataFrame()
for i in range(5):
    ffnn_fold_prediction = pd.read_csv(f'data/egnn/evaluation/external_predictions/FFNN_fold_{i}_predictions.csv')
    if i == 0:
        ffnn_prediction = ffnn_fold_prediction
    else:
        ffnn_prediction[f'prediction_{i}'] = ffnn_fold_prediction['prediction']

ffnn_prediction['avg_prediction'] = ffnn_prediction[['prediction','prediction_1','prediction_2','prediction_3','prediction_4']].mean(axis=1)

ffnn_prediction = true_label.merge(ffnn_prediction,
                                    on = ['SMILES'],
                                    how = 'left')

ffnn_prediction = ffnn_prediction.fillna(0)
 
fpr, tpr, threshold = roc_curve(ffnn_prediction['inhibition'], ffnn_prediction['avg_prediction'])

plt.plot(fpr, tpr, 
         label = "FFNN: AUROC = %0.2f" % roc_auc_score(ffnn_prediction['inhibition'], ffnn_prediction['avg_prediction']))

## Random Forest
rf_prediction = pd.DataFrame()
for i in range(5):
    rf_fold_prediction = pd.read_csv(f'evaluation/inhibition_prediction/random_forest/output/fold{i}_external_predictions.csv')
    if i == 0:
        rf_prediction = rf_fold_prediction
    else:
        rf_prediction[f'prediction_{i}'] = rf_fold_prediction['pred']

rf_prediction['avg_prediction'] = rf_prediction[['pred','prediction_1','prediction_2','prediction_3','prediction_4']].mean(axis=1)

rf_prediction = rf_prediction.fillna(0)
 
fpr, tpr, threshold = roc_curve(rf_prediction['inhibition'], rf_prediction['avg_prediction'])

plt.plot(fpr, tpr, 
         label = "RandomForest: AUROC = %0.2f" % roc_auc_score(rf_prediction['inhibition'], rf_prediction['avg_prediction']))


## Evidential DMPNN
hc_dmpnn_prediction = dmpnn_prediction[dmpnn_prediction['inhibition_dirichlet_uncal_uncertainty'] < 0.4]
hc_true_label = true_label[dmpnn_prediction['inhibition_dirichlet_uncal_uncertainty'] < 0.4]

fpr, tpr, threshold = roc_curve(hc_true_label['inhibition'], hc_dmpnn_prediction['inhibition'])

plt.plot(fpr, tpr, 
         label = "Evidential DMPNN: AUROC = %0.2f" % roc_auc_score(hc_true_label['inhibition'], hc_dmpnn_prediction['inhibition']))

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    # title=f"Mean ROC curve with variability",
)
ax.spines[['right','top']].set_visible(False)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('evaluation/inhibition_prediction/output/External_Validation_AUROC')

## AUPRC
plt.figure(figsize=(12.8,4.8))
ax = plt.subplot(111)

# DMPNN
precision, recall, thresholds = precision_recall_curve(true_label['inhibition'],
                                                       dmpnn_prediction['inhibition'])

plt.plot(recall, precision, 
         label = "DMPNN: AUPRC = %0.2f" % auc(recall, precision))

# EGNN
precision, recall, thresholds = precision_recall_curve(egnn_prediction['inhibition'],
                                                       egnn_prediction['avg_prediction'])

plt.plot(recall, precision, 
         label = "EGNN: AUPRC = %0.2f" % auc(recall, precision))

# FFNN
precision, recall, thresholds = precision_recall_curve(ffnn_prediction['inhibition'],
                                                       ffnn_prediction['avg_prediction'])

plt.plot(recall, precision, 
         label = "FFNN: AUPRC = %0.2f" % auc(recall, precision))

# Random Forest
precision, recall, thresholds = precision_recall_curve(rf_prediction['inhibition'],
                                                       rf_prediction['avg_prediction'])

plt.plot(recall, precision, 
         label = "RandomForest: AUPRC = %0.2f" % auc(recall, precision))

# Evidential
precision, recall, thresholds = precision_recall_curve(hc_true_label['inhibition'], hc_dmpnn_prediction['inhibition'])

plt.plot(recall, precision, 
         label = "Evidential DMPNN: AUPRC = %0.2f" % auc(recall, precision))

ax.set(
    xlabel="Recall",
    ylabel="Precision",
)

ax.spines[['right','top']].set_visible(False)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('evaluation/inhibition_prediction/output/External_Validation_AUPRC')
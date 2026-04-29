import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt

label = pd.read_csv('data/output/dragonfold/screening_result/screening_labels.csv')
prediction_1 = pd.read_csv('data/output/dragonfold/screening_result/raw_inhibition_prediction_1.csv')
prediction_2 = pd.read_csv('data/output/dragonfold/screening_result/raw_inhibition_prediction_2.csv')

egnn_prediction = pd.read_csv('data/egnn/evaluation/external_predictions/bench_EGNN_fold_0_predictions.csv')
ffnn_prediction = pd.read_csv('data/egnn/evaluation/external_predictions/bench_FFNN_fold_0_predictions.csv')
rf_prediction = pd.read_csv('evaluation/inhibition_prediction/random_forest/output/bench_pred.csv')
transformerM_prediction = pd.read_csv('data/ML_Drug_Discovery/evaluation/bench_pred.csv')

# print(roc_auc_score(label['inhibition'],
#                     prediction_1['inhibition']))

prediction_2_hc = prediction_2[prediction_2['inhibition_dirichlet_uncal_uncertainty'] <= 0.5]
label_hc = label[label['SMILES'].isin(prediction_2_hc['SMILES'])]

print(roc_auc_score(label['inhibition'],
                    prediction_2['inhibition']))

print(roc_auc_score(label_hc['inhibition'],
                    prediction_2_hc['inhibition']))

print(roc_auc_score(label['inhibition'],
                    egnn_prediction['prediction']))

print(roc_auc_score(label['inhibition'],
                    ffnn_prediction['prediction']))

print(roc_auc_score(label['inhibition'],
                    rf_prediction['prediction']))

print(roc_auc_score(label['inhibition'],
                    transformerM_prediction['prediction']))

models = [prediction_2['inhibition'],egnn_prediction['prediction'],ffnn_prediction['prediction']]

plt.rcParams['axes.labelsize'] = 20   # Axis labels
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 20
plt.figure(figsize=(10,10))
ax = plt.subplot(111)

# DMPNN
fpr, tpr, thresholds = roc_curve(label['inhibition'], prediction_2['inhibition'])
youdens_index = tpr - fpr 
best_index = youdens_index.argmax()
best_threshold = thresholds[best_index]
print(best_threshold)
plt.plot(fpr, tpr,
         label = "DMPNN: AUROC = %.2f" % roc_auc_score(label['inhibition'], prediction_2['inhibition']))

# EGNN
fpr, tpr, thresholds = roc_curve(label['inhibition'], egnn_prediction['prediction'])
plt.plot(fpr, tpr,
         label = "EGNN: AUROC = %.2f" % roc_auc_score(label['inhibition'], egnn_prediction['prediction']))

# FFNN
fpr, tpr, thresholds = roc_curve(label['inhibition'], ffnn_prediction['prediction'])
plt.plot(fpr, tpr,
         label = "FFNN: AUROC = %.2f" % roc_auc_score(label['inhibition'], ffnn_prediction['prediction']))

# Transformer-M
fpr, tpr, thresholds = roc_curve(label['inhibition'], transformerM_prediction['prediction'])
plt.plot(fpr, tpr,
         label = "Transformer-M: AUROC = %.2f" % roc_auc_score(label['inhibition'], transformerM_prediction['prediction']))

# Random Forest
fpr, tpr, thresholds = roc_curve(label['inhibition'], rf_prediction['prediction'])
plt.plot(fpr, tpr,
         label = "Random Forest: AUROC = %.2f" % roc_auc_score(label['inhibition'], rf_prediction['prediction']))

# Evidential DMPNN
fpr, tpr, thresholds = roc_curve(label_hc['inhibition'], prediction_2_hc['inhibition'])
plt.plot(fpr, tpr,
         label = "Evidential DMPNN: AUROC = %.2f" % roc_auc_score(label_hc['inhibition'], prediction_2_hc['inhibition']))

plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    # title=f"Mean ROC curve with variability",
)
ax.spines[['right','top']].set_visible(False)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('evaluation/inhibition_prediction/output/bench_validation_AUROC.png',bbox_inches='tight')

  
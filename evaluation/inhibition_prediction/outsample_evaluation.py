import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, accuracy_score, precision_recall_curve, auc

outsample_prediction_dir = 'evaluation/inhibition_prediction/output/new_outsample_prediction.csv'
outsample_prediction = pd.read_csv(outsample_prediction_dir)

outsample_label = pd.read_csv('data/validation_dataset/outsample_data.csv')

inhibition_threshold = 0.7
confidence_threshold = 0.55

outsample_prediction['label'] = outsample_label['inhibition']
outsample_prediction['confidence'] = 1 - outsample_prediction['inhibition_dirichlet_uncal_uncertainty']
outsample_prediction['prediction'] = np.where(outsample_prediction['inhibition'] > inhibition_threshold, 1, 0)

# Calculate for full performance
tn, fp, fn, tp = confusion_matrix(outsample_prediction['label'],outsample_prediction['prediction']).ravel()
print(tp)
specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn) 

precision, recall, _ = precision_recall_curve(outsample_prediction['label'], outsample_prediction['inhibition'])

print("Overall AUROC:", roc_auc_score(outsample_prediction['label'],outsample_prediction['inhibition']))
print("Overall Accuracy:", accuracy_score(outsample_prediction['label'],outsample_prediction['prediction']))
print("Overall Sensitivity: ", sensitivity)
print("Overall Specificity: ", specificity)
print("Overall AUPRC: ", auc(recall, precision))

# Calculate for High Conf Performance
high_conf_df = outsample_prediction[outsample_prediction['confidence']>confidence_threshold]
tn, fp, fn, tp = confusion_matrix(high_conf_df['label'],high_conf_df['prediction']).ravel()
print(tp)
specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn) 
# specificity = tn/len(outsample_prediction[outsample_prediction['label']==0])
# sensitivity = tp/len(outsample_prediction[outsample_prediction['label']==1])

precision, recall, _ = precision_recall_curve(high_conf_df['label'], high_conf_df['prediction'])

print("High Confidence AUROC:", roc_auc_score(high_conf_df['label'],high_conf_df['inhibition']))
print("High Confidence Accuracy:", accuracy_score(high_conf_df['label'],high_conf_df['prediction']))
print("High Confidence Sensitivity: ", sensitivity)
print("High Confidence Specificity: ", specificity)
print("High Confidence AUPRC: ", auc(recall, precision))

print(len(outsample_label))
print(high_conf_df[high_conf_df['prediction']==1])
print(high_conf_df[high_conf_df['label']==1])
print(high_conf_df[(high_conf_df['prediction']==1) & (high_conf_df['label']==1)])


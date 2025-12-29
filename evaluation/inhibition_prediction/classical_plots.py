import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

models = ['FFNN','DMPNN','EGNN','RandomForest']

auroc = [0.7129, 0.7586, 0.7064,0.5883]
auroc_err = [0.0130, 0.0090, 0.0078,0.0629]
auprc = [0.1409, 0.1781, 0.1364,0.1411]
auprc_err = [0.0107, 0.0100, 0.0091,0.0567]
auspc = []
auspc_err = []

plt.figure(figsize=(6.4,4.8))
ax = plt.subplot(111)
plt.bar(models, 
        auroc, 0.5, 
        label = 'DMPNN',
        color=['lightsteelblue','lightcoral','darkseagreen','darkorange'])
plt.errorbar(models,
             auroc,
             yerr=auroc_err,
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.ylim(0.5,1.0)
plt.ylabel("AUROC")
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/classic_performance_AUROC.png')

plt.figure(figsize=(6.4,4.8))
ax = plt.subplot(111)
plt.bar(models, 
        auprc, 0.5, 
        label = 'DMPNN',
        color=['lightsteelblue','lightcoral','darkseagreen','darkorange'])
plt.errorbar(models,
             auprc,
             yerr=auprc_err,
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.ylim(0.0,0.5)
plt.ylabel("AUPRC")
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/classic_performance_AUPRC.png')
plt.close()

# Filtered AUPRC
plt.figure()
ax = plt.subplot(111)
for fold in np.arange(0,5,1):
    precision_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_precision_df.csv')
    recall_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_recall_df.csv')

    precision = precision_df.iloc[1,:].to_list()
    recall = recall_df.iloc[1,:].to_list()

    plt.plot(recall,
             precision,
             marker='.',
             markersize=1,
             color='darkgray')

plt.xlabel('Recall')
plt.ylabel('Precision')

ax.text(0.61,0.96, 
        'AUPRC = 0.2393 +/- 0.0588',
        verticalalignment='top')

ax.spines[['right','top']].set_visible(False)
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/AUPRC_Plot_DMPNN_ConfidenceFiltering=0.8.png')
plt.close()

# Raw AUPRC
plt.figure()
ax = plt.subplot(111)
for fold in np.arange(0,5,1):
    precision_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_precision_df.csv')
    recall_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_recall_df.csv')

    precision = precision_df.iloc[-1,:].to_list()
    recall = recall_df.iloc[-1,:].to_list()

    plt.plot(recall,
             precision,
             marker='.',
             markersize=1,
             color='darkgray')

plt.xlabel('Recall')
plt.ylabel('Precision')

ax.text(0.61,0.96, 
        'AUPRC = 0.1781 +/- 0.0100',
        verticalalignment='top')

ax.spines[['right','top']].set_visible(False)
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/AUPRC_Plot_DMPNN_Classic.png')
plt.close()

# Raw AUROC
plt.figure()
ax = plt.subplot(111)
for fold in np.arange(0,5,1):
    tpr_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_tpr_df.csv')
    fpr_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_fpr_df.csv')

    tpr = tpr_df.iloc[-1,:].to_list()
    fpr = fpr_df.iloc[-1,:].to_list()

    plt.plot(fpr,
             tpr,
             color='darkgray')
    
plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

ax.text(0.05,0.96, 
        'AUROC = 0.7586 +/- 0.0090',
        verticalalignment='top')

ax.spines[['right','top']].set_visible(False)
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/AUROC_Plot_DMPNN_Classic.png')
plt.close()

# Filtered AUROC
plt.figure()
ax = plt.subplot(111)
for fold in np.arange(0,5,1):
    tpr_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_tpr_df.csv')
    fpr_df = pd.read_csv(f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/cv_evaluation/inhibition_dmpnn/fold_{fold}/evidential_fpr_df.csv')

    tpr = tpr_df.iloc[1,:].to_list()
    fpr = fpr_df.iloc[1,:].to_list()

    plt.plot(fpr,
             tpr,
             color='darkgray')
    
plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

ax.text(0.05,0.96, 
        'AUROC = 0.8200 +/- 0.0617',
        verticalalignment='top')

ax.spines[['right','top']].set_visible(False)
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/AUROC_Plot_DMPNN_ConfidenceFiltering=0.8.png')
plt.close()



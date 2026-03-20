import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

task = ['Cross-Validation', 'External Validation']

# Pre-computed summary statistics from 5-fold cross-validation and external validation runs.
# Cross-validation values are mean ± std over 5 folds; external validation values are from
# evaluation/inhibition_prediction/bench_validation/evaluation.py.
dmpnn_mean_auroc = [0.7586,0.7316]
dmpnn_auroc_err = [0.0090,0.0120]
dmpnn_mean_auprc = [0.1781,0.2412]
dmpnn_auprc_err = [0.0100,0.0144]

# Evidential DMPNN values use confidence filter = 0.6 (uncertainty threshold = 0.4).
evidential_mean_auroc = [0.7903,0.7652]
evidential_auroc_err = [0.0263,0.0282]
evidential_mean_auprc = [0.1993,0.2818]
evidential_auprc_err = [0.0123,0.0156]

x_axis = np.arange(0,0.5,0.25)
plt.figure(figsize=(3.2,4.8))
ax = plt.subplot(111)
plt.bar(x_axis - 0.05, 
        dmpnn_mean_auroc, 0.1, 
        label = 'DMPNN',
        color='lightsteelblue')
plt.errorbar(x_axis - 0.05,
             dmpnn_mean_auroc,
             yerr=dmpnn_auroc_err,
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.bar(x_axis + 0.05, 
        evidential_mean_auroc, 0.1, 
        label = 'Evidential DMPNN with Confidence Filter = 0.6',
        color='lightcoral')
plt.errorbar(x_axis + 0.05,
             evidential_mean_auroc,
             yerr=evidential_auroc_err,
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.xticks(x_axis, task, fontsize=8)
plt.ylim(0.5,1.0)
plt.ylabel("AUROC")
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/evidential_performance_comparison_AUROC.png')

plt.figure(figsize=(3.2,4.8))
ax = plt.subplot(111)
plt.bar(x_axis - 0.05, 
        dmpnn_mean_auprc, 0.1, 
        label = 'DMPNN',
        color='lightsteelblue')
plt.errorbar(x_axis - 0.05,
             dmpnn_mean_auprc,
             yerr=dmpnn_auprc_err,
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.bar(x_axis + 0.05, 
        evidential_mean_auprc, 0.1, 
        label = 'Evidential DMPNN',
        color='lightcoral')
plt.errorbar(x_axis + 0.05,
             evidential_mean_auprc,
             yerr=evidential_auprc_err,
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.xticks(x_axis, task, fontsize=8)
plt.ylim(0.0,0.5)
plt.ylabel("AUPRC")
plt.legend(loc = "upper center")
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/evidential_performance_comparison_AUPRC.png')
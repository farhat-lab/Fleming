import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, PrecisionRecallDisplay, precision_recall_curve
from matplotlib import pyplot as plt
prediction_df = pd.read_csv('data/model_checkpoints/Transformer-M/fold_4/val_pred_2.csv')

# DMPNN
dmpnn_tprs = []
dmpnn_aucs = []
mean_dmpnn_fpr = np.linspace(0, 1, 100)

fig_si, ax_si = plt.subplots()
plt.figure(figsize=(12.8,4.8))
ax = plt.subplot(111)
size = 12
plt.rc('font', size=size)          # controls default text sizes
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title

plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')

# for fold in range(0,5):

viz = RocCurveDisplay.from_predictions(
    prediction_df['inhibition'],
    prediction_df['pred'],
    # name=f"ROC fold {fold}",
    alpha=0.3,
    lw=1,
    ax=ax_si,
    # plot_chance_level=(fold == 5 - 1),
)
interp_tpr = np.interp(mean_dmpnn_fpr, viz.fpr, viz.tpr)
interp_tpr[0] = 0.0
dmpnn_tprs.append(interp_tpr)
dmpnn_aucs.append(viz.roc_auc)

mean_tpr = np.mean(dmpnn_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_dmpnn_fpr, mean_tpr)
std_auc = np.std(dmpnn_aucs)
ax.plot(
    mean_dmpnn_fpr,
    mean_tpr,
    # color="b",
    label=r"DMPNN (AUROC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(dmpnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_dmpnn_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)
plt.legend()
plt.savefig('data/ML_Drug_Discovery/evaluation/cv_auroc_plot.png')
# print(pos_df)
# print(neg_df)

plt.figure()
for fold in range(0,5):
    prediction_df = pd.read_csv('data/model_checkpoints/Transformer-M/fold_4/val_pred_2.csv')
    

    precision, recall, thresholds = precision_recall_curve(prediction_df['inhibition'],
                                                           prediction_df['pred'])
    
    auprc = auc(recall, precision)

    if fold == 0:
        plt.plot(recall, 
                precision,
                color = 'blue',
                label = r'DMPNN: AUPRC = 0.18 $\pm$ 0.01')
    else:
        plt.plot(recall, 
                precision,
                color = 'blue')

plt.legend()
plt.savefig('data/ML_Drug_Discovery/evaluation/cv_auprc_plot.png')
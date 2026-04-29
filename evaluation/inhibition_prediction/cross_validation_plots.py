import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, PrecisionRecallDisplay, precision_recall_curve
import seaborn as sns

line_width = 5
colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00"]
model_order = ['DMPNN', 'EGNN', 'FFNN', 'Transformer-M', 'Random Forest', 'Evidential DMPNN']
display_names = ['DMPNN', 'EGNN', 'FFNN', 'Transformer-M', 'Random Forest', 'Evidential DMPNN']
palette_dict = {model: color for model, color in zip(display_names, colors)}
colors = palette_dict

### DMPNN vs. EGNN vs. FFNN
dmpnn_prediction_dir = 'data/cv_evaluation/inhibition_dmpnn/'
egnn_prediction_dir = 'data/cv_evaluation/inhibition_egnn/'
ffnn_prediction_dir = 'data/cv_evaluation/inhibition_ffnn/'

## AUROC
plt.rcParams['axes.labelsize'] = 20   # Axis labels
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 20

fig_si, ax_si = plt.subplots()
plt.figure(figsize=(9.6,7.2))
ax = plt.subplot(111)

for spine in ax.spines.values():
    spine.set_linewidth(line_width)

plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--', lw=line_width)

# Evidential
dmpnn_tprs = []
dmpnn_aucs = []
mean_dmpnn_fpr = np.linspace(0, 1, 100)

for fold in range(0,5):
    prediction_df = pd.read_csv(dmpnn_prediction_dir + f"fold_{fold}/test_pred.csv")
    true_label_df = pd.read_csv(f'data/final_checkpoints/Full_TB_Dataset_scaffold_dropout=0.2_cv=5/fold_{fold}/test_data.csv')

    assert len(set(prediction_df['SMILES']).intersection(set(true_label_df['SMILES']))) == len(prediction_df), "Prediction and True Label not aligned."

    prediction_df = prediction_df[prediction_df['inhibition_dirichlet_uncal_uncertainty']<0.4]
    true_label_df = true_label_df[true_label_df['SMILES'].isin(prediction_df['SMILES'])]

    viz = RocCurveDisplay.from_predictions(
        true_label_df['inhibition'],
        prediction_df['inhibition'],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax_si,
        plot_chance_level=(fold == 5 - 1),
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
    color= colors['Evidential DMPNN'],
    label=r"Evidential DMPNN - %0.2f[%0.2f - %0.2f]" % (mean_auc, mean_auc - std_auc, mean_auc + std_auc),
    lw=line_width,
    alpha=0.8,
)

std_tpr = np.std(dmpnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_dmpnn_fpr,
    tprs_lower,
    tprs_upper,
    color=colors['Evidential DMPNN'],
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)

# DMPNN
dmpnn_tprs = []
dmpnn_aucs = []
mean_dmpnn_fpr = np.linspace(0, 1, 100)

for fold in range(0,5):
    prediction_df = pd.read_csv(dmpnn_prediction_dir + f"fold_{fold}/test_pred.csv")
    true_label_df = pd.read_csv(f'data/final_checkpoints/Full_TB_Dataset_scaffold_dropout=0.2_cv=5/fold_{fold}/test_data.csv')

    assert len(set(prediction_df['SMILES']).intersection(set(true_label_df['SMILES']))) == len(prediction_df), "Prediction and True Label not aligned."

    viz = RocCurveDisplay.from_predictions(
        true_label_df['inhibition'],
        prediction_df['inhibition'],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax_si,
        plot_chance_level=(fold == 5 - 1),
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
    color=colors['DMPNN'],
    label=r"DMPNN - %0.2f[%0.2f - %0.2f]" % (mean_auc, mean_auc - std_auc, mean_auc + std_auc),
    lw=line_width,
    alpha=0.8,
)

std_tpr = np.std(dmpnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_dmpnn_fpr,
    tprs_lower,
    tprs_upper,
    color=colors['DMPNN'],
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)

# EGNN
egnn_tprs = []
egnn_aucs = []
mean_egnn_fpr = np.linspace(0, 1, 100)

for fold in range(0,5):
    prediction_df = pd.read_csv(egnn_prediction_dir + f"fold_{fold}/test_prediction.csv")

    viz = RocCurveDisplay.from_predictions(
        prediction_df['labels'],
        prediction_df['predictions'],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax_si,
        plot_chance_level=(fold == 5 - 1),
    )
    interp_tpr = np.interp(mean_egnn_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    egnn_tprs.append(interp_tpr)
    egnn_aucs.append(viz.roc_auc)

mean_tpr = np.mean(egnn_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_egnn_fpr, mean_tpr)
std_auc = np.std(egnn_aucs)
ax.plot(
    mean_egnn_fpr,
    mean_tpr,
    color=colors['EGNN'],
    label=r"EGNN - %0.2f[%0.2f - %0.2f]" % (mean_auc, mean_auc - std_auc, mean_auc + std_auc),
    lw=line_width,
    alpha=0.8,
)

std_tpr = np.std(egnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_egnn_fpr,
    tprs_lower,
    tprs_upper,
    color=colors['EGNN'],
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)

# FFNN
ffnn_tprs = []
ffnn_aucs = []
mean_ffnn_fpr = np.linspace(0, 1, 100)

for fold in range(0,5):
    prediction_df = pd.read_csv(ffnn_prediction_dir + f"fold_{fold}/test_prediction.csv")

    viz = RocCurveDisplay.from_predictions(
        prediction_df['labels'],
        prediction_df['predictions'],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax_si,
        plot_chance_level=(fold == 5 - 1),
    )
    interp_tpr = np.interp(mean_ffnn_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    ffnn_tprs.append(interp_tpr)
    ffnn_aucs.append(viz.roc_auc)

mean_tpr = np.mean(ffnn_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_ffnn_fpr, mean_tpr)
std_auc = np.std(ffnn_aucs)
ax.plot(
    mean_ffnn_fpr,
    mean_tpr,
    color=colors['FFNN'],
    label=r"FFNN - %0.2f[%0.2f - %0.2f]" % (mean_auc, mean_auc - std_auc, mean_auc + std_auc),
    lw=line_width,
    alpha=0.8,
)

std_tpr = np.std(ffnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_ffnn_fpr,
    tprs_lower,
    tprs_upper,
    color=colors['FFNN'],
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)

# Transformer-M
fold_pred_df_list = [# 'data/model_checkpoints/Transformer-M/fold_0/val_pred.csv',
                     # 'data/model_checkpoints/Transformer-M/fold_1/val_pred.csv',
                     'data/model_checkpoints/Transformer-M/fold_2/val_pred_11.csv',
                     'data/model_checkpoints/Transformer-M/fold_3/val_pred_28.csv',
                     'data/model_checkpoints/Transformer-M/fold_4/val_pred_16.csv']

ffnn_tprs = []
ffnn_aucs = []
mean_ffnn_fpr = np.linspace(0, 1, 100)

# fig_si, ax_si = plt.subplots()
# fig, ax = plt.subplots(figsize=(6, 6))
for fold in range(0,3):
    prediction_df = pd.read_csv(fold_pred_df_list[fold])
    # true_label_df = pd.read_csv(f'data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')
    # true_label_df = true_label_df[true_label_df['SMILES'].isin(prediction_df['SMILES'])]

    # assert len(set(prediction_df['SMILES']).intersection(set(true_label_df['SMILES']))) == len(prediction_df), "Prediction and True Label not aligned."

    viz = RocCurveDisplay.from_predictions(
        prediction_df['inhibition'],
        prediction_df['pred'],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax_si,
        plot_chance_level=(fold == 5 - 1),
    )
    interp_tpr = np.interp(mean_ffnn_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    ffnn_tprs.append(interp_tpr)
    ffnn_aucs.append(viz.roc_auc)

mean_tpr = np.mean(ffnn_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_ffnn_fpr, mean_tpr)
std_auc = np.std(ffnn_aucs)
ax.plot(
    mean_ffnn_fpr,
    mean_tpr,
    color=colors['Transformer-M'],
    label=rf"Transformer-M - %0.2f[%0.2f - %0.2f]" % (mean_auc, mean_auc - std_auc, mean_auc + std_auc),
    lw=line_width,
    alpha=0.8,
)

std_tpr = np.std(ffnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_ffnn_fpr,
    tprs_lower,
    tprs_upper,
    color=colors['Transformer-M'],
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)

# Random Forest
ffnn_tprs = []
ffnn_aucs = []
mean_ffnn_fpr = np.linspace(0, 1, 100)

# fig_si, ax_si = plt.subplots()
# fig, ax = plt.subplots(figsize=(6, 6))
for fold in range(0,5):
    prediction_df = pd.read_csv('evaluation/inhibition_prediction/random_forest/output/' + f"fold{fold}_test_predictions.csv")
    # true_label_df = pd.read_csv(f'data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')
    # true_label_df = true_label_df[true_label_df['SMILES'].isin(prediction_df['SMILES'])]

    # assert len(set(prediction_df['SMILES']).intersection(set(true_label_df['SMILES']))) == len(prediction_df), "Prediction and True Label not aligned."

    viz = RocCurveDisplay.from_predictions(
        prediction_df['inhibition'],
        prediction_df['pred'],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax_si,
        plot_chance_level=(fold == 5 - 1),
    )
    interp_tpr = np.interp(mean_ffnn_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    ffnn_tprs.append(interp_tpr)
    ffnn_aucs.append(viz.roc_auc)

mean_tpr = np.mean(ffnn_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_ffnn_fpr, mean_tpr)
std_auc = np.std(ffnn_aucs)
ax.plot(
    mean_ffnn_fpr,
    mean_tpr,
    color= colors['Random Forest'],
    label=rf"Random Forest - %0.2f[%0.2f - %0.2f]" % (mean_auc, mean_auc - std_auc, mean_auc + std_auc),
    lw=line_width,
    alpha=0.8,
)

std_tpr = np.std(ffnn_tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_ffnn_fpr,
    tprs_lower,
    tprs_upper,
    color=colors['Random Forest'],
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)

# Plot
ax.set_xlabel("False Positive Rate", 
              fontsize=30, 
              # fontweight='bold'
              )
ax.set_ylabel("True Positive Rate", 
              fontsize=30, 
              # fontweight='bold'
              )
ax.spines[['right','top']].set_visible(False)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Make legend texts bold and increase font size
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(20)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
# plt.legend(bbox_to_anchor = (2.2, 0.6), loc='center right')
# plt.tight_layout()
# plt.savefig('evaluation/inhibition_prediction/output/Cross_Validation_AUROC.png',
#             bbox_inches='tight')
plt.savefig('evaluation/inhibition_prediction/output/Cross_Validation_AUROC_evi.png',
            bbox_inches='tight')
plt.savefig('evaluation/inhibition_prediction/output/Cross_Validation_AUROC_evi.pdf',
            bbox_inches='tight')

## AUPRC
plt.rcParams['axes.labelsize'] = 20   # Axis labels
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 20

fig, ax = plt.subplots(figsize=(9.6,7.2))

for spine in ax.spines.values():
    spine.set_linewidth(line_width)

# Evidential DMPNN
evi_dmpnn_auprcs = []
for fold in range(0,5):
    prediction_df = pd.read_csv(dmpnn_prediction_dir + f"fold_{fold}/test_pred.csv")
    true_label_df = pd.read_csv(f'data/final_checkpoints/Full_TB_Dataset_scaffold_dropout=0.2_cv=5/fold_{fold}/test_data.csv')

    prediction_df = prediction_df[prediction_df['inhibition_dirichlet_uncal_uncertainty'] < 0.4]
    true_label_df = true_label_df[true_label_df['SMILES'].isin(prediction_df['SMILES'])]

    precision, recall, thresholds = precision_recall_curve(true_label_df['inhibition'],
                                                           prediction_df['inhibition'])

    auprc = auc(recall, precision)
    evi_dmpnn_auprcs.append(auprc)

    ax.plot(recall,
            precision,
            color = colors['Evidential DMPNN'])

ax.plot([], [],
        color = colors['Evidential DMPNN'],
        label = r'Evidential DMPNN: AUPRC = %.2f $\pm$ %.2f' % (np.mean(evi_dmpnn_auprcs), np.std(evi_dmpnn_auprcs)))

# DMPNN
dmpnn_auprcs = []
for fold in range(0,5):
    prediction_df = pd.read_csv(dmpnn_prediction_dir + f"fold_{fold}/test_pred.csv")
    true_label_df = pd.read_csv(f'data/final_checkpoints/Full_TB_Dataset_scaffold_dropout=0.2_cv=5/fold_{fold}/test_data.csv')

    precision, recall, thresholds = precision_recall_curve(true_label_df['inhibition'],
                                                           prediction_df['inhibition'])

    auprc = auc(recall, precision)
    dmpnn_auprcs.append(auprc)

    ax.plot(recall,
            precision,
            color = colors['DMPNN'])

ax.plot([], [],
        color = colors['DMPNN'],
        label = r'DMPNN: AUPRC = %.2f $\pm$ %.2f' % (np.mean(dmpnn_auprcs), np.std(dmpnn_auprcs)))
    
# EGNN
egnn_auprcs = []
for fold in range(0,5):
    prediction_df = pd.read_csv(egnn_prediction_dir + f"fold_{fold}/test_prediction.csv")

    precision, recall, thresholds = precision_recall_curve(prediction_df['labels'],
                                                           prediction_df['predictions'])

    auprc = auc(recall, precision)
    egnn_auprcs.append(auprc)

    ax.plot(recall,
            precision,
            color = colors['EGNN'])

ax.plot([], [],
        color = colors['EGNN'],
        label = r'EGNN: AUPRC = %.2f $\pm$ %.2f' % (np.mean(egnn_auprcs), np.std(egnn_auprcs)))

# FFNN
ffnn_auprcs = []
for fold in range(0,5):
    prediction_df = pd.read_csv(ffnn_prediction_dir + f"fold_{fold}/test_prediction.csv")

    precision, recall, thresholds = precision_recall_curve(prediction_df['labels'],
                                                           prediction_df['predictions'])

    auprc = auc(recall, precision)
    ffnn_auprcs.append(auprc)
    ax.plot(recall,
            precision,
            color = colors['FFNN'])

ax.plot([], [],
        color = colors['FFNN'],
        label = r'FFNN: AUPRC = %.2f $\pm$ %.2f' % (np.mean(ffnn_auprcs), np.std(ffnn_auprcs)))

# Transformer-M (only 3 folds available: fold_2, fold_3, fold_4)
transformerm_auprcs = []
for fold in range(0,3):
    prediction_df = pd.read_csv(fold_pred_df_list[fold])

    precision, recall, thresholds = precision_recall_curve(prediction_df['inhibition'],
                                                           prediction_df['pred'])

    auprc = auc(recall, precision)
    transformerm_auprcs.append(auprc)
    ax.plot(recall,
            precision,
            color = colors['Transformer-M'])

ax.plot([], [],
        color = colors['Transformer-M'],
        label = r'Transformer-M: AUPRC = %.2f $\pm$ %.2f' % (np.mean(transformerm_auprcs), np.std(transformerm_auprcs)))

# Random Forest
rf_auprcs = []
for fold in range(0,5):
    prediction_df = pd.read_csv('evaluation/inhibition_prediction/random_forest/output/' + f"fold{fold}_test_predictions.csv")

    precision, recall, thresholds = precision_recall_curve(prediction_df['inhibition'],
                                                           prediction_df['pred'])

    auprc = auc(recall, precision)
    rf_auprcs.append(auprc)
    ax.plot(recall,
            precision,
            color = colors['Random Forest'])

ax.plot([], [],
        color = colors['Random Forest'],
        label = r'Random Forest: AUPRC = %.2f $\pm$ %.2f' % (np.mean(rf_auprcs), np.std(rf_auprcs)))
        
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("Recall", 
              fontsize=30, 
              # fontweight='bold'
              )
ax.set_ylabel("Precision", 
              fontsize=30, 
              # fontweight='bold'
              )
ax.spines[['right','top']].set_visible(False)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Make legend texts bold and increase font size
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(20)

plt.savefig('evaluation/inhibition_prediction/output/Cross_Validation_AUPRC.png',
            bbox_inches='tight')
plt.savefig('evaluation/inhibition_prediction/output/Cross_Validation_AUPRC.pdf',
            bbox_inches='tight')
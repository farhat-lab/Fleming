import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

evidential_threshold = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

dmpnn_auroc = [0.75864,0.75864,0.75868,0.75868,0.75954,0.76824,0.7903,0.81074,0.82002]
dmpnn_auroc_std = [0.00899, 0.00899, 0.00902, 0.00902, 0.00988, 0.01683, 0.02632, 0.03958, 0.06165]
dmpnn_auprc = [0.17814,0.17814,0.17816,0.17816,0.17864,0.18402,0.19938,0.23768,0.23926]
dmpnn_auprc_std = [0.00998, 0.00998, 0.00999, 0.00999, 0.00994, 0.00984, 0.01298, 0.03805, 0.05878]

egnn_auroc = [0.70638, 0.70864, 0.71222, 0.72184, 0.73782, 0.75456, 0.7719, 0.7801, 0.75358]
egnn_auroc_std = [0.00781, 0.00720, 0.00677, 0.00672, 0.00924, 0.01091, 0.01513, 0.02959, 0.06495]
egnn_auprc = [0.13644, 0.13674, 0.13854, 0.14332, 0.15222, 0.16288, 0.17544, 0.18408, 0.17608]
egnn_auprc_std = [0.00908, 0.00915, 0.00935, 0.00985, 0.01165, 0.01280, 0.01432, 0.02003, 0.03525]

ffnn_auroc = [0.71286, 0.73084, 0.74214, 0.75246, 0.763, 0.76314, 0.75004, 0.72716, 0.6886]
ffnn_auroc_std = [0.01299, 0.01437, 0.01864, 0.02055, 0.02102, 0.04090, 0.05659, 0.08310, 0.13528]
ffnn_auprc = [0.1409, 0.15078, 0.15732, 0.16418, 0.17274, 0.18054, 0.18588, 0.1917, 0.1902]
ffnn_auprc_std = [0.01071, 0.01261, 0.01515, 0.01732, 0.02132, 0.03678, 0.07065, 0.14067, 0.22010]

### AUROC Plot
plt.figure(figsize=(6.4,3.2))
ax=plt.subplot(111)
plt.plot(evidential_threshold,
         dmpnn_auroc,
         label = 'DMPNN',
         linewidth = 2.5)
plt.fill_between(
    evidential_threshold,
    np.array(dmpnn_auroc) - np.array(dmpnn_auroc_std),
    np.array(dmpnn_auroc) + np.array(dmpnn_auroc_std),
    color="lightsteelblue",
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)
# plt.plot(evidential_threshold,
#          egnn_auroc,
#          label = 'EGNN',
#          linewidth = 2.5)
# plt.fill_between(
#     evidential_threshold,
#     np.array(egnn_auroc) - np.array(egnn_auroc_std),
#     np.array(egnn_auroc) + np.array(egnn_auroc_std),
#     color="moccasin",
#     alpha=0.2,
#     # label=r"$\pm$ 1 std. dev.",
# )
# plt.plot(evidential_threshold,
#          ffnn_auroc,
#          label = 'FFNN',
#          linewidth = 2.5)
# plt.fill_between(
#     evidential_threshold,
#     np.array(ffnn_auroc) - np.array(ffnn_auroc_std),
#     np.array(ffnn_auroc) + np.array(ffnn_auroc_std),
#     color="lightgreen",
#     alpha=0.2,
#     # label=r"$\pm$ 1 std. dev.",
# )

plt.ylim(0.6,1)
plt.xlabel('Confidence Threshold')
plt.ylabel('AUROC')
plt.legend(bbox_to_anchor = (1.25, 0.6), loc='center right')
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/evidential_performance_change_auroc.png')

### AUPRC Plot
plt.figure(figsize=(6.4,3.2))
ax=plt.subplot(111)
plt.plot(evidential_threshold,
         dmpnn_auprc,
         label = 'DMPNN',
         linewidth = 2.5)
plt.fill_between(
    evidential_threshold,
    np.array(dmpnn_auprc) - np.array(dmpnn_auprc_std),
    np.array(dmpnn_auprc) + np.array(dmpnn_auprc_std),
    color="lightsteelblue",
    alpha=0.2,
    # label=r"$\pm$ 1 std. dev.",
)
# plt.plot(evidential_threshold,
#          egnn_auprc,
#          label = 'EGNN',
#          linewidth = 2.5)
# plt.fill_between(
#     evidential_threshold,
#     np.array(egnn_auprc) - np.array(egnn_auprc_std),
#     np.array(egnn_auprc) + np.array(egnn_auprc_std),
#     color="moccasin",
#     alpha=0.2,
#     # label=r"$\pm$ 1 std. dev.",
# )
# plt.plot(evidential_threshold,
#          ffnn_auprc,
#          label = 'FFNN',
#          linewidth = 2.5)
# plt.fill_between(
#     evidential_threshold,
#     np.array(ffnn_auprc) - np.array(ffnn_auprc_std),
#     np.array(ffnn_auprc) + np.array(ffnn_auprc_std),
#     color="lightgreen",
#     alpha=0.2,
#     # label=r"$\pm$ 1 std. dev.",
# )
plt.ylim(0.0,0.4)
plt.xlabel('Confidence Threshold')
plt.ylabel('AUPRC')
plt.legend(bbox_to_anchor = (1.25, 0.6), loc='center right')
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/evidential_performance_change_auprc.png')
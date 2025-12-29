import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# === Settings to match your style ===
line_width = 5
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.linewidth'] = line_width

# Colorblind-safe palette (edit as desired)
colors = ["#000000", "#E69F00", "#56B4E9"]
display_names = ['Fleming', 'o4-mini', 'o4-mini + Web Search']
palette_dict = {name: color for name, color in zip(display_names, colors)}

# Read data
df = pd.read_csv('evaluation/fleming/results_of_updated_distinguish_09_03_2025.csv')

# Model definitions: (display_name, [replicate column names])
# model_replicates = [
#     ('Fleming', ['Fleming Rating_x', 'Fleming Rating_y', 'Fleming Rating']),
#     ('o4-mini', ['o4 mini Rating_x', 'o4 mini Rating_y', 'o4 mini Rating']),
#     ('o4-mini + Web Search', ['o4mini ws Rating_x', 'o4mini ws Rating_y', 'o4mini ws Rating'])
# ]
model_replicates = [
    ('Fleming', ['rating_fleming_1', 'rating_fleming_2', 'rating_fleming_3']),
    ('o4-mini', ['rating_o4mini_1', 'rating_o4mini_2', 'rating_o4mini_3']),
    ('o4-mini + Web Search', ['rating_o4mini_ws1', 'rating_o4mini_ws2', 'rating_o4mini_ws3'])
]

mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(9.6, 7.2))
ax = plt.gca()

# Plot diagonal
plt.plot([0, 1], [0, 1], 'k--', lw=line_width, alpha=0.5)

for name, cols in model_replicates:
    tprs = []
    aucs = []
    for col in cols:
        y_true = df['Y'].values
        y_score = df[col].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure starts at 0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr,
            label=f'{name} \n ({mean_auc:.2f} ± {std_auc:.2f})',
            color=palette_dict[name],
            lw=line_width)

    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, 
                   color=palette_dict[name], alpha=0.2, linewidth=0)

# Axis styling
for spine in ax.spines.values():
    spine.set_linewidth(line_width)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
legend = ax.legend(loc='center left', 
                    bbox_to_anchor=(0.6, 0.3), 
                    frameon=False)
for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)

ax.spines[['right','top']].set_visible(False)
plt.savefig('evaluation/fleming/fleming_distinguish_roc_curve.png', 
            dpi=300, 
            bbox_inches='tight')
plt.savefig('evaluation/fleming/fleming_distinguish_roc_curve.pdf', 
            bbox_inches='tight')
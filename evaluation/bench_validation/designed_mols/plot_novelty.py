import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load data
maximum_sim_df = pd.read_csv('evaluation/bench_validation/designed_mols/max_similarities_new_antibiotics.csv')
maximum_sim_df['type'] = maximum_sim_df['type'].str.strip()  # Clean up whitespace

# Prepare Data
data_for_box = maximum_sim_df[maximum_sim_df['type'].isin(['new_anti', 'ours'])].copy()
data_for_box['Group'] = data_for_box['type'].replace({
    'new_anti': 'Preclinical and Phase 1 Antibiotics for MTB',
    'ours': 'Fleming-Designed Molecules'
})

# Calculate P-value (Mann-Whitney U)
x = data_for_box[data_for_box['Group'] == 'Preclinical and Phase 1 Antibiotics for MTB']['max_similarity']
y = data_for_box[data_for_box['Group'] == 'Fleming-Designed Molecules']['max_similarity']
stat, pvalue = mannwhitneyu(x, y, alternative='two-sided')
print(f"Mann-Whitney U p-value: {pvalue:.4e}")

# Set your custom plotting parameters
plt.rcParams['axes.labelsize'] = 20   # Axis labels
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 20

plt.figure(figsize=(9.6,7.2))
ax = plt.subplot(111)
for spine in ax.spines.values():
    spine.set_linewidth(5)

# Boxplot
palette = ['orange', 'green']
sns.boxplot(x='Group', y='max_similarity', data=data_for_box,
            width=0.5, fliersize=8, linewidth=4,
            showmeans=True,
            meanprops={"marker":"o",
                       "markerfacecolor":"black",
                       "markeredgecolor":"black",
                       "markersize":12},
            medianprops={"color": "firebrick", "linewidth": 4},
            boxprops={"facecolor": "none", "edgecolor": "k"},
            whiskerprops={"linewidth": 3},
            capprops={"linewidth": 3},
            palette=palette, ax=ax,
            )

# Means and error bars (mean±SD)
# group_stats = data_for_box.groupby('Group')['max_similarity'].agg(['mean', 'std'])
# for i, (group, row) in enumerate(group_stats.iterrows()):
#     ax.errorbar(i, row['mean'], yerr=row['std'], fmt='o', color='black', capsize=10, linewidth=4, markersize=0)

# X and Y limits
ax.set_ylim(0, 1)
ax.set_xlabel("",
              fontsize=20)
ax.set_ylabel("Maximum Pairwise Similarity", fontsize=20)
ax.set_xticklabels(['Preclinical and Phase 1 \n Antibiotics for MTB', 'Fleming-Designed \n Molecules'], fontweight='bold')
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Remove right, top, left spines for style (keep bottom for x-axis)
ax.spines[['right', 'top']].set_visible(False)

# Optional: Add p-value text above the boxes
# y_max = group_stats['mean'].max() + group_stats['std'].max() + 0.05
# ax.text(0.5, y_max, f'p = {pvalue:.2e}',
#         ha='center', va='bottom', fontsize=24, fontweight='bold')

plt.savefig('evaluation/bench_validation/designed_mols/output/maximum_similarity_new_antibiotics_boxplot.png',
            bbox_inches='tight', dpi=300)
plt.savefig('evaluation/bench_validation/designed_mols/output/maximum_similarity_new_antibiotics_boxplot.pdf',
            bbox_inches='tight')
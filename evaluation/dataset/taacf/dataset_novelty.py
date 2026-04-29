import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sim_df = pd.read_csv('data/FlemingDev/experiments/compare_datasets/all_datasets_tanimoto_similarity2.csv')
sim_df = sim_df[sim_df['max_tanimoto'] < 1.0]  # Exclude self-similarity

MLSMR_df = sim_df[sim_df['dataset'] == 'MLSMR'].copy()
TAACF_df = sim_df[sim_df['dataset'] == 'TAACF'].copy()
fleming_df = sim_df[sim_df['dataset'] == 'Our Data'].copy()

MLSMR_hits_df = MLSMR_df[MLSMR_df['inhibition']==1].copy()
TAACF_hits_df = TAACF_df[TAACF_df['inhibition']==1].copy()
fleming_hits_df = fleming_df[fleming_df['inhibition']==1].copy()

# Set style and context for publication
sns.set_style("white")
sns.set_context("talk", font_scale=1.2)  # Larger fonts

# Nature-style color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# --- Overall Similarity Distributions ---
plt.rcParams['axes.labelsize'] = 20   # Axis labels
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 20

plt.figure(figsize=(9.6,7.2))
ax = plt.subplot(111)

ax.hist(fleming_df['max_tanimoto'], 
         bins=50, 
         alpha=0.6, 
         label='Our Dataset', 
         color=colors[0], 
         density=True)
ax.hist(MLSMR_df['max_tanimoto'], 
         bins=50, 
         alpha=0.6, 
         label='MLSMR', 
         color=colors[1], density=True)
ax.hist(TAACF_df['max_tanimoto'], 
         bins=50, 
         alpha=0.6, 
         label='TAACF', 
         color=colors[2], 
         density=True)


plt.xlabel('Within Dataset Maximum \n Pairwise Tanimoto Similarity',
           fontsize=30)
plt.ylabel('Density',
           fontsize=30)
# plt.title('Overall Similarity Distributions')
plt.xlim(0, 1)

# Remove top and right borders
ax.spines[['right','top']].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
legend = ax.legend(loc='center left', bbox_to_anchor=(0.8, 1.0), frameon=False)

# Customize legend
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(20)

# Thicker ticks and axes
# plt.tick_params(width=2, length=6)
# plt.gca().set_yticks([y for y in plt.gca().get_yticks() if y != 0])
# plt.gca().spines['left'].set_linewidth(3)
# plt.gca().spines['bottom'].set_linewidth(3)

# Save figure
plt.savefig('evaluation/dataset/taacf/output/overall_similarity_distributions.png',
            dpi=300,
            bbox_inches='tight')
plt.savefig('evaluation/dataset/taacf/output/overall_similarity_distributions.pdf',
            bbox_inches='tight')
plt.close()

# --- Hit Similarity Distributions ---
plt.figure(figsize=(9.6,7.2))
ax = plt.subplot(111)
plt.hist(fleming_hits_df['max_tanimoto'], 
         bins=50, 
         alpha=0.6, 
         label='Our Hits', 
         color=colors[0], 
         density=True)
plt.hist(MLSMR_hits_df['max_tanimoto'], 
         bins=50, 
         alpha=0.6, 
         label='MLSMR Hits', 
         color=colors[1], 
         density=True)
plt.hist(TAACF_hits_df['max_tanimoto'], 
         bins=50, 
         alpha=0.6, 
         label='TAACF Hits', 
         color=colors[2], 
         density=True)

plt.xlabel('Within Dataset Maximum Pairwise Tanimoto Similarity')
plt.ylabel('Density')
# plt.title('Hit Similarity Distributions')
plt.xlim(0, 1)

# Remove top and right borders
ax.spines[['right','top']].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Customize legend
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(20)

# Thicker ticks and axes
# plt.tick_params(width=2, length=6)
# plt.gca().set_yticks([y for y in plt.gca().get_yticks() if y != 0])
# plt.gca().spines['left'].set_linewidth(2)
# plt.gca().spines['bottom'].set_linewidth(2)

# Save figure
plt.savefig('evaluation/dataset/taacf/output/hit_similarity_distributions.png',
            dpi=300,
            bbox_inches='tight')
plt.savefig('evaluation/dataset/taacf/output/hit_similarity_distributions.pdf',
            bbox_inches='tight')
plt.close()

# Mann-Whitney-test between datasets
from scipy.stats import mannwhitneyu
stat, p = mannwhitneyu(fleming_df['max_tanimoto'], MLSMR_df['max_tanimoto'])
print('Mann-Whitney U test between Our Data and MLSMR: statistic=%.3f, p-value=%.3f' % (stat, p))
stat, p = mannwhitneyu(fleming_df['max_tanimoto'], TAACF_df['max_tanimoto'])
print('Mann-Whitney U test between Our Data and TAACF: statistic=%.3f, p-value=%.3f' % (stat, p))
stat, p = mannwhitneyu(MLSMR_df['max_tanimoto'], TAACF_df['max_tanimoto'])
print('Mann-Whitney U test between MLSMR and TAACF: statistic=%.3f, p-value=%.3f' % (stat, p))


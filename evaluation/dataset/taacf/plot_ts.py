import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

mlsmr_taccf_max_sim = pd.read_csv('evaluation/dataset/taacf/output/mlsmr_taacf_max_similarities.csv')

print("############ Pooled mlsmr-taacf Intra-Dataset Similarity ############")
print(f'Number of molecules in pooled mlsmr and taacf dataset: {len(mlsmr_taccf_max_sim)}')
print(f'Mean similarity: {mlsmr_taccf_max_sim["Max_Tanimoto_Similarity"].mean()}')
print(f'Median similarity: {mlsmr_taccf_max_sim["Max_Tanimoto_Similarity"].median()}')
print(f'Standard deviation of similarity: {mlsmr_taccf_max_sim["Max_Tanimoto_Similarity"].std()}')
print(f'75% Quantile of similarity: {np.percentile(mlsmr_taccf_max_sim["Max_Tanimoto_Similarity"], 75)}')
print(f'25% Quantile of similarity: {np.percentile(mlsmr_taccf_max_sim["Max_Tanimoto_Similarity"], 25)}')

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

ax.hist(mlsmr_taccf_max_sim['Max_Tanimoto_Similarity'], 
         bins=50, 
         alpha=0.6, 
         label='Pooled MLSMR TAACF', 
         color=colors[0], 
         density=True)


plt.xlabel('Maximum \n Pairwise Tanimoto Similarity',
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
plt.savefig('evaluation/dataset/taacf/output/pooled_mlsmr_taacf_similarity_distributions.png',
            dpi=300,
            bbox_inches='tight')
plt.close()


inter_dataset_max_sim = pd.read_csv('evaluation/dataset/taacf/output/inter_dataset_max_similarities.csv')
print("############ Inter-Dataset Structural Overlap Between Our dataset and Pooled mlsmr-taacf ############")
print(f'Number of molecules in Our dataset: {len(inter_dataset_max_sim)}')
print(f"Number of overlapping molecules (Tanimoto similarity = 1.0) between Our dataset and pooled mlsmr-taacf: {(inter_dataset_max_sim['Max_Tanimoto_Similarity'] == 1.0).sum()}")
print(f'Mean similarity: {inter_dataset_max_sim["Max_Tanimoto_Similarity"].mean()}')
print(f'Median similarity: {inter_dataset_max_sim["Max_Tanimoto_Similarity"].median()}')
print(f'Standard deviation of similarity: {inter_dataset_max_sim["Max_Tanimoto_Similarity"].std()}')
print(f'75% Quantile of similarity: {np.percentile(inter_dataset_max_sim["Max_Tanimoto_Similarity"], 75)}')
print(f'25% Quantile of similarity: {np.percentile(inter_dataset_max_sim["Max_Tanimoto_Similarity"], 25)}')

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

ax.hist(inter_dataset_max_sim['Max_Tanimoto_Similarity'], 
         bins=50, 
         alpha=0.6, 
         label='Structural Overlap Between Our Dataset and Pooled MLSMR-TAACF', 
         color=colors[0], 
         density=True)


plt.xlabel('Maximum \n Pairwise Tanimoto Similarity',
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
plt.savefig('evaluation/dataset/taacf/output/interdatset_similarity_distributions.png',
            dpi=300,
            bbox_inches='tight')
plt.close()
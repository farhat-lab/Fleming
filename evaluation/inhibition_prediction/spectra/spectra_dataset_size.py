import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Collect split lengths
split_lengths = []  # List to store the lengths of each split

for split in range(4, 20):
    df = pd.read_csv(f'evaluation/inhibition_prediction/spectra/output/spectra_splits/{split}.0_data.csv')
    split_lengths.append(len(df))
    print(f"Split {split}: {len(df)} samples")

# Nature-style plot settings
plt.figure(figsize=(8, 6))
plt.plot(1 - np.array(range(4, 20)) / 20,
         split_lengths, marker='o', 
         color='black', linewidth=1.5, markersize=6, label='Dataset Size')
plt.xlabel('SPECTRA Parameter', fontsize=14)
plt.ylabel('Number of Molecules', fontsize=14)
plt.ylim(0,120000)
# plt.title('Dataset Size Across Splits', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Remove top and right spines
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)  # Make left edge thicker
ax.spines['bottom'].set_linewidth(3)  # Make bottom edge thicker

# Save the plot
plt.savefig('evaluation/inhibition_prediction/spectra/output/spectra_dataset_sizes_nature_style.png', dpi=300)
plt.savefig('evaluation/inhibition_prediction/spectra/output/spectra_dataset_sizes_nature_style.pdf')
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np

significant_features = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/testset_significant_features.csv')

medium_effect = significant_features[(significant_features['cohend'] >= 0.5) | (significant_features['cohend'] <= -0.5)]

x_axis = np.arange(len(medium_effect))
width=0.3
plt.figure(figsize=(9.6,4.8))
ax = plt.subplot(111)
plt.bar(x_axis-0.15,
        medium_effect['low_confidence_mean'], 
        width, 
        label = 'Molecules with Low Confidence',
        color='royalblue')
plt.errorbar(x_axis-0.15,
             medium_effect['low_confidence_mean'],
             yerr=medium_effect['low_confidence_std'],
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')
plt.bar(x_axis + 0.15, 
        medium_effect['high_confidence_mean'], 
        width, 
        label = 'Molecules with High Confidence',
        color='orange')
plt.errorbar(x_axis + 0.15,
             medium_effect['high_confidence_mean'],
             yerr=medium_effect['high_confidence_std'],
             fmt='o',
             markersize=3,
             capsize=3,
             color = 'black')

plt.xticks(x_axis, 
           medium_effect['feature'], 
           fontsize=8,
           rotation=45)

plt.ylabel("Normalized RDKit Feature Values")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
ax.spines[['right','top']].set_visible(False)

plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/confidence_inference.png')



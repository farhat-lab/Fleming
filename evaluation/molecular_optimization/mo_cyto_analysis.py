import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon

# hepg2 cytotoxicity analysis
hepg2_before = pd.read_csv('evaluation/molecular_optimization/output/hepg2_before.csv')
hepg2_after = pd.read_csv('evaluation/molecular_optimization/output/hepg2_after.csv')

## Remove non-optimized molecules
hepg2_before_filtered = hepg2_before[~hepg2_before['SMILES'].isin(hepg2_after['SMILES'])]
hepg2_after_filtered = hepg2_after[~hepg2_after['SMILES'].isin(hepg2_before['SMILES'])]
hepg2_before_binarized = hepg2_before_filtered[hepg2_before_filtered['Y'] >= 0.7]
hepg2_after_binarized = hepg2_after_filtered[hepg2_after_filtered['Y'] >= 0.7]

## Calculate mean and std
print("HepG2 Cytotoxicity Before MO: Mean =", np.mean(hepg2_before_filtered['Y']), "Std Dev =", np.std(hepg2_before_filtered['Y']))
print("HepG2 Cytotoxicity After MO: Mean =", np.mean(hepg2_after_filtered['Y']), "Std Dev =", np.std(hepg2_after_filtered['Y']))
## Wilcoxon test
stat, p = wilcoxon(hepg2_before_filtered['Y'], hepg2_after_filtered['Y'])
print('Wilcoxon test statistic=%.3f, p-value=%.3f' % (stat, p))

## Print number of molecules and percent of molecules above threshold
print("Number of molecules before MO:", len(hepg2_before_filtered), "(", len(hepg2_before_binarized), "above threshold, ", (len(hepg2_before_binarized)/len(hepg2_before_filtered))*100, "% )")
print("Number of molecules after MO:", len(hepg2_after_filtered), "(", len(hepg2_after_binarized), "above threshold, ", (len(hepg2_after_binarized)/len(hepg2_after_filtered))*100, "% )")

## Plot histogram
plt.figure(figsize=(10,6))
plt.hist(hepg2_before_filtered['Y'], bins=30, alpha=0.5, label='Before MO', color='blue', density=True)
plt.hist(hepg2_after_filtered['Y'], bins=30, alpha=0.5, label='After MO', color='orange', density=True)
plt.axvline(x=0.7, color='red', linestyle='--', label='Toxicity Threshold (0.7)')
plt.title('HepG2 Cytotoxicity Distribution Before and After Molecular Optimization')
plt.xlabel('Cytotoxicity Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('evaluation/molecular_optimization/output/hepg2_cytotoxicity_distribution.png')
plt.close()

# hskmc totoxicity analysis
hskmc_before = pd.read_csv('evaluation/molecular_optimization/output/hskmc_before.csv')
hskmc_after = pd.read_csv('evaluation/molecular_optimization/output/hskmc_after.csv')

## Remove non-optimized molecules
hskmc_before_filtered = hskmc_before[~hskmc_before['SMILES'].isin(hskmc_after['SMILES'])]
hskmc_after_filtered = hskmc_after[~hskmc_after['SMILES'].isin(hskmc_before['SMILES'])]
hskmc_before_binarized = hskmc_before_filtered[hskmc_before_filtered['Y'] >= 0.7]
hskmc_after_binarized = hskmc_after_filtered[hskmc_after_filtered['Y'] >= 0.7]

## Calculate mean and std
print("HskMC Cytotoxicity Before MO: Mean =", np.mean(hskmc_before_filtered['Y']), "Std Dev =", np.std(hskmc_before_filtered['Y']))
print("HskMC Cytotoxicity After MO: Mean =", np.mean(hskmc_after_filtered['Y']), "Std Dev =", np.std(hskmc_after_filtered['Y']))
## Wilcoxon test
stat, p = wilcoxon(hskmc_before_filtered['Y'], hskmc_after_filtered['Y'])
print('Wilcoxon test statistic=%.3f, p-value=%.3f' % (stat, p))

## Print number of molecules and percent of molecules above threshold
print("Number of molecules before MO:", len(hskmc_before_filtered), "(", len(hskmc_before_binarized), "above threshold, ", (len(hskmc_before_binarized)/len(hskmc_before_filtered))*100, "% )")
print("Number of molecules after MO:", len(hskmc_after_filtered), "(", len(hskmc_after_binarized), "above threshold, ", (len(hskmc_after_binarized)/len(hskmc_after_filtered))*100, "% )")

## Plot histogram
plt.figure(figsize=(10,6))
plt.hist(hskmc_before_filtered['Y'], bins=30, alpha=0.5, label='Before MO', color='blue', density=True)
plt.hist(hskmc_after_filtered['Y'], bins=30, alpha=0.5, label='After MO', color='orange', density=True)
plt.axvline(x=0.7, color='red', linestyle='--', label='Toxicity Threshold (0.7)')
plt.title('HskMC Cytotoxicity Distribution Before and After Molecular Optimization')
plt.xlabel('Cytotoxicity Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('evaluation/molecular_optimization/output/hskmc_cytotoxicity_distribution.png')
plt.close()

# imr90 toxicity analysis
imr90_before = pd.read_csv('evaluation/molecular_optimization/output/imr90_before.csv')
imr90_after = pd.read_csv('evaluation/molecular_optimization/output/imr90_after.csv')

## Remove non-optimized molecules
imr90_before_filtered = imr90_before[~imr90_before['SMILES'].isin(imr90_after['SMILES'])]
imr90_after_filtered = imr90_after[~imr90_after['SMILES'].isin(imr90_before['SMILES'])]
imr90_before_binarized = imr90_before_filtered[imr90_before_filtered['Y'] >= 0.7]
imr90_after_binarized = imr90_after_filtered[imr90_after_filtered['Y'] >= 0.7]

## Calculate mean and std
print("IMR90 Cytotoxicity Before MO: Mean =", np.mean(imr90_before_filtered['Y']), "Std Dev =", np.std(imr90_before_filtered['Y']))
print("IMR90 Cytotoxicity After MO: Mean =", np.mean(imr90_after_filtered['Y']), "Std Dev =", np.std(imr90_after_filtered['Y']))
## Wilcoxon test
stat, p = wilcoxon(imr90_before_filtered['Y'], imr90_after_filtered['Y'])
print('Wilcoxon test statistic=%.3f, p-value=%.3f' % (stat, p))

## Print number of molecules and percent of molecules above threshold
print("Number of molecules before MO:", len(imr90_before_filtered), "(", len(imr90_before_binarized), "above threshold, ", (len(imr90_before_binarized)/len(imr90_before_filtered))*100, "% )")
print("Number of molecules after MO:", len(imr90_after_filtered), "(", len(imr90_after_binarized), "above threshold, ", (len(imr90_after_binarized)/len(imr90_after_filtered))*100, "% )")

## Plot histogram
plt.figure(figsize=(10,6))
plt.hist(imr90_before_filtered['Y'], bins=30, alpha=0.5, label='Before MO', color='blue', density=True)
plt.hist(imr90_after_filtered['Y'], bins=30, alpha=0.5, label='After MO', color='orange', density=True)
plt.axvline(x=0.7, color='red', linestyle='--', label='Toxicity Threshold (0.7)')
plt.title('IMR90 Cytotoxicity Distribution Before and After Molecular Optimization')
plt.xlabel('Cytotoxicity Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('evaluation/molecular_optimization/output/imr90_cytotoxicity_distribution.png')
plt.close()


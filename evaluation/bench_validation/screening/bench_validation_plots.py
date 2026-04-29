import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import ttest_ind, mannwhitneyu

# Bench Result DataFrame
bench_validation_results = pd.read_excel('data/output/dragonfold/updated_screening_result/Charm_mols_WCA_full_SMILES_bgCorrected_forAlex.xlsx',
                                         sheet_name='CDD Export 1')

bench_validation_results = bench_validation_results[['SMILES', 
                                                     'Mtb_WCA_mc2-7000_Resazurin: concentration (uM)', 
                                                     'Mtb_WCA_mc2-7000_Resazurin: % inhibition (%)']]

def maybe_split(val):
    if isinstance(val, str) and ',' in val:
        return [v.strip() for v in val.split(',') if v.strip() != '']
    return val

# Split the Concentration and %inhibiton column by comma and make it a list
bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: concentration (uM)'] = bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: concentration (uM)'].apply(maybe_split)
bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: % inhibition (%)'] = bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: % inhibition (%)'].apply(maybe_split)

# Prediction DataFrame
final_output = pd.read_csv('data/output/dragonfold/updated_screening_result/ensemble_pass_both.csv')
print(f"Predicted Hits: {len(final_output)}")
print(final_output[~final_output['SMILES'].isin(bench_validation_results['SMILES'])])

# Keep only the highest percentrge inhibition for each molecule
bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: % inhibition (%)'] = bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: % inhibition (%)'].apply(lambda x: [float(i) for i in x] if isinstance(x, list) else [float(x)])
bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: concentration (uM)'] = bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: concentration (uM)'].apply(lambda x: [float(i) for i in x] if isinstance(x, list) else [float(x)])
bench_validation_results['max_inhibition'] = bench_validation_results['Mtb_WCA_mc2-7000_Resazurin: % inhibition (%)'].apply(lambda x: max(x))

# max_concentration_results = bench_validation_results.loc[bench_validation_results.groupby('SMILES')['Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation (%)'].idxmax()]
max_concentration_results = bench_validation_results.copy()
max_concentration_results['predicted_hit'] = np.where(max_concentration_results['SMILES'].isin(final_output['SMILES']),"Predicted Hit","Predicted Non-hit")

print(max_concentration_results[max_concentration_results['predicted_hit'] == 'Predicted Hit'])

print(f"Hits: {len(max_concentration_results[max_concentration_results['max_inhibition'] >= 40])}")
print(f"Non-Hits: {len(max_concentration_results[max_concentration_results['max_inhibition'] < 40])}")

## Predicted Hits and Non Hits Percentage Inhibition
plt.figure(figsize=(9.6,7.2))
ax = plt.subplot(111)
for spine in ax.spines.values():
    spine.set_linewidth(5)
sns.swarmplot(y="max_inhibition", 
              data=max_concentration_results,
              hue = 'predicted_hit',
              legend = ['predicted hits','predicted non-hits'],
              size=10)
plt.axhline(y=40, color='black', linestyle='--', lw=5)

legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(20)

ax.spines[['right','top']].set_visible(False)
ax.set_xlabel('Bench Validation Molecules', 
              fontsize=25, 
              #fontweight='bold'
              )
ax.set_ylabel('MTB (mc2-7000) Whole Cell \n Activity Readout (%Inhibition)', 
              fontsize=25, 
              # fontweight='bold'
              )

ax.tick_params(axis='y', labelsize=20)

# Make legend texts bold and increase font size
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(20)

plt.savefig('evaluation/bench_validation/screening/output/updated_prediction_inhibition_plot.png',
            bbox_inches='tight')
plt.savefig('evaluation/bench_validation/screening/output/updated_prediction_inhibition_plot.pdf',
            bbox_inches='tight')

## Predicted Hits Concentration Plot
# hit_molecules = bench_validation_results[bench_validation_results['SMILES'].isin(final_output['SMILES'])]

# plt.figure()
# fig, axs = plt.subplots(2,3,
#                        figsize=(12.8,4.8))
# axs = axs.ravel()

# count = 0
# for molecule in hit_molecules['SMILES'].unique():
#     sub_df = hit_molecules[hit_molecules['SMILES'] == molecule]
#     sub_df = sub_df.groupby(['SMILES','Mtb_WCA_mc2-7000_Resazurin: concentration (uM)'], as_index=False)['Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation (%)'].mean()
#     axs[count].plot(sub_df['Mtb_WCA_mc2-7000_Resazurin: concentration (uM)'],
#              sub_df['Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation (%)'],
#              marker='o')
#     axs[count].axhline(y=40, color='black', linestyle='--', lw=2)
#     axs[count].spines[['right','top']].set_visible(False)
#     count += 1

# fig.supxlabel('Concentration (uM)', fontsize=14)
# fig.supylabel('MTB (mc2-7000) Whole Cell \n Activity Readout (%Inhibition)', fontsize=14)

# plt.tight_layout()
# plt.savefig('evaluation/bench_validation/screening/output/hit_concentration_plot.png')
# plt.savefig('evaluation/bench_validation/screening/output/hit_concentration_plot.pdf')

## Novelty Plots
similarity_matrix = pd.read_csv('evaluation/bench_validation/screening/output/train_similarity.csv')
similarity_matrix = similarity_matrix[similarity_matrix['train_max_sim'] != 1]

similarity_matrix = similarity_matrix.merge(max_concentration_results[['SMILES','max_inhibition','predicted_hit']],
                                            on='SMILES',
                                            how='left')

similarity_matrix['binary_inhibition'] = np.where(similarity_matrix['max_inhibition'] >= 40, 1, 0)
hit_matrix = similarity_matrix[similarity_matrix['binary_inhibition'] == 1]
nonhit_matrix = similarity_matrix[similarity_matrix['binary_inhibition'] == 0]

true_hit = hit_matrix[hit_matrix['predicted_hit_y'] == 'Predicted Hit']
miss_hit = hit_matrix[hit_matrix['predicted_hit_y'] == 'Predicted Non-hit']

print(f"True Hit Mean Similarity: {np.mean(true_hit['train_max_sim'])} +/- {np.std(true_hit['train_max_sim'])}")
print(f"True Hit Median Similarity: {np.median(true_hit['train_max_sim'])} +/- {np.quantile(true_hit['train_max_sim'],[0.25,0.75])}")
print(f"Miss Hit Mean Similarity: {np.mean(miss_hit['train_max_sim'])} +/- {np.std(miss_hit['train_max_sim'])}")
print(f"Miss Hit Median Similarity: {np.median(miss_hit['train_max_sim'])} +/- {np.quantile(miss_hit['train_max_sim'],[0.25,0.75])}")
print(mannwhitneyu(true_hit['train_max_sim'],miss_hit['train_max_sim']).pvalue)

exit()

## Diversity Histogram
plt.figure()
ax = plt.subplot(111)
sns.histplot(x="train_max_sim", 
             data=similarity_matrix,
             stat="percent",)
# sns.histplot(x="train_max_sim", 
#              data=hit_matrix,
#              stat="percent",)
# sns.histplot(x="train_max_sim", 
#              data=nonhit_matrix,
#              stat="percent",)
plt.xlabel('Maximum Tanimoto Similarity with Training Data', fontsize=14)
plt.ylabel('Percentage of Molecules', fontsize=14)
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/bench_validation/screening/output/similarity_histogram.png')
plt.savefig('evaluation/bench_validation/screening/output/similarity_histogram.pdf')
print(f"Hits Mean Similarity: {np.mean(hit_matrix['train_max_sim'])} +/- {np.std(hit_matrix['train_max_sim'])}, Median Similarity: {np.median(hit_matrix['train_max_sim'])} [{np.quantile(hit_matrix['train_max_sim'],[0.25,0.75])}]")
print(f"Non-Hits Mean Similarity: {np.mean(nonhit_matrix['train_max_sim'])} +/- {np.std(nonhit_matrix['train_max_sim'])} , Median Similarity: {np.median(nonhit_matrix['train_max_sim'])} [{np.quantile(nonhit_matrix['train_max_sim'],[0.25,0.75])}]")

## Hits and Diversity
plt.figure(figsize=(5,5))
ax = plt.subplot(111)
sns.swarmplot(y="train_max_sim", 
              data=hit_matrix,
              hue = 'predicted_hit',
              legend = ['predicted hits','predicted non-hits'])
# plt.axhline(y=40, color='black', linestyle='--')
plt.xlabel('Bench Validation Hits', fontsize=14)
plt.ylabel('Maximum Tanimoto Similarity \n with Training Data', fontsize=14)
plt.ylim(0,1)
plt.legend(title=None, loc="center left", bbox_to_anchor=(1, 0.5))
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/bench_validation/screening/output/hit_similarity.png')
plt.savefig('evaluation/bench_validation/screening/output/hit_similarity.pdf')

## Non-Hits and Diversity
plt.figure(figsize=(5,5))
ax = plt.subplot(111)
sns.swarmplot(y="train_max_sim", 
              data=nonhit_matrix,
              hue = 'predicted_hit',
              legend = ['predicted hits','predicted non-hits'])
# plt.axhline(y=40, color='black', linestyle='--')
plt.xlabel('Bench Validation Non-hits', fontsize=14)
plt.ylabel('Maximum Tanimoto Similarity \n with Training Data', fontsize=14)
plt.ylim(0,1)
plt.legend(title=None, loc="center left", bbox_to_anchor=(1, 0.5))
ax.spines[['right','top']].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/bench_validation/screening/output/nonhit_similarity.png')
plt.savefig('evaluation/bench_validation/screening/output/nonhit_similarity.pdf')
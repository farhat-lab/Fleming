import sys
import os
import seaborn as sns
# sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
sys.path.append(os.path.join(sys.prefix, 'share', 'RDKit', 'Contrib'))

from SA_Score import sascorer
from rdkit import Chem
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

def SAScore(prediction_dir_1: pd.DataFrame,
            prediction_dir_2: pd.DataFrame,
            score_save_dir: pd.DataFrame):

    prediction_df_1 = pd.read_csv(prediction_dir_1)
    prediction_df_2 = pd.read_csv(prediction_dir_2)

    sascore_list = []
    for smiles in tqdm(prediction_df_1['SMILES']):
        m  = Chem.MolFromSmiles(smiles)
        sascore_list.append(sascorer.calculateScore(m))
    prediction_df_1['sa_score'] = sascore_list

    sascore_list = []
    for smiles in tqdm(prediction_df_2['SMILES']):
        m  = Chem.MolFromSmiles(smiles)
        sascore_list.append(sascorer.calculateScore(m))
    prediction_df_2['sa_score'] = sascore_list

    plt.rcParams['axes.labelsize'] = 20   # Axis labels
    plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 20

    plt.figure(figsize=(9.6,7.2))
    ax=plt.subplot(111)
    for spine in ax.spines.values():
        spine.set_linewidth(5)

    plt.hist(prediction_df_1['sa_score'], 
             # label='Substructure Based Generative Model',
             density=True,
             bins=20,
             alpha=0.1,
             color='orange')
    sns.kdeplot(prediction_df_1['sa_score'],
                label='SyntheMol',
                color = 'orange',
                lw=5)
    plt.hist(prediction_df_2['sa_score'], 
             # label='Diffusion Generative Model',
             density=True,
             bins=10,
             alpha=0.1,
             color='green')
    sns.kdeplot(prediction_df_2['sa_score'],
                label='Diffusion',
                color='green',
                lw=5)
    plt.xlim(1,10)
    legend = ax.legend(loc='center left', 
                       bbox_to_anchor=(0.8, 0.5), 
                       frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)
    ax.set_xlabel("SAScore", 
              fontsize=30, 
              # fontweight='bold'
              )
    ax.set_ylabel("Density", 
              fontsize=30, 
              # fontweight='bold'
              )
    ax.spines[['right','top']].set_visible(False)
    
    plt.savefig(score_save_dir + 'synthesizability_distribution_comparison.png',
                bbox_inches='tight')
    plt.savefig(score_save_dir + 'synthesizability_distribution_comparison.pdf',
                bbox_inches='tight')

    prediction_df_1 = prediction_df_1.sort_values(by=['sa_score'],
                                                  ascending=True,)
    prediction_df_1.to_csv(score_save_dir + 'synthemol_molecule_synthesizability.csv',
                          index = False)
    prediction_df_2 = prediction_df_2.sort_values(by=['sa_score'],
                                                  ascending=True,)
    prediction_df_2.to_csv(score_save_dir + 'diffusion_molecule_synthesizability.csv',
                          index = False)

    # Calculate Wilcoxon Rank Sun p-value
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(prediction_df_1['sa_score'], prediction_df_2['sa_score'], alternative='two-sided')
    print(f"Wilcoxon Rank Sum (Mann-Whitney U) test p-value: {p_value}")
    

    # return prediction_df

if __name__ == '__main__':

    SAScore(prediction_dir_1='data/output/generative/synthemol_generated/ensemble_pass_both.csv', # Synthemol Generated
            prediction_dir_2='data/output/generative/diffusion_generated/ensemble_pass_both.csv', # Diffusion Generated
            score_save_dir='evaluation/molecule_generation/output/')

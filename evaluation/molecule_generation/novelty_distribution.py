import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

def plot_molecular_novelty (molecule_sim_dir_1:str,
                            molecule_sim_dir_2:str,
                            output_dir:str):

    molecular_sim_1_df = pd.read_csv(molecule_sim_dir_1)
    print(len(molecular_sim_1_df))
    molecular_sim_2_df = pd.read_csv(molecule_sim_dir_2)
    print(molecular_sim_2_df)

    plt.rcParams['axes.labelsize'] = 20   # Axis labels
    plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 20
    
    plt.figure(figsize=(9.6,7.2))
    ax=plt.subplot(111)
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    
    plt.hist(molecular_sim_1_df['max_sim'],
             alpha = 0.1,
             bins=20,
             density = True,
             color = 'orange'
             # label = 'MCTS Generative Model',
             )
    sns.kdeplot(molecular_sim_1_df['max_sim'],
                label='SyntheMol',
                color = 'orange',
                lw=5)
    plt.hist(molecular_sim_2_df['max_sim'],
             alpha = 0.1,
             density = True,
             bins=20,
             color = 'green'
             # label = 'Diffusion Generative Model',
             )
    sns.kdeplot(molecular_sim_2_df['max_sim'],
                label='Diffusion',
                color = 'green',
                lw=5)
    legend = ax.legend(loc='center left', 
                       bbox_to_anchor=(0.8, 0.5), 
                       frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)
    ax.set_xlabel("Maximum Similarity", 
              fontsize=30, 
              # fontweight='bold'
              )
    ax.set_ylabel("Density", 
              fontsize=30, 
              # fontweight='bold'
              )
    ax.spines[['right','top']].set_visible(False)

    plt.savefig(output_dir + 'novelty_distribution.png',
                bbox_inches='tight')
    plt.savefig(output_dir + 'novelty_distribution.pdf',
                bbox_inches='tight')

    print(f"Synthemol Model: Mean Sim= {np.mean(molecular_sim_1_df['max_sim'])} +/- {np.std(molecular_sim_1_df['max_sim'])}, Median Sim = {np.median(molecular_sim_1_df['max_sim'])} [{np.quantile(molecular_sim_1_df['max_sim'],[0.25,0.75])}]")
    print(f"Diff Model: Mean Sim= {np.mean(molecular_sim_2_df['max_sim'])} +/- {np.std(molecular_sim_2_df['max_sim'])}, Median Sim = {np.median(molecular_sim_2_df['max_sim'])} [{np.quantile(molecular_sim_2_df['max_sim'],[0.25,0.75])}]")

    # Statistical Test
    print("p-value = ", mannwhitneyu(molecular_sim_1_df['max_sim'], molecular_sim_2_df['max_sim']).pvalue)

if __name__ == '__main__':

    plot_molecular_novelty('data/output/generative/synthemol_generated/with_sim.csv',
                           'data/output/generative/diffusion_generated/with_sim.csv',
                           output_dir='evaluation/molecule_generation/output/')
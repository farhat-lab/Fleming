import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def plot_pharmacokinetics(prediction_dir_1:str,
                          prediction_dir_2:str,
                          output_dir:str):
    
    prediction_df_1 = pd.read_csv(prediction_dir_1)
    prediction_df_2 = pd.read_csv(prediction_dir_2)

    plt.rcParams['axes.labelsize'] = 20   # Axis labels
    plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 20

    plt.figure(figsize=(15,10))
    ax=plt.subplot(111)
    for spine in ax.spines.values():
        spine.set_linewidth(5)

    plt.hist((37 - prediction_df_1['fail_sum'])/37 * 100,
             alpha = 0.1,
             bins=10,
             density = True,
             color = 'lightsteelblue'
             # label = 'MCTS Generative Model',
             )
    sns.kdeplot((37 - prediction_df_1['fail_sum'])/37 * 100,
                label='MCTS Generative Model',
                color = 'lightsteelblue',
                lw=5)
    plt.hist((37 - prediction_df_2['fail_sum'])/37 * 100,
             alpha = 0.1,
             density = True,
             bins=10,
             color = 'orange'
             # label = 'Diffusion Generative Model',
             )
    sns.kdeplot((37 - prediction_df_2['fail_sum'])/37 * 100,
                label='Diffusion Generative Model',
                color = 'orange',
                lw=5)
    legend = ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5), frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)
    ax.set_xlabel("Pharmacokinetics Pass Rate (%)", 
              fontsize=30, 
              # fontweight='bold'
              )
    ax.set_ylabel("Density", 
              fontsize=30, 
              # fontweight='bold'
              )
    ax.spines[['right','top']].set_visible(False)
    plt.savefig(output_dir)


if __name__ == '__main__':

    plot_pharmacokinetics('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/generative/synthemol_generated/ranking_admet_prediction.csv', # Synthemol
                          '/home/aw274/Mtb_Inhibition_DL_Pipeline/output/generative/diffusion_generated/ranking_admet_prediction.csv', # Diffusion Model
                          output_dir='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/pharmacokinetics_distribution.png')
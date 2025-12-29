import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns

def compare_Pharmacokinetics_distribution (before: str,
                                           after: str,
                                           output_dir: str):

    before_df = pd.read_csv(before)
    after_df = pd.read_csv(after)

    before_total_fail = before_df['fail_sum'].to_list()
    after_total_fail = after_df['fail_sum'].to_list()[9:]

    # Plot Distribution Before after MO
    plt.rcParams['axes.labelsize'] = 20   # Axis labels
    plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 20
    plt.figure(figsize=(9.6,7.2))
    ax = plt.subplot(111)
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    plt.hist(before_total_fail,
             alpha=0.1,
             density = True,
             color = '#999999')
    sns.kdeplot(before_total_fail,
                label = 'Before Molecular \n Optimization',
                color = '#999999',
                lw = 5)
    plt.hist(after_total_fail,
             alpha=0.1,
             density = True,
             color = '#0072B2')
    sns.kdeplot(after_total_fail,
                label = 'After Molecular \n Optimization',
                color = '#0072B2',
                lw=5)
    
    plt.xlim(0,17)
    ax.spines[['right','top']].set_visible(False)
    ax.set_xlabel('Number of Failed ADMET.AI Property Tests', 
              fontsize=30, 
              #fontweight='bold'
              )
    ax.set_ylabel('Density', 
                fontsize=30, 
                # fontweight='bold'
                )
    legend = ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5), frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)
    
    plt.savefig(output_dir + "pharmacokinetics_before_after_MO.png",
                bbox_inches='tight')
    plt.savefig(output_dir + "pharmacokinetics_before_after_MO.pdf",
                bbox_inches='tight')

    # Plot Inhibition Distribution Before after MO
    plt.figure(figsize=(9.6,7.2))
    ax = plt.subplot(111)
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    plt.hist(before_df['inhibition'],
             alpha=0.1,
             density = True,
             color = 'lightsteelblue')
    sns.kdeplot(before_df['inhibition'],
                label = 'Before Molecular Optimization',
                color = 'lightsteelblue',
                lw = 5)
    plt.hist(after_df['inhibition'],
             alpha=0.1,
             density = True,
             color = 'orange')
    sns.kdeplot(after_df['inhibition'],
                label = 'After Molecular Optimization',
                color = 'orange',
                lw = 5)

    plt.xlim(0.5,1)
    ax.spines[['right','top']].set_visible(False)
    ax.set_xlabel('Predicted Inhibition', 
              fontsize=30, 
              #fontweight='bold'
              )
    ax.set_ylabel('Density', 
                fontsize=30, 
                # fontweight='bold'
                )
    legend = ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5), frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)
    plt.savefig(output_dir + "inhibition_before_after_MO.png",
                bbox_inches='tight')
    plt.savefig(output_dir + "inhibition_before_after_MO.pdf",
                bbox_inches='tight')

    print(f"Mean Inhibition Before MO: {np.mean(before_df['inhibition'])} +/- {np.std(before_df['inhibition'])}")
    print(f"Mean Inhibition After MO: {np.mean(after_df['inhibition'])} +/- {np.std(after_df['inhibition'])}")
    print(f"Median Inhibition Before MO: {np.median(before_df['inhibition'])} [{np.quantile(before_df['inhibition'],[0.25,0.75])}]")
    print(f"Median Inhibition Before MO: {np.median(after_df['inhibition'])} [{np.quantile(after_df['inhibition'],[0.25,0.75])}]")


    # Calculate P Value
    pvalue = mannwhitneyu(before_total_fail, 
                       after_total_fail,
                       alternative='greater').pvalue    
    print("Before Mean:", sum(before_total_fail) / len(before_total_fail))
    print("After Mean:", sum(after_total_fail)/len(after_total_fail))
    print(f"Median Before MO: {np.median(before_total_fail)} [{np.quantile(before_total_fail,[0.25,0.75])}]")
    print(f"Median Before MO: {np.median(after_total_fail)} [{np.quantile(after_total_fail,[0.25,0.75])}]")
    print(pvalue)

    # Bar Plot
    before_df['group'] = 'Seed Molecules'
    after_df['group'] = 'Molecular Optimized Molecules'
    bar_df = pd.concat([before_df,after_df],axis=0)
    bar_df = bar_df.reset_index(drop=True)
    plt.figure()
    sns.set_theme(style='whitegrid')
    sns.barplot(x = "group",
                y = 'fail_sum',
                data = bar_df,
                capsize=.1,
                errorbar='sd',
                palette=['blue','orange'])
    sns.swarmplot(x = "group",
                y = 'fail_sum',
                data = bar_df,
                color='0',
                alpha=.35)
    plt.savefig(output_dir + "MO_barplot.png")

    # Calculate the number of molecules failed more than 10 ADMET properties
    before_fail_10 = len([x for x in before_total_fail if x >= 10])
    after_fail_10 = len([x for x in after_total_fail if x >= 10])
    
    # Get the molecules that failed 11 tasks, and see how many they fail after optimization, use index not smiles
    before_fail_11 = [before_df.iloc[i]['SMILES'] for i,x in enumerate(before_total_fail) if x == 11]
    after_fail_11 = after_df[after_df['SMILES'].isin(before_fail_11)]
    after_fail_11_counts = after_fail_11['fail_sum'].to_list()
    print("Num Molecules failed 11 tasks before MO:", len(before_fail_11))
    print("Their fail counts after MO:", after_fail_11_counts)

    print(f"Number of molecules failed more than 10 ADMET properties before MO: {before_fail_10} / {len(before_total_fail)} ({before_fail_10/len(before_total_fail)*100:.2f}%)")
    print(f"Number of molecules failed more than 10 ADMET properties after MO: {after_fail_10} / {len(after_total_fail)} ({after_fail_10/len(after_total_fail)*100:.2f}%)")


if __name__ == '__main__':

    compare_Pharmacokinetics_distribution(before = "/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/molecular_optimization_result_v2/top_generated_molecules.csv",
                                          # before = "/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/top_100_output.csv",
                                          after = "/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/molecular_optimization_result_v2/step_5/best_siblings.csv",
                                          # after = "/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/molecular_optimization_result/best_siblings.csv",
                                          output_dir = "/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/molecular_optimization/",
                                          # output_dir = "/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/molecular_optimization_result/"
                                         )
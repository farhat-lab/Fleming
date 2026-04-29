import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter

name_mapping = pd.read_excel('evaluation/bench_validation/designed_mols/FinGen_mols_WCA.xlsx')
inhibition_df = pd.read_excel('evaluation/bench_validation/designed_mols/WCA_PKJ_RB_GenAICompounds_100uM_RY_05_13_25_Rezasurin.xlsx')

all_df = pd.DataFrame([])
for plate in inhibition_df['Plate name'].unique():

    plate_df = inhibition_df[inhibition_df['Plate name'] == plate]

    negative_control = plate_df[plate_df['Well'].isin(['A01','B01','C01','D01','E01',
                                                       'F01','G01','H01','I01','J01',
                                                       'K01','L01','M01','N01','O01',
                                                       'P01'])][' Raw Data (580-80)'].mean()

    positive_control = plate_df[plate_df['Well'].isin(['A23','B23','C23','D23','E23',
                                                       'F23'])][' Raw Data (580-80)'].mean()

    plate_df['% Inhibition'] =  (negative_control - plate_df[' Raw Data (580-80)']) / (negative_control - positive_control) * 100

    all_df = pd.concat([all_df, plate_df])

all_df = all_df.dropna(subset=['Molecule Batch ID'])

# 4PL model
def four_param_logistic(x, A1, A2, IC50, p):
    return (A1 - A2) / (1 + (x / IC50)**p) + A2

for mol_id in name_mapping['Molecule Name']:

    summary = all_df[all_df['Molecule Batch ID'].str.contains(mol_id)] \
            .groupby('Concentration', as_index=False)['% Inhibition'] \
            .agg(['mean', 'std', 'count'])

    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    concentrations = summary['Concentration'].values
    means = summary['mean'].values
    stds = summary['std'].values

    try:
        popt, _ = curve_fit(
            four_param_logistic,
            concentrations,
            means,
            p0=[0, 100, 1, 1],
            bounds=([0, 0, 1e-6, 0], [100, 100, np.inf, 10]),
            maxfev=10000
        )

        # Smooth curve for plotting
        # print(concentrations)
        x_fit = np.logspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 200)
        y_fit = four_param_logistic(x_fit, *popt)

        # Plot
        plt.figure(figsize=(7, 5))
        plt.errorbar(concentrations, means, yerr=summary['se'].values, fmt='o', 
                     capsize=0, markersize=5, elinewidth=2, label='Data', 
                     color = '#aec7e8', zorder=10)
        plt.plot(x_fit, y_fit, '-', label='4PL Fit', 
                 linewidth=3, color = '#1f77b4')
        plt.xscale('log')
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.gca().ticklabel_format(style='plain', axis='x')
        plt.xlabel('Concentration (uM)',
                   fontsize=18)
        plt.ylabel('% Inhibition',
                   fontsize=18)
        plt.ylim(-20,120)
        # plt.title(f'Dose-response curve: {mol_id}')
        # plt.grid(True)
        plt.tick_params(axis='both', width=3)  # Thicker ticks
        plt.tick_params(axis='x', which='minor', bottom=False)
        plt.gca().spines['bottom'].set_linewidth(3)  # Thicker x-axis
        plt.gca().spines['left'].set_linewidth(3)    # Thicker y-axis
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(f'evaluation/bench_validation/designed_mols/Inhibition_Curves/{mol_id}_inhibition_curve.png')
        plt.savefig(f'evaluation/bench_validation/designed_mols/Inhibition_Curves/{mol_id}_inhibition_curve.pdf')

    except RuntimeError as e:
        print(f"Fit failed for {mol_id}: {e}")
    
#print(all_df[all_df['Molecule Batch ID'].str.contains('SACC-0504827')][['Concentration','% Inhibition']].groupby('Concentration').mean()['% Inhibition'])

# print(name_mapping)
# print(inhibition_df)
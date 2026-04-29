import pandas as pd

# Load the CSV file
results_df = pd.read_csv('evaluation/inhibition_prediction/spectra/output/rf_test_results.csv')

# Group by 'split' and calculate mean and standard deviation for AUROC and AUPRC
summary = results_df.groupby('split').agg(
    mean_auroc=('AUROC', 'mean'),
    sd_auroc=('AUROC', 'std'),
    mean_auprc=('AUPRC', 'mean'),
    sd_auprc=('AUPRC', 'std')
).reset_index()

# Save the summary to a new CSV file
summary.to_csv('evaluation/inhibition_prediction/spectra/output/rf_summary_results.csv', index=False)

# Print the summary
print(summary)
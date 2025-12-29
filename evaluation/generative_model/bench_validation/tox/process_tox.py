import pandas as pd
import numpy as np

admet_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_admet_ranking.csv')

hepg2_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_hepg2.csv')
hskmc_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_hskmc.csv')
imr90_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_imr90.csv')

cyto_df = hepg2_df.merge(
    hskmc_df, on='SMILES', suffixes=('_hepg2', '_hskmc')
)

cyto_df = cyto_df.merge(
    imr90_df.add_suffix('_imr90').rename(columns={'SMILES_imr90': 'SMILES'}),
    on='SMILES'
)

hepg2_current_antibiotic_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/current_antibiotic_hepg2.csv')
hskmc_current_antibiotic_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/current_antibiotic_hskmc.csv')
imr90_current_antibiotic_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/current_antibiotic_imr90.csv')

cyto_current_antibiotic_df = hepg2_current_antibiotic_df.merge(
    hskmc_current_antibiotic_df, on='SMILES', suffixes=('_hepg2', '_hskmc')
)   

cyto_current_antibiotic_df = cyto_current_antibiotic_df.merge(
    imr90_current_antibiotic_df.add_suffix('_imr90').rename(columns={'SMILES_imr90': 'SMILES'}),
    on='SMILES'
)

# Get prediction quantiles of 0.1 and 0.9 for each cytotoxicity model from current antibiotic predictions
hepg2_lower_threshold = cyto_current_antibiotic_df['Y_hepg2'].quantile(0.9)
hskmc_lower_threshold = cyto_current_antibiotic_df['Y_hskmc'].quantile(0.9)
imr90_lower_threshold = cyto_current_antibiotic_df['Y_imr90'].quantile(0.9)
print(f"HepG2 Lower Threshold (0.1 quantile): {hepg2_lower_threshold}")
print(f"HSKMC Lower Threshold (0.1 quantile): {hskmc_lower_threshold}")
print(f"IMR90 Lower Threshold (0.1 quantile): {imr90_lower_threshold}") 

cyto_df['pass_hepg2'] = np.where(cyto_df['Y_hepg2'] <= hepg2_lower_threshold, 1, 0)
cyto_df['pass_hskmc'] = np.where(cyto_df['Y_hskmc'] <= hskmc_lower_threshold, 1, 0)
cyto_df['pass_imr90'] = np.where(cyto_df['Y_imr90'] <= imr90_lower_threshold, 1, 0)

final_df = admet_df.merge(
    cyto_df, on='SMILES'
)

## Filter for columns start with pass_
pass_columns = [col for col in final_df.columns if col.startswith('pass_')]
final_df = final_df[['SMILES'] + pass_columns]
final_df['fail_sum'] = final_df[pass_columns].apply(lambda x: (x == 0).sum(), axis=1)
final_df = final_df.sort_values(by=['fail_sum'])

# Create a column containing list of failed tests
def get_failed_tests(row):
    failed_tests = []
    for col in pass_columns:
        if row[col] == 0:
            failed_tests.append(col.replace('pass_', ''))
    return failed_tests
final_df['failed_tests'] = final_df.apply(get_failed_tests, axis=1)

final_df.to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_tox_admet_summary.csv',
                index=False)

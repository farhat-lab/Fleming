import pandas as pd 
import numpy as np
from current_antibiotic_threshold import get_admet_threshold
import argparse

task_weight_dict = {
    ## Absorption Task:
    'Bioavailability_Ma': 1,
    'HIA_Hou': 1,
    'PAMPA_NCATS': 1,
    'Pgp_Broccatelli': 1,
    'Caco2_Wang': 1,
    'Lipophilicity_AstraZeneca': 1,
    'Solubility_AqSolDB': 1,

    ## Distribution Tasks:
    'BBB_Martins': 1,
    'PPBR_AZ': 1,
    'VDss_Lombardo': 1,

    ## Metabolism Tasks:
    'CYP1A2_Veith': 1,
    'CYP2C19_Veith':  1,
    # 'CYP2C9_Substrate_CarbonMangels': 1,
    'CYP2C9_Veith': 1,
    # 'CYP2D6_Substrate_CarbonMangels': 1,
    'CYP2D6_Veith': 1,
    # 'CYP3A4_Substrate_CarbonMangels': 1,
    'CYP3A4_Veith': 1,

    ## Excretion Tasks:
    'Clearance_Hepatocyte_AZ': 1,
    'Clearance_Microsome_AZ': 1,
    'Half_Life_Obach': 1,

    ## Toxicity Tasks:
    'AMES': 1,
    'Carcinogens_Lagunin': 1,
    'ClinTox': 1,
    'DILI': 1,
    'NR-AR-LBD': 1,
    'NR-AR': 1,
    'NR-AhR': 1,
    'NR-Aromatase': 1,
    'NR-ER-LBD': 1,
    'NR-ER': 1,
    'NR-PPAR-gamma': 1,
    'SR-ARE': 1,
    'SR-ATAD5': 1,
    'SR-HSE': 1,
    'SR-MMP': 1,
    'SR-p53': 1,
    'Skin_Reaction': 1,
    'hERG': 1,
    'LD50_Zhu': 1,    
}

admet_weights = {'absorption': 1,
                 'distribution': 1,
                 'metabolism': 1,
                 'excretion': 1,
                 'toxicity': 1}

def filter_admet(raw_admet_prediction_dir:str,
                 antibiotic_admet_prediction_dir:str,
                 filtering_output_dir:str,
                 ranking_output_dir:str,):

    raw_admet_prediction_df = pd.read_csv(raw_admet_prediction_dir)

    features = raw_admet_prediction_df.columns[1:]
    feature_ranks = [f"{item}_rank" for item in features]
    raw_admet_prediction_df['rank_sum'] = 0

    for feature in features:
        raw_admet_prediction_df[f'{feature}_rank'] = raw_admet_prediction_df[feature].rank()
        raw_admet_prediction_df['rank_sum'] += raw_admet_prediction_df[f'{feature}_rank']
    
    rank_df = raw_admet_prediction_df[['SMILES'] + feature_ranks + ['rank_sum']]
    rank_df = rank_df.sort_values(by=['rank_sum'],
                                  ascending=True)
    
    threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = get_admet_threshold(antibiotic_admet_prediction_dir)

    for key in threshold_antibiotic_dict:
        if threshold_type_dict[key] == 'lower':
            print(f"Threshold for {key} is Max={threshold_value_dict[key]}")
        else:
            print(f"Threshold for {key} is Min={threshold_value_dict[key]}")

    filtered_admet_prediction_df = raw_admet_prediction_df[['SMILES']+features.to_list()].copy()

    for key in threshold_value_dict:  
        if threshold_type_dict[key] == 'lower':
            filtered_admet_prediction_df['pass_'+key] = np.where(filtered_admet_prediction_df[key] <= threshold_value_dict[key],1,0)
        else:
            filtered_admet_prediction_df['pass_'+key] = np.where(filtered_admet_prediction_df[key] >= threshold_value_dict[key],1,0)

    absorption_tasks = ['Bioavailability_Ma','HIA_Hou', 'PAMPA_NCATS','Pgp_Broccatelli','Solubility_AqSolDB',
                        'Caco2_Wang','Lipophilicity_AstraZeneca']
    absorption_tasks = [f"pass_{item}" for item in absorption_tasks]

    distribution_tasks = ['BBB_Martins','PPBR_AZ','VDss_Lombardo']
    distribution_tasks = [f"pass_{item}" for item in distribution_tasks]

    metabolism_tasks = ['CYP1A2_Veith','CYP2C19_Veith','CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith',]
    metabolism_tasks = [f"pass_{item}" for item in metabolism_tasks]

    excretion_tasks = ['Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach']
    excretion_tasks = [f"pass_{item}" for item in excretion_tasks]

    toxicity_tasks = ['AMES','Carcinogens_Lagunin','ClinTox','DILI',
                      'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                      'NR-ER','NR-PPAR-gamma','SR-ATAD5','SR-HSE','SR-MMP','SR-p53',
                      'Skin_Reaction','hERG','LD50_Zhu','SR-ARE']
    toxicity_tasks = [f"pass_{item}" for item in toxicity_tasks]

    total_task_num = len(absorption_tasks) + len(distribution_tasks) + len(metabolism_tasks) + len(excretion_tasks) + len(toxicity_tasks)

    filtered_admet_prediction_df['fail_absorption'] = len(absorption_tasks) - filtered_admet_prediction_df[absorption_tasks].sum(axis=1)
    filtered_admet_prediction_df['fail_distribution'] = len(distribution_tasks) - filtered_admet_prediction_df[distribution_tasks].sum(axis=1)
    filtered_admet_prediction_df['fail_metabolism'] = len(metabolism_tasks) - filtered_admet_prediction_df[metabolism_tasks].sum(axis=1)
    filtered_admet_prediction_df['fail_excretion'] = len(excretion_tasks) - filtered_admet_prediction_df[excretion_tasks].sum(axis=1)
    filtered_admet_prediction_df['fail_toxicity'] = len(toxicity_tasks) - filtered_admet_prediction_df[toxicity_tasks].sum(axis=1)
    filtered_admet_prediction_df['fail_sum'] = total_task_num - filtered_admet_prediction_df[filtered_admet_prediction_df.columns[pd.Series(filtered_admet_prediction_df.columns).str.startswith('pass_')]].sum(axis=1)
    
    ## Calculating ADMET Scores
    filtered_admet_prediction_df['absorption_admet_score'] = 0
    for task in absorption_tasks:
        filtered_admet_prediction_df['absorption_admet_score'] += (filtered_admet_prediction_df[task] - 1) * task_weight_dict[task[5:]]
    
    filtered_admet_prediction_df['distribution_admet_score'] = 0
    for task in distribution_tasks:
        filtered_admet_prediction_df['distribution_admet_score'] += (filtered_admet_prediction_df[task] - 1) * task_weight_dict[task[5:]]
    
    filtered_admet_prediction_df['metabolism_admet_score'] = 0
    for task in metabolism_tasks:
        filtered_admet_prediction_df['metabolism_admet_score'] += (filtered_admet_prediction_df[task] - 1) * task_weight_dict[task[5:]]
    
    filtered_admet_prediction_df['excretion_admet_score'] = 0
    for task in excretion_tasks:
        filtered_admet_prediction_df['excretion_admet_score'] += (filtered_admet_prediction_df[task] - 1) * task_weight_dict[task[5:]]
    
    filtered_admet_prediction_df['toxicity_admet_score'] = 0
    for task in toxicity_tasks:
        filtered_admet_prediction_df['toxicity_admet_score'] += (filtered_admet_prediction_df[task] - 1) * task_weight_dict[task[5:]]
    
    filtered_admet_prediction_df = filtered_admet_prediction_df.sort_values(by=['fail_sum'],
                                                                            ascending=True)
    # filtered_admet_prediction_df = filtered_admet_prediction_df.sort_values(by=['admet_score'],
    #                                                                         ascending=False)

    filtered_admet_prediction_df.to_csv(ranking_output_dir,
                                        index=False)

    all_pass_df = filtered_admet_prediction_df[filtered_admet_prediction_df['fail_sum'] == 0]
    
    if len(all_pass_df) == 0:
        print(f"NO molecule passes all ADMET tests, please see '{ranking_output_dir}' for detailed filtering results.")

    all_pass_df.to_csv(filtering_output_dir,
                       index = False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_admet_out_dir', type=str,
                        help = 'The directory of raw ADMET prediction output')

    parser.add_argument('--filtered_admet_out_dir', type=str,
                        help = 'The directory of filtered ADMET prediction output')

    parser.add_argument('--antibiotic_admet_prediction_dir', type=str,
                        help = 'The directory of FDA approved antibiotic ADMET predictions for thresholding')

    parser.add_argument('--ranking_admet_out_dir', type=str,
                        help = 'ranking output directory')

    args = parser.parse_args()

    filter_admet(raw_admet_prediction_dir=args.raw_admet_out_dir,
                 antibiotic_admet_prediction_dir=args.antibiotic_admet_prediction_dir,
                 filtering_output_dir=args.filtered_admet_out_dir,
                 ranking_output_dir=args.ranking_admet_out_dir)
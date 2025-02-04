import pandas as pd
import numpy as np

lower_the_better = ['AMES', 'BBB_Martins','CYP1A2_Veith','CYP2C19_Veith','PPBR_AZ',
                    'CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith','Carcinogens_Lagunin',
                    'ClinTox','DILI','NR-AR-LBD','NR-AR','NR-AhR',
                    'NR-Aromatase','NR-ER-LBD','NR-ER','NR-PPAR-gamma','SR-ATAD5',
                    'SR-ARE','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                    'hERG','Pgp_Broccatelli','LD50_Zhu',]

higher_the_better = ['Bioavailability_Ma', 'HIA_Hou','PAMPA_NCATS','Caco2_Wang','Clearance_Hepatocyte_AZ',
                     'Clearance_Microsome_AZ','Half_Life_Obach','Lipophilicity_AstraZeneca','Solubility_AqSolDB',
                     'VDss_Lombardo']


def get_admet_threshold(current_antibiotic_admet_prediction_dir: str):

    threshold_value_dict = {}
    threshold_antibiotic_dict = {}
    threshold_type_dict = {}

    admet_prediction = pd.read_csv(current_antibiotic_admet_prediction_dir)

    for feature in lower_the_better:
        threshold_value_dict[feature] = admet_prediction[feature].quantile(0.9)
        threshold_antibiotic_dict[feature] = admet_prediction['Name']
        threshold_type_dict[feature] = 'lower'
    for feature in higher_the_better:
        threshold_value_dict[feature] = admet_prediction[feature].quantile(0.1)
        threshold_antibiotic_dict[feature] = admet_prediction['Name']
        threshold_type_dict[feature] = 'higher'
    
    return threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict
    

if __name__ == '__main__':

    current_antibiotic_admet_prediction_dir = '/home/aw274/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv'    
    threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = get_admet_threshold(current_antibiotic_admet_prediction_dir)

    print(threshold_value_dict)
    print(threshold_antibiotic_dict)
    print(threshold_type_dict)
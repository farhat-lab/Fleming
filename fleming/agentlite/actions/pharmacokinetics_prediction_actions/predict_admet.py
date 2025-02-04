import argparse 
from admet_ai import ADMETModel
import pandas as pd

def run_toxicity_model_ADMETAI(classifier_output_dir: str,
                               save_dir: str,
                               desired_properties: list = None,):

    classifier_output_df = pd.read_csv(classifier_output_dir)

    smiles_list = classifier_output_df['SMILES'].to_list()

    model = ADMETModel()
    preds = model.predict(smiles_list)

    all_property = ['AMES','BBB_Martins','Bioavailability_Ma','CYP1A2_Veith','CYP2C19_Veith',
                        'CYP2C9_Substrate_CarbonMangels','CYP2C9_Veith','CYP2D6_Substrate_CarbonMangels','CYP2D6_Veith','CYP3A4_Substrate_CarbonMangels',
                        'CYP3A4_Veith', 'Carcinogens_Lagunin' ,'ClinTox' ,'DILI','HIA_Hou',
                        'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                        'NR-ER','NR-PPAR-gamma','PAMPA_NCATS','Pgp_Broccatelli','SR-ARE',
                        'SR-ATAD5','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                        'hERG','Caco2_Wang','Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach',
                        'HydrationFreeEnergy_FreeSolv','LD50_Zhu','Lipophilicity_AstraZeneca','PPBR_AZ','Solubility_AqSolDB',
                        'VDss_Lombardo']
    
    if desired_properties is None:
        desired_properties = all_property

    # available_desired_properties = []
    # for _property in desired_properties:
    #     if _property not in all_property:
    #         print(_property, "not in available property list, thus removed")
    #     else:
    #         available_desired_properties += _property

    preds = preds[desired_properties]
    preds.insert(0,'SMILES',smiles_list)

    preds.to_csv(save_dir,
                 index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--inhibition_out_dir', type=str,
                        help = "The directory of M.tb inhibition prediction output")
    
    parser.add_argument('--desired_properties', type=str,
                        help = "The desired property list")

    parser.add_argument('--admet_out_dir', type=str,
                        help = "The directory of ADMET prediction output")

    args = parser.parse_args()
    
    run_toxicity_model_ADMETAI(classifier_output_dir = args.inhibition_out_dir,
                               desired_properties=args.desired_properties,
                               save_dir = args.admet_out_dir)
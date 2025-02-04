import pandas as pd
from admet_ai import ADMETModel
import rdkit.Chem as Chem
from tqdm import tqdm

cheese_property_thresholds = {'Caco2_Wang':[-5.15], 
                              'Clearance_Hepatocyte_AZ':['>=5'],
                              'Clearance_Microsome_AZ':['>=5'], 
                              'Half_Life_Obach':['>=3'],
                              'LD50_Zhu':[-2.4], 
                              'Lipophilicity_AstraZeneca':[0.0,3.0],
                              'PPBR_AZ':['<=90'], 
                              'Solubility_AqSolDB':[-4.0,0.5], 
                              'VDss_Lombardo':[0.0-20.0]}

lower_the_better = ['AMES', 'BBB_Martins','CYP1A2_Veith','CYP2C19_Veith','PPBR_AZ',
                    'CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith','Carcinogens_Lagunin',
                    'ClinTox','DILI','NR-AR-LBD','NR-AR','NR-AhR',
                    'NR-Aromatase','NR-ER-LBD','NR-ER','NR-PPAR-gamma','SR-ATAD5',
                    'SR-ARE','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                    'hERG','Pgp_Broccatelli','LD50_Zhu',]

higher_the_better = ['Bioavailability_Ma', 'HIA_Hou','PAMPA_NCATS','Caco2_Wang','Clearance_Hepatocyte_AZ',
                     'Clearance_Microsome_AZ','Half_Life_Obach','Lipophilicity_AstraZeneca','Solubility_AqSolDB',
                     'VDss_Lombardo']


def get_current_antibiotics_admet_predictions(current_antibiotic_dir: str,
                                              save_dir:str):
    
    current_antibiotic_df = pd.read_excel(current_antibiotic_dir)

    smiles = []
    names = []
    for idx, row in current_antibiotic_df.iterrows():
        if pd.isnull(row['Isomeric_SMILES']):
            print(f"No Isomeric SMILES for {row['Antibiotics_Name']}, using Canonical SMILES")
            molecule = Chem.MolFromSmiles(row['Canonical_SMILES'])
            if molecule is not None:                
                smiles.append(row['Canonical_SMILES'])
                names.append(row['Antibiotics_Name'])
            else:
                print(f"Invalid SMILES structure for {row['Antibiotics_Name']}, skipping")
        else:
            molecule = Chem.MolFromSmiles(row['Isomeric_SMILES'])
            if molecule is not None:
                smiles.append(row['Isomeric_SMILES'])
                names.append(row['Antibiotics_Name'])
            else:
                print(f"Invalid Isomeric SMILES structure for {row['Antibiotics_Name']}, trying Canonical SMILES")
                molecule = Chem.MolFromSmiles(row['Canonical_SMILES'])
                if molecule is not None:
                    smiles.append(row['Canonical_SMILES'])
                    names.append(row['Antibiotics_Name'])
                else:
                    print(f"Invalid SMILES structure for {row['Antibiotics_Name']}, skipping")

    model = ADMETModel()
    preds = model.predict(smiles)

    desired_property = ['AMES','BBB_Martins','Bioavailability_Ma','CYP1A2_Veith','CYP2C19_Veith',
                        'CYP2C9_Substrate_CarbonMangels','CYP2C9_Veith','CYP2D6_Substrate_CarbonMangels','CYP2D6_Veith','CYP3A4_Substrate_CarbonMangels',
                        'CYP3A4_Veith', 'Carcinogens_Lagunin' ,'ClinTox' ,'DILI','HIA_Hou',
                        'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                        'NR-ER','NR-PPAR-gamma','PAMPA_NCATS','Pgp_Broccatelli','SR-ARE',
                        'SR-ATAD5','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                        'hERG','Caco2_Wang','Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach',
                        'HydrationFreeEnergy_FreeSolv','LD50_Zhu','Lipophilicity_AstraZeneca','PPBR_AZ','Solubility_AqSolDB',
                        'VDss_Lombardo']

    preds = preds[desired_property]
    preds.insert(0,'SMILES',smiles)
    preds.insert(0,'Name',names)

    preds.to_csv(save_dir,
                 index = False)


if __name__ == '__main__':

    current_antibiotic_dir = '/n/data1/hms/dbmi/farhat/alex/admet/Current_Antibiotic.xlsx'
    save_dir = '/home/aw274/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv'
    get_current_antibiotics_admet_predictions(current_antibiotic_dir,
                                              save_dir=save_dir)
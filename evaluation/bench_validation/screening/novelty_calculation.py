import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np

# Bench Result DataFrame
bench_validation_results = pd.read_excel('data/output/dragonfold/screening_result/Charm_mols_WCA_nov24.xlsx',
                                         sheet_name='CDD Export 1')

bench_validation_results = bench_validation_results[['SMILES', 
                                                     'Mtb_WCA_mc2-7000_Resazurin: concentration (uM)', 
                                                     'Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation (%)',
                                                     'Mtb_WCA_mc2-7000_Resazurin: EC50 (uM)',
                                                     'Mtb_WCA_mc2-7000_Resazurin: EC90 (uM)']]

# Prediction DataFrame
final_output = pd.read_csv('data/output/dragonfold/screening_result/final_output.csv')
# print(final_output)

# Drop Duplicates
max_concentration_results = bench_validation_results.loc[bench_validation_results.groupby('SMILES')['Mtb_WCA_mc2-7000_Resazurin: % inhibition or activation (%)'].idxmax()]
max_concentration_results['predicted_hit'] = np.where(max_concentration_results['SMILES'].isin(final_output['SMILES']),"Predicted Hit","Predicted Non-hit")

## Tanimoto Similarity with Training
training_df = pd.read_csv('data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')

max_sim_list = []
for bv_smiles in tqdm(max_concentration_results['SMILES']):
    bv_mol = Chem.MolFromSmiles(bv_smiles)
    bv_fp = AllChem.GetMorganFingerprintAsBitVect(bv_mol,2,nBits=2048)

    sim_list = []
    for train_smiles in training_df['SMILES']:
        train_mol = Chem.MolFromSmiles(train_smiles)
        train_fp = AllChem.GetMorganFingerprintAsBitVect(train_mol,2,nBits=2048)
        sim = AllChem.DataStructs.TanimotoSimilarity(bv_fp,train_fp)
        sim_list.append(sim)

    max_sim_list.append(max(sim_list))

max_concentration_results['train_max_sim'] = max_sim_list

max_concentration_results.to_csv('evaluation/bench_validation/screening/output/train_similarity.csv',
                                 index = False)
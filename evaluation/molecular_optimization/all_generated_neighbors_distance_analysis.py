import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

pd.set_option('display.max_columns', None)

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)


for step in range(1,6):
    
    siblings = pd.read_csv(f"data/output/generative/molecular_optimization_result_v2/step_{step}/sibling_dict.csv")
    siblings = siblings[['SMILES','seed']]
    if step == 1:
        siblings.rename(columns={'SMILES':f'siblings_step_{step}',
                                 'seed':'original_seed'}, inplace=True)
        steps_df = siblings.copy()
    else:
        siblings.rename(columns={'SMILES':f'siblings_step_{step}',
                                 'seed':f'siblings_step_{step-1}'}, inplace=True)
        steps_df = steps_df.merge(siblings, 
                                  on=f'siblings_step_{step-1}',
                                  how='left').drop_duplicates().reset_index(drop=True)

    steps_df = steps_df.dropna().drop_duplicates().reset_index(drop=True)

    # print(steps_df[steps_df['original_seed']=='CCCCc1ccc(N(C)CCCC2CCCC2)cc1'])
        
    # Similarity to original seed
    sim_list = []
    for i, mol in enumerate(steps_df[f'siblings_step_{step}']):
        sib_fp = get_morgan_fp(mol)
        seed_fp = get_morgan_fp(steps_df['original_seed'][i])       
        sim_list.append(TanimotoSimilarity(sib_fp, seed_fp))

    steps_df[f'Step_{step}_Tanimoto_Similarity'] = sim_list

    # print(f"########## Step {step} ##########")

    change_df = steps_df[steps_df[f'Step_{step}_Tanimoto_Similarity'] < 1]
    change_df = change_df[['original_seed',f'Step_{step}_Tanimoto_Similarity']].groupby('original_seed',as_index=False).median()

    print("Changed Molecules: ", len(change_df['original_seed'].unique()), f"({len(change_df['original_seed'].unique())/200*100:.2f}%)")

    print("Mean Similarity of Changed Molecules: ", change_df[f'Step_{step}_Tanimoto_Similarity'].mean())
    print("Median Similarity of Changed Molecules: ", change_df[f'Step_{step}_Tanimoto_Similarity'].median())
    print("Standard Deviation of Similarity of Changed Molecules: ", change_df[f'Step_{step}_Tanimoto_Similarity'].std())
    print("75% Quantile of Similarity of Changed Molecules: ", np.percentile(change_df[f'Step_{step}_Tanimoto_Similarity'], 75))
    print("25% Quantile of Similarity of Changed Molecules: ", np.percentile(change_df[f'Step_{step}_Tanimoto_Similarity'], 25))
 


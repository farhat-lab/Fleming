import synspace
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import multiprocessing
import argparse

def generate_siblings_SMILES(smiles:str):
    mols, sibling_props = synspace.chemical_space(smiles,
                                                  num_samples = 2000,)
    sibling_smiles = [Chem.MolToSmiles(item) for item in mols]
    return smiles, sibling_smiles, sibling_props

def generate_siblings(smiles_dir:str,
                      sibling_smiles_save_dir:str,
                      sibling_dict_save_dir:str,):

    print("Generating Siblings ...")

    smiles_df = pd.read_csv(smiles_dir)
    smiles_list = smiles_df['SMILES'].tolist()

    sibling_smiles_list = []
    seed_smiles_list = []
    sibling_props_list = []

    for smiles in tqdm(smiles_list):
        seed_smiles, sibling_smiles, sibling_props = generate_siblings_SMILES(smiles)
        seed_smiles_list = seed_smiles_list + [seed_smiles] * len(sibling_smiles)
        sibling_smiles_list = sibling_smiles_list + sibling_smiles
        sibling_props_list = sibling_props_list + sibling_props
    
    sibling_dict_df = pd.DataFrame(columns=['SMILES','seed'])
    sibling_dict_df['SMILES'] = sibling_smiles_list
    sibling_dict_df['seed'] = seed_smiles_list
    sibling_dict_df['props'] = sibling_props_list
    
    sibling_df = sibling_dict_df[['SMILES']].copy()

    ## Filter for Valid SMILES
    valid_smiles = []
    invalid_smiles = []
    for smile in tqdm(sibling_df['SMILES']):
        m = Chem.MolFromSmiles(smile)
        if m is not None:
            valid_smiles.append(smile)                
        else:
            invalid_smiles.append(smile)

    sibling_df['SMILES'] = valid_smiles

    sibling_df.to_csv(sibling_smiles_save_dir,
                      index = False)
    sibling_dict_df.to_csv(sibling_dict_save_dir,
                      index = False)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_smiles_dir', type=str,
                        help = "The directory of seed smiles")

    parser.add_argument('--sibling_df_save_dir', type=str,
                        help = "The output sibling df directory of molecular optimization")

    parser.add_argument('--sibling_dict_save_dir', type=str,
                        help = "The output sibling dictionary directory of molecular optimization")

    args = parser.parse_args()

    generate_siblings(smiles_dir=args.seed_smiles_dir,
                      sibling_smiles_save_dir=args.sibling_df_save_dir,
                      sibling_dict_save_dir=args.sibling_dict_save_dir)    
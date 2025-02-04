import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import multiprocessing
import argparse
from rdkit.Chem.Descriptors import qed

def sibling_QC(sibling_smiles_save_dir:str):

    print("Running Sibling QC ...")

    siblings_df = pd.read_csv(sibling_smiles_save_dir)

    qualified_smiles = []
    for smile in siblings_df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smile)
            qed(mol)
            qualified_smiles.append(smile)

        except:
            print(smile, "Fails QC and removes")
    
    print(len(qualified_smiles), "Qualified Siblings")

    siblings_df = pd.DataFrame(qualified_smiles,
                               columns = ['SMILES'])

    siblings_df.to_csv(sibling_smiles_save_dir,
                       index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sibling_df_save_dir', type=str,
                        help = "The output sibling df directory of molecular optimization")

    args = parser.parse_args()

    sibling_QC(sibling_smiles_save_dir=args.sibling_df_save_dir)    
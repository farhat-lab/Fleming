from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
import multiprocessing
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--target_molecule_dir', type=str,
                    help = "Target Molecule Directory")
parser.add_argument('--output_dir', type=str,
                    help = "Output Directory with the most alike molecule and the similarity")
parser.add_argument('--start_idx', type=int,
                    help = 'Start Index')
parser.add_argument('--end_idx', type=int,
                    help = 'End Index')

args = parser.parse_args()
target_smiles = pd.read_csv(args.target_molecule_dir)

target_smiles = list(target_smiles['SMILES'])

fp_list = []
for smile in target_smiles:
    mol = Chem.MolFromSmiles(smile)
    fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

similarity_list = []
def tanimoto_similarity(target_fp: str):
    output = []
    for reference_fp in fp_list:
        output.append(DataStructs.TanimotoSimilarity(target_fp, reference_fp))
    return output

if args.end_idx < len(fp_list):
    batch_fp_list = fp_list[args.start_idx: args.end_idx]
else:
    batch_fp_list = fp_list[args.start_idx: len(fp_list)]

with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
    with tqdm(total=len(batch_fp_list)) as pbar:
        for output in p.imap(tanimoto_similarity, batch_fp_list):
            similarity_list.append(output)
            pbar.update()

similarity_array = np.array(similarity_list)

np.save(args.output_dir, similarity_array)


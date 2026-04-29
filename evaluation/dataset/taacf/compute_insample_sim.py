import pandas as pd 
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm


# Load pre-split train/test datasets
train_mlsmr = pd.read_csv('data/FlemingDev/experiments/compare_datasets/mlsmr_train.csv')
test_mlsmr = pd.read_csv('data/FlemingDev/experiments/compare_datasets/mlsmr_test.csv')
val_mlsmr = pd.read_csv('data/FlemingDev/experiments/compare_datasets/mlsmr_val.csv')

train_taacf = pd.read_csv('data/FlemingDev/experiments/compare_datasets/taacf_train.csv')
val_taacf = pd.read_csv('data/FlemingDev/experiments/compare_datasets/taacf_val.csv')
test_taacf = pd.read_csv('data/FlemingDev/experiments/compare_datasets/taacf_test.csv')

# Remove NaNsmiles
print('removing nans')
for df in [train_mlsmr, test_mlsmr, val_mlsmr, train_taacf, test_taacf, val_taacf]:
    df.dropna(subset=['SMILES'], inplace=True)
    df['SMILES'] = df['SMILES'].astype(str)

# Build dataset registry using preloaded splits
mlsmr = pd.concat([train_mlsmr, test_mlsmr, val_mlsmr])
taacf = pd.concat([train_taacf, test_taacf, val_taacf])

mlsmr = mlsmr.drop_duplicates(subset=['SMILES'])
taacf = taacf.drop_duplicates(subset=['SMILES'])

mlsmr_taacf = pd.concat([mlsmr, taacf]).drop_duplicates(subset=['SMILES'])
mlsmr_taacf = mlsmr_taacf.reset_index(drop=True)
# mlsmr_taacf = mlsmr_taacf[:10]
print(f'Number of unique molecules in combined mlsmr and taacf dataset: {len(mlsmr_taacf)}')

def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

# Generate fingerprints
print("Generating fingerprints for mlsmr_taacf...")
mlsmr_taacf_fps = [get_morgan_fp(s) for s in tqdm(mlsmr_taacf['SMILES'])]

mlsmr_taacf_smiles = mlsmr_taacf['SMILES'].tolist()

# Filter out invalid SMILES

mlsmr_taacf_valid_fps = []
mlsmr_taacf_valid_smiles = []
for fp, smiles in zip(mlsmr_taacf_fps, mlsmr_taacf_smiles):
    if fp is not None:
        mlsmr_taacf_valid_fps.append(fp)
        mlsmr_taacf_valid_smiles.append(smiles)

# Calculate Maximum Pairwise Tanimoto Similarity (not including self similarity)
mlsmr_taacf_max_similarities = []
for i, fp in enumerate(tqdm(mlsmr_taacf_valid_fps)):
    # Get index of the current fingerprint in the combined dataset to avoid self-comparison
    other_fps = mlsmr_taacf_valid_fps[:i] + mlsmr_taacf_valid_fps[i+1:]
    similarities = BulkTanimotoSimilarity(fp, other_fps)
    max_sim = max(similarities) if similarities else 0.0
    mlsmr_taacf_max_similarities.append(max_sim)

pd.DataFrame({
    'SMILES': mlsmr_taacf_valid_smiles,
    'Max_Tanimoto_Similarity': mlsmr_taacf_max_similarities
}).to_csv('evaluation/dataset/taacf/mlsmr_taacf_max_similarities.csv', index=False)


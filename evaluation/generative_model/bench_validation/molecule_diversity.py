import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)

final_mols = pd.read_csv("/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/bench_validation/train_similarity.csv")

final_mols = final_mols['SMILES'].tolist()
final_mols_fps = [get_morgan_fp(s) for s in final_mols]
final_mols_valid_fps = []
final_mols_valid_smiles = []
for fp, smiles in zip(final_mols_fps, final_mols):
    if fp is not None:
        final_mols_valid_fps.append(fp)
        final_mols_valid_smiles.append(smiles)

# Calculate Maximum Pairwise Tanimoto Similarity (not including self similarity) and Save the most similar molecule and its similarity score
final_mols_max_similarities = []
most_similar_molecules = []
for i, fp in enumerate(final_mols_valid_fps):
    # Get index of the current fingerprint in the combined dataset to avoid self-comparison
    similarities = BulkTanimotoSimilarity(fp, final_mols_valid_fps)
    similarities[i] = -1  # Exclude self-similarity by setting it to -1
    max_similarity = max(similarities)
    final_mols_max_similarities.append(max_similarity)
    max_index = similarities.index(max_similarity)
    most_similar_mol = final_mols_valid_smiles[max_index]
    most_similar_molecules.append(most_similar_mol)
    
final_mols_df = pd.DataFrame({
    'SMILES': final_mols_valid_smiles,
    'Most_Similar_Molecule': most_similar_molecules,
    'Max_Tanimoto_Similarity': final_mols_max_similarities
})

print(f"Mean Maximum Similarity: {np.mean(final_mols_max_similarities)}")
print(f"Median Maximum Similarity: {np.median(final_mols_max_similarities)}")
print(f"Standard Deviation of Maximum Similarity: {np.std(final_mols_max_similarities)}")
print(f"75% Quantile of Maximum Similarity: {np.percentile(final_mols_max_similarities, 75)}")
print(f"25% Quantile of Maximum Similarity: {np.percentile(final_mols_max_similarities, 25)}")

print(final_mols_df)
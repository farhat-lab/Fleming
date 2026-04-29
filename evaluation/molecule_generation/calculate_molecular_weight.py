from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import pandas as pd
import numpy as np

data = pd.read_csv('data/output/generative/molecular_optimization_result/top_generated_molecules.csv')

mass = []
for smiles in data['SMILES']:
    mass.append(ExactMolWt(Chem.MolFromSmiles(smiles)))

print("Top 100: ", np.mean(mass[:100]), np.std(mass[:100]))
print("Top 10: ", np.mean(mass[:10]), np.std(mass[:10]))
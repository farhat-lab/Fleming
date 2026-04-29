import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

## Bioactives
bioactives = pd.read_csv('bioactive_mols.csv')
metabolites = pd.read_csv('metabolites.csv')

## 5 Selected Molecules
generated_molecules = pd.read_excel('inna_selected_5_mols.xlsx')

gen_fp = []
for smiles in generated_molecules['SMILES']:
    gen_mol = Chem.MolFromSmiles(smiles)
    gen_fp.append(AllChem.GetMorganFingerprintAsBitVect(gen_mol,2,nBits=2048))

## Bioactive Similarity
bioactive_fp = []
for smiles in tqdm(bioactives['smiles']):
    mol = Chem.MolFromSmiles(smiles)
    bioactive_fp.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048))

bio_maximum_sim = []
bio_maximum_sim_mol = []
for fp in tqdm(gen_fp):
    sim = []
    for target_fp in bioactive_fp:
        sim.append(DataStructs.TanimotoSimilarity(target_fp, fp))

    bio_maximum_sim.append(max(sim))
    bio_maximum_sim_mol.append(list(bioactives['smiles'])[sim.index(max(sim))])

generated_molecules['bioactive_max_sim'] = bio_maximum_sim
generated_molecules['most_similar_bioactive'] = bio_maximum_sim_mol

## Metabolites Similarity
metabolites_fp = []
for smiles in tqdm(metabolites['smiles']):
    mol = Chem.MolFromSmiles(smiles)
    metabolites_fp.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048))

met_maximum_sim = []
met_maximum_sim_mol = []
for fp in tqdm(gen_fp):
    sim = []
    for target_fp in metabolites_fp:
        sim.append(DataStructs.TanimotoSimilarity(target_fp, fp))

    met_maximum_sim.append(max(sim))
    met_maximum_sim_mol.append(list(metabolites['smiles'])[sim.index(max(sim))])

generated_molecules['metabolites_max_sim'] = met_maximum_sim
generated_molecules['most_similar_metabolites'] = met_maximum_sim_mol

generated_molecules.to_csv('evaluation/bench_validation/designed_mols/output/bioactive_metabolite_similarity.csv',
                           index = False)
    


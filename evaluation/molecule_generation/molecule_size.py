import pandas as pd
import numpy as np
from rdkit import Chem
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

def get_num_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()  

# Diffusion Generative Molecules
generative_mols = pd.read_csv("data/output/generative/diffusion_generated/diffusion_generated_molecules_20241001.csv")
generative_mols['num_atoms'] = generative_mols['SMILES'].apply(get_num_atoms)
print(f"Diffusion Generative Model: Mean Num Atoms = {np.mean(generative_mols['num_atoms'])} +/- {np.std(generative_mols['num_atoms'])}, Median Num Atoms = {np.median(generative_mols['num_atoms'])} [{np.quantile(generative_mols['num_atoms'],[0.25,0.75])}]")

# Synthemol
synthemol_mols = pd.read_csv("data/output/generative/synthemol_generated/molecules.csv")
synthemol_mols['num_atoms'] = synthemol_mols['SMILES'].apply(get_num_atoms)
print(f"Synthemol Model: Mean Num Atoms = {np.mean(synthemol_mols['num_atoms'])} +/- {np.std(synthemol_mols['num_atoms'])}, Median Num Atoms = {np.median(synthemol_mols['num_atoms'])} [{np.quantile(synthemol_mols['num_atoms'],[0.25,0.75])}]")

# Statistical Test
print("p-value = ", mannwhitneyu(generative_mols['num_atoms'], synthemol_mols['num_atoms']).pvalue)

# Plot Histogram
plt.figure(figsize=(6.4,6.4))
ax = plt.subplot(111)
plt.rcParams['patch.edgecolor'] = 'none'

sns.histplot(x="num_atoms", 
             data=generative_mols,
             alpha=0.1,
             stat="density",)
sns.histplot(x="num_atoms", 
             data=synthemol_mols,
             alpha=0.1,
             stat="density",)

sns.kdeplot(
    x="num_atoms", 
    data=generative_mols,
    linewidth=2,
    label='Diffusion Generated Molecules'
)
sns.kdeplot(
    x="num_atoms", 
    data=synthemol_mols,
    linewidth=2,
    label='Synthemol Generated Molecules'
)


plt.xlabel('Number of Atoms', fontsize=14)
plt.ylabel('Density', fontsize=14)
ax.spines[['right','top']].set_visible(False)
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig('evaluation/molecule_generation/output/molecule_size.png')
plt.savefig('evaluation/molecule_generation/output/molecule_size.pdf')
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
import pandas as pd 

sample_molecule_df = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/generative/molecular_optimization_result/top_generated_molecules.csv')
sample_molecule_df = sample_molecule_df[:20]
rank = 1

for input_smiles in sample_molecule_df['SMILES']:

    mol = Chem.MolFromSmiles(input_smiles)
    clean_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

    valid_smiles = Chem.MolToSmiles(mol)
    print("Plotting: ", valid_smiles)

    plt.figure()
    img = Draw.MolToImage(mol,
                          kekulize=True,
                          fitImage=True,
                          fitWidth=True)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.grid(False)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/sample_molecule_plots/rank{rank}_{clean_smiles}.png')
    plt.close()

    rank += 1



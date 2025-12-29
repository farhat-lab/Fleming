from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D

# final_mol = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/inna_selected_5_mols.csv')

# for smile in final_mol['SMILES']:
#     mol = Chem.MolFromSmiles(smile)
#     img = Draw.MolToFile(mol,
#                          f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/mol_plots/{smile}.svg')

# most_sim_mol = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/max_similarities.csv')

# for idx, smile in enumerate(most_sim_mol['best_match_smiles']):
#     mol = Chem.MolFromSmiles(smile)
#     img = Draw.MolToFile(mol,
#                          f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/mol_plots/{idx}_{smile}_atom_size_fixed.svg')

most_sim_mol = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/max_similarities.csv')
output_dir = '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/mol_plots'

for idx, smile in enumerate(most_sim_mol['best_match_smiles']):
    mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol, clearAromaticFlags=True)

    # Standardize coordinates
    AllChem.Compute2DCoords(mol)

    # Fix figure size
    width, height = 300, 300
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()

    # Fix atom and bond sizes
    options.fixedFontSize = 10
    options.minFontSize = 10
    options.bondLineWidth = 1.5
    options.scaleBondWidth = False
    options.useMolBlockCoords = True  # Don't rescale to fit canvas
    options.padding = 0.2  # White border fraction

    # Draw and export
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open(f"{output_dir}/{idx}_atom_fixed.svg", "w") as f:
        f.write(svg)

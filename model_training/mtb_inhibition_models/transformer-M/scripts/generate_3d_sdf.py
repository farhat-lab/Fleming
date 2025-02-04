import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np

def generate_3d_structure(smiles_file_dir, save_file_dir):
    
    smiles_df = pd.read_csv(smiles_file_dir)

    smiles = [smi for smi in smiles_df.SMILES]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_df.SMILES]
    tb_inhibition = [label for label in smiles_df.inhibition]

    print(len(mols))
    print(len(smiles))
    print(len(tb_inhibition))

    hmols = []
    for m in range(len(mols)):
        if mols[m] is not None:
            hmols.append(Chem.AddHs(mols[m]))
        else:
            smiles[m] = None
            tb_inhibition[m] = None
            continue

    smiles = [i for i in smiles if i is not None]
    tb_inhibition = [i for i in tb_inhibition if i is not None]

    print(len(hmols))
    print(len(smiles))
    print(len(tb_inhibition))

    optimize_status_list = []
    failed_mol_index = []

    for i in tqdm(range(len(hmols))):
        embed_status = AllChem.EmbedMolecule(hmols[i], AllChem.ETKDG())
        if embed_status != 0:
            hmols[i] = None
            smiles[i] = None
            tb_inhibition[i] = None
            failed_mol_index.append(i)
            continue
        else:
            optimize_status = AllChem.UFFOptimizeMolecule(hmols[i],1000)
            optimize_status_list.append(optimize_status)

            if optimize_status != 0:
                print(f"Optimization Failed on Molecule Index: {i}")
                hmols[i] = None
                smiles[i] = None
                tb_inhibition[i] = None
                # print(hmols[i])
    
    hmols = [i for i in hmols if i is not None]
    smiles = [i for i in smiles if i is not None]
    tb_inhibition = [i for i in tb_inhibition if i is not None]

    print(len(hmols))
    print(len(smiles))
    print(len(tb_inhibition))

    print(failed_mol_index)
    
    print(f"RDKit Optimization failed on {sum(optimize_status_list)} Molecules")
    
    # smiles = list(smiles_df.SMILES)
    writer = Chem.SDWriter(save_file_dir)

    print(f">>>>> Successfully Converted {len(hmols)} molecules into 3-D Structure >>>>>")

    print(">>>>> Saving to File >>>>>")
    for n in tqdm(range(len(hmols))):
        hmols[n].SetProp("_SMILES","%s"%smiles[n])
        hmols[n].SetProp("_tb_inhibition","%s"%tb_inhibition[n])
        writer.write(hmols[n])
    
    writer.close()

    np.save('/n/scratch/users/a/aw274/data/tb_data/raw_data/3d_conversion_failure_index.npy', np.array(failed_mol_index), allow_pickle=True)

    return hmols

# hmols = generate_3d_structure(smiles_file_dir = '/n/scratch/users/a/aw274/data/tb_data/raw_data/Batch1and2_data.csv',
#                       save_file_dir = '/n/scratch/users/a/aw274/data/tb_data/raw_data/Batch1and2_3d_data.sdf')

hmols = generate_3d_structure(smiles_file_dir = '/n/scratch/users/a/aw274/data/tb_data/raw_data/Batch1and2_data.csv',
                              save_file_dir = '/n/scratch/users/a/aw274/data/tb_data/raw_data/TB_Inhibition_3d_data_with_labels.sdf')

# hmols = Chem.SDMolSupplier('/n/scratch/users/a/aw274/data/tb_data/raw_data/Batch1and2_3d_data.csv')

# print(hmols[0].GetProp("_tb_inhibition"))

# print(hmols[27451])
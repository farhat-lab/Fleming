import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

train_data = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')
train_3d_data = Chem.SDMolSupplier('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/final_tb_data_114933_3d.sdf',
                                   removeHs = False)

# target_split = [5.0,7.0,10.0,15.0]
target_split = [1.0,2.0,3.0,4.0,6.0,8.0,9.0,11.0,12.0,13.0,14.0,16.0,17.0,18.0,19.0,20.0] # Run for the remaining splits

for split_id in tqdm(target_split):
    split = np.load(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/{str(split_id)}_independent_set.npy')
    split_data = train_data.iloc[split,:]

    # 2D Data
    split_data = split_data.reset_index(drop=True)
    split_data.to_csv(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/{str(split_id)}_data.csv',
                      index = False)

    # 3D Data
    split_3d_data = []
    for idx, mol in enumerate(train_3d_data):
        if mol.GetProp('_SMILES') in list(split_data['SMILES']):
            split_3d_data.append(mol)
    
    print(len(split_data))
    print(len(split_3d_data))

    writer = Chem.SDWriter(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/{str(split_id)}_3d_data.sdf')
    for molecule in split_3d_data:
        writer.write(molecule, confId=-1)
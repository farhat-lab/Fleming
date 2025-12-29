from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm

train_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/Binary_Inhibition_114850_mols_inclusive.csv')
train_smiles = train_df['SMILES']
synthemol_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/synthemol_generated/ensemble_pass_both.csv')
synthemol_smiles = synthemol_df['SMILES']
diffusion_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/diffusion_generated/ensemble_pass_both.csv')
diffusion_smiles = diffusion_df['SMILES']

def mols_to_fps(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols if m is not None]

train_fps = mols_to_fps(train_smiles)
synthemol_fps = mols_to_fps(synthemol_smiles)
diffusion_fps = mols_to_fps(diffusion_smiles)

# === Compute pairwise similarities ===
def compute_max_tanimoto(generated_fps, train_fps):
    max_sims = []
    for fp in tqdm(generated_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sims.append(max(sims))
    return np.array(max_sims)

synthemol_max_sim = compute_max_tanimoto(synthemol_fps, train_fps)
synthemol_df['max_sim'] = synthemol_max_sim
synthemol_df.to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/synthemol_generated/with_sim.csv',index=False)

diffusion_max_sim = compute_max_tanimoto(diffusion_fps, train_fps)
diffusion_df['max_sim'] = diffusion_max_sim
diffusion_df.to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/diffusion_generated/with_sim.csv', index=False)
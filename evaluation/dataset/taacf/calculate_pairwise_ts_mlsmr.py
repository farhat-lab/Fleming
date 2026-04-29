import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm
mlsmr1 = pd.read_csv('data/FlemingDev/model_training/Pubchem_TB_Data/AID_1332_datatable.csv', header=3)
mlsmr2 = pd.read_csv('data/FlemingDev/model_training/Pubchem_TB_Data/AID_1626_datatable.csv', header=3)
mlsmr_total = pd.concat([mlsmr1, mlsmr2], ignore_index=True)
mlsmr_total = pd.concat([mlsmr1, mlsmr2])
mlsmr_total = mlsmr_total[['Unnamed: 3', 'Unnamed: 4']]
mlsmr_total.columns = ['SMILES', 'inhibition']
mlsmr_total = mlsmr_total.loc[mlsmr_total['inhibition'] != 'Inconclusive']
mlsmr_total['inhibition'] = mlsmr_total['inhibition'].apply(lambda x: 1 if x == 'Active' else 0)
mlsmr_total


our_data =pd.read_csv('data/inhibition_training_data/Binary_Inhibition_114850_mols_inclusive.csv')


taacf = pd.read_csv('data/FlemingDev/model_training/Pubchem_TB_Data/AID_1949_datatable.csv', header=3)
taacf2 = pd.read_csv('data/FlemingDev/model_training/Pubchem_TB_Data/AID_2842_datatable.csv', header=3)
taacf = pd.concat([taacf, taacf2])
taacf = taacf[['Unnamed: 3', 'Unnamed: 4']]
taacf.columns = ['SMILES', 'inhibition']
taacf = taacf.loc[taacf['inhibition'] != 'Inconclusive']
taacf['inhibition'] = taacf['inhibition'].apply(lambda x: 1 if x == 'Active' else 0)
from sklearn.model_selection import train_test_split

# First split into train+val and test sets


# Load pre-split train/test datasets
train_our = pd.read_csv('data/FlemingDev/data/data_original_from_alex/train_data.csv')
test_our = pd.read_csv('data/FlemingDev/data/data_original_from_alex/test_data.csv')
val_our = pd.read_csv('data/FlemingDev/data/data_original_from_alex/val_data.csv')
train_mlsmr = pd.read_csv('data/FlemingDev/experiments/compare_datasets/mlsmr_train.csv')
test_mlsmr = pd.read_csv('data/FlemingDev/experiments/compare_datasets/mlsmr_test.csv')
val_mlsmr = pd.read_csv('data/FlemingDev/experiments/compare_datasets/mlsmr_val.csv')
train_taacf = pd.read_csv('data/FlemingDev/experiments/compare_datasets/taacf_train.csv')
val_taacf = pd.read_csv('data/FlemingDev/experiments/compare_datasets/taacf_val.csv')
# If taacf_test is needed, add its path and loading here
test_taacf = pd.read_csv('data/FlemingDev/experiments/compare_datasets/taacf_test.csv')

# Remove NaNsmiles
print('removing nans')
for df in [train_our, test_our, val_our, train_mlsmr, test_mlsmr, val_mlsmr, train_taacf, test_taacf, val_taacf]:
    df.dropna(subset=['SMILES'], inplace=True)
    df['SMILES'] = df['SMILES'].astype(str)

# Build dataset registry using preloaded splits

mlsmr = pd.concat([train_mlsmr, test_mlsmr, val_mlsmr])
taacf = pd.concat([train_taacf, test_taacf, val_taacf])
our_data = pd.concat([train_our, test_our, val_our])

mlsmr = mlsmr.drop_duplicates(subset=['SMILES'])
taacf = taacf.drop_duplicates(subset=['SMILES'])
our_data = our_data.drop_duplicates(subset=['SMILES'])


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def main():
    parser = argparse.ArgumentParser(description='Calculate maximum pairwise Tanimoto similarity.')
    parser.add_argument('dataset', type=str, choices=['mlsmr', 'taacf'], help='The dataset to compare against (mlsmr or taacf).')
    args = parser.parse_args()

    if args.dataset == 'mlsmr':
        other_df = mlsmr
        dataset_name = 'mlsmr'
    elif args.dataset == 'taacf':
        other_df = taacf
        dataset_name = 'taacf'

    print("Generating fingerprints for our_data...")
    our_data_fps = [get_morgan_fp(s) for s in tqdm(our_data['SMILES'])]
    print(f"Generating fingerprints for {dataset_name}...")
    other_df_fps = [get_morgan_fp(s) for s in tqdm(other_df['SMILES'])]

    our_data_smiles = our_data['SMILES'].tolist()
    other_df_smiles = other_df['SMILES'].tolist()

    # Filter out invalid SMILES
    our_data_valid_fps = []
    our_data_valid_smiles = []
    for fp, smiles in zip(our_data_fps, our_data_smiles):
        if fp is not None:
            our_data_valid_fps.append(fp)
            our_data_valid_smiles.append(smiles)

    other_df_valid_fps = []
    other_df_valid_smiles = []
    for fp, smiles in zip(other_df_fps, other_df_smiles):
        if fp is not None:
            other_df_valid_fps.append(fp)
            other_df_valid_smiles.append(smiles)
    
    results = []
    print(f"Calculating similarities against {dataset_name}...")
    for i, fp1 in enumerate(tqdm(our_data_valid_fps)):
        similarities = BulkTanimotoSimilarity(fp1, other_df_valid_fps)
        max_sim = 0
        max_idx = -1
        if similarities:
            max_sim = max(similarities)
            max_idx = similarities.index(max_sim)

        our_smiles = our_data_valid_smiles[i]
        if max_idx != -1:
            most_similar_smiles = other_df_valid_smiles[max_idx]
        else:
            most_similar_smiles = None
        
        results.append({
            'SMILES': our_smiles,
            'max_tanimoto': max_sim,
            'other_dataset': dataset_name,
            'most_similar_smiles': most_similar_smiles
        })

    results_df = pd.DataFrame(results)
    output_filename = f'data/FlemingDev/experiments/compare_datasets/results/pairwise_similarity_our_data_vs_{dataset_name}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

if __name__ == '__main__':
    main()


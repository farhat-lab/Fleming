import pandas as pd 
import numpy as np
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import LocalOutlierFactor

save_dir = 'data/chemical_space/'

##### Train Data Plot #####

## Compound Data
compound_df = pd.read_csv("data/inhibition_training_data/tb_inhibition_A&M/processed_data/Binarized_Inhibition_113851_mols.csv")
compound_df['group'] = "Compound"
compound_save_dir = save_dir+'train_compound_morgan.csv'

if os.path.exists(compound_save_dir):
    compound_df = pd.read_csv(compound_save_dir)
else:
    feature_list = []
    for smiles in tqdm(compound_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        feature_list.append(features.tolist())

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
    compound_df = pd.concat([compound_df,feature_df],axis=1)

    compound_df.to_csv(compound_save_dir, index=False)

## Fragment Data
fragment_df = pd.read_csv("data/inhibition_training_data/fragment/binarized_fragment_inhibition_999_mols.csv")
fragment_df['group'] = "Fragment"
fragment_save_dir = save_dir+'train_fragment_morgan.csv'

if os.path.exists(fragment_save_dir):
    fragment_df = pd.read_csv(fragment_save_dir)
else:
    feature_list = []
    for smiles in tqdm(fragment_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        feature_list.append(features.tolist())

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
    fragment_df = pd.concat([fragment_df,feature_df],axis=1)

    fragment_df.to_csv(fragment_save_dir, index=False)

## Antibiotics Data
current_antibiotics_df = pd.read_csv("data/inhibition_model/current_tb_antibiotic_SMILES.csv")
current_antibiotics_df['group'] = "Current_TB_Antibiotics"
current_antibiotics_df['inhibition'] = 1
antibiotics_save_dir = save_dir+'train_antibiotics_morgan.csv'

if os.path.exists(antibiotics_save_dir):
    current_antibiotics_df= pd.read_csv(antibiotics_save_dir)
else:
    feature_list = []
    for smiles in tqdm(current_antibiotics_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        feature_list.append(features.tolist())

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
    current_antibiotics_df = pd.concat([current_antibiotics_df,feature_df],axis=1)
    current_antibiotics_df.to_csv(antibiotics_save_dir, index=False)

training_df = pd.concat([compound_df,fragment_df,current_antibiotics_df])
training_df = training_df.reset_index(drop=True)


##### Screening Data ##### 
## Synthemol Generated Molecules
synthemol_df = pd.read_csv("data/output/generative/synthemol_generated/molecules.csv")
synthemol_df['group'] = "Synthemol Generated Molecules"
feature_list = []
for smiles in tqdm(synthemol_df['SMILES']):
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    feature_list.append(features.tolist())
feature_df = pd.DataFrame(feature_list)
feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
synthemol_df = pd.concat([synthemol_df,feature_df],axis=1)

dgm_df = pd.read_csv("data/output/generative/diffusion_generated/diffusion_generated_molecules_20241001.csv")
dgm_df["group"] = "Diffusion Generated Molecules"
feature_list = []
for smiles in tqdm(dgm_df['SMILES']):
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    feature_list.append(features.tolist())

feature_df = pd.DataFrame(feature_list)
feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
dgm_df = pd.concat([dgm_df,feature_df],axis=1)


##### Plot ##### 
# screening_df = screening_df[screening_df['inhibition']==1]
embed_df = pd.concat([training_df, synthemol_df, dgm_df])
plot_df = pd.concat([training_df, synthemol_df, dgm_df])
plot_df = plot_df.reset_index(drop=True)

reducer = umap.UMAP(n_neighbors=30,
                    min_dist=0.0,
                    n_components=2,
                    metric="jaccard",
                    random_state=42)

feature_names = list(np.arange(0,2048))
feature_names = [str(item) for item in feature_names]
# print(feature_names)
embed_feature_data = embed_df.loc[:,feature_names].values
generated_feature_data = plot_df.loc[:,feature_names].values

# feature_data = StandardScaler().fit_transform(feature_data)
trans = reducer.fit(embed_feature_data)
embedding = trans.transform(generated_feature_data)

embedding_df = pd.DataFrame(embedding)
embedding_df['group'] = plot_df['group']

print(embedding_df)

plt.figure(dpi=1200)

plt.scatter(
    embedding_df[(embedding_df['group'] != 'Synthemol Generated Molecules') & (embedding_df['group'] != 'Diffusion Generated Molecules')][0],
    embedding_df[(embedding_df['group'] != 'Synthemol Generated Molecules') & (embedding_df['group'] != 'Diffusion Generated Molecules')][1],
    s = 0.1,
    label = 'Training Data'
)

plt.scatter(
    embedding_df[embedding_df['group'] == 'Synthemol Generated Molecules'][0],
    embedding_df[embedding_df['group'] == 'Synthemol Generated Molecules'][1],
    s = 0.1,
    label = 'Synthemol Generated Molecules'
)

plt.scatter(
    embedding_df[embedding_df['group'] == 'Diffusion Generated Molecules'][0],
    embedding_df[embedding_df['group'] == 'Diffusion Generated Molecules'][1],
    s = 0.1,
    label = 'Diffusion Generated Molecules'
)

lgnd = plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
# plt.xlim(-5,12.5)
# plt.ylim(0,20)
plt.axis('off')
plt.savefig("evaluation/molecule_generation/output/morgan_data_distribution.png",
            bbox_inches="tight")
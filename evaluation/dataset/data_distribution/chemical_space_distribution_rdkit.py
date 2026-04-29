import pandas as pd 
import numpy as np
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
import os


save_dir = 'evaluation/dataset/data_distribution/output/'

##### Train Data Plot #####

## Fragment Data
fragment_df = pd.read_csv("data/inhibition_training_data/fragment/binarized_fragment_inhibition_999_mols.csv")
fragment_df['group'] = "Fragment"
fragment_save_dir = save_dir+'train_fragment_rdkit.csv'

if os.path.exists(fragment_save_dir):
    fragment_df = pd.read_csv(fragment_save_dir)
else:
    feature_list = []
    for smiles in tqdm(fragment_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,200))]
    fragment_df = pd.concat([fragment_df,feature_df],axis=1)
    fragment_df.to_csv(fragment_save_dir, index=False)

## Compound Data
compound_df = pd.read_csv("data/inhibition_training_data/tb_inhibition_A&M/processed_data/Binarized_Inhibition_113851_mols.csv")
compound_df['group'] = "Compound"
compound_save_dir = save_dir+'train_compound_rdkit.csv'

if os.path.exists(compound_save_dir):
    compound_df = pd.read_csv(compound_save_dir)
else:
    feature_list = []
    for smiles in tqdm(compound_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,200))]
    compound_df = pd.concat([compound_df,feature_df],axis=1)

    compound_df.to_csv(compound_save_dir, index=False)

## Antibiotics Data
current_antibiotics_df = pd.read_csv("data/inhibition_model/current_tb_antibiotic_SMILES.csv")
current_antibiotics_df['group'] = "Current_TB_Antibiotics"
current_antibiotics_df['inhibition'] = 1
antibiotics_save_dir = save_dir+'train_antibiotics_rdkit.csv'

if os.path.exists(antibiotics_save_dir):
    current_antibiotics_df= pd.read_csv(antibiotics_save_dir)
else:
    feature_list = []
    for smiles in tqdm(current_antibiotics_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,200))]
    current_antibiotics_df = pd.concat([current_antibiotics_df,feature_df],axis=1)
    current_antibiotics_df.to_csv(antibiotics_save_dir, index=False)

# training_df = pd.concat([compound_df,fragment_df,current_antibiotics_df])
training_df = pd.concat([compound_df, fragment_df,current_antibiotics_df])
training_df = training_df.reset_index(drop=True)
training_df = training_df.fillna(0)

print(training_df)

##### Outsample Data Plot ##### 


##### Screening Data ##### 
## Dragonfold Lead-Like
dragonfold_leadlike_df = pd.read_csv("data/screen_dataset/processed/dragonfold/leadlike_49608_mols.csv")
dragonfold_leadlike_df['group'] = "dragonfold_leadlike"
dragonfold_leadlike_save_sir = save_dir + 'dragonfold_leadlike_morgan.csv'

if os.path.exists(dragonfold_leadlike_save_sir):
    dragonfold_leadlike_df= pd.read_csv(dragonfold_leadlike_save_sir)
else:
    feature_list = []
    for smiles in tqdm(dragonfold_leadlike_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,200))]
    dragonfold_leadlike_df = pd.concat([dragonfold_leadlike_df,feature_df],axis=1)
    dragonfold_leadlike_df.to_csv(dragonfold_leadlike_save_sir, index=False)

## Seed Molecule
# seed_df = pd.read_csv("data/screen_dataset/processed/dragonfold/leadlike_49608_mols.csv")

## Neighbor Molecules
screening_df = pd.concat([dragonfold_leadlike_df])
screening_df = screening_df.reset_index(drop=True)
##### Plot ##### 
plot_df = pd.concat([training_df, screening_df])
plot_df = plot_df.reset_index(drop=True)

reducer = umap.UMAP(n_neighbors=25,
                    min_dist=0,
                    n_components=2)
feature_names = list(np.arange(0,200))
feature_names = [str(item) for item in feature_names]

feature_data = plot_df.loc[:,feature_names].values

embedding = reducer.fit_transform(feature_data)

embedding_df = pd.DataFrame(embedding)

embedding_df['inhibition'] = plot_df['inhibition']
embedding_df['group'] = plot_df['group']

plt.figure()
plt.scatter(
    embedding_df[embedding_df['group'] == 'Compound'][0],
    embedding_df[embedding_df['group'] == 'Compound'][1],
    s = 0.1,
    label = 'Compound Data'
)
plt.scatter(
    embedding_df[embedding_df['group'] == 'Fragment'][0],
    embedding_df[embedding_df['group'] == 'Fragment'][1],
    s = 0.1,
    label = 'Fragment Data'
)
plt.scatter(
    embedding_df[embedding_df['group'] == 'Current_TB_Antibiotics'][0],
    embedding_df[embedding_df['group'] == 'Current_TB_Antibiotics'][1],
    s = 0.1,
    label = 'Current TB_Antibiotics Data'
)
plt.scatter(
    embedding_df[embedding_df['group'] == 'dragonfold_leadlike'][0],
    embedding_df[embedding_df['group'] == 'dragonfold_leadlike'][1],
    s = 0.1,
    label = 'Dragonfold_Leaklike'
)
plt.legend()
plt.savefig("evaluation/dataset/data_distribution/output/rdkit_data_distribution.png")
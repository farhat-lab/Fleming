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
import plotly.express as px

save_dir = 'data/molecular_optimization/'

##### Screening Data ##### 
## Dragonfold Lead-Like
dragonfold_leadlike_df = pd.read_csv("data/screen_dataset/processed/dragonfold/leadlike_49608_mols.csv")
dragonfold_leadlike_df['group'] = "dragonfold_leadlike"
dragonfold_leadlike_save_sir = 'data/chemical_space/' + 'dragonfold_leadlike_morgan.csv'

if os.path.exists(dragonfold_leadlike_save_sir):
    dragonfold_leadlike_df= pd.read_csv(dragonfold_leadlike_save_sir)
else:
    feature_list = []
    for smiles in tqdm(dragonfold_leadlike_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        feature_list.append(features.tolist())

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
    dragonfold_leadlike_df = pd.concat([dragonfold_leadlike_df,feature_df],axis=1)
    dragonfold_leadlike_df.to_csv(dragonfold_leadlike_save_sir, index=False)

## Top 100
top_100_df = pd.read_csv('data/output/dragonfold/leadlike/top_100_output.csv')
top_100_df['group'] = "Top 100 Predicted Molecules"

feature_list = []
for smiles in tqdm(top_100_df['SMILES']):
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    feature_list.append(features.tolist())

feature_df = pd.DataFrame(feature_list)
feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
top_100_df = pd.concat([top_100_df,feature_df],axis=1)

## Sample Data
molecular_optimization_sample_df = pd.read_csv("evaluation/molecular_optimization/output/sample_output.csv")

feature_list = []
for smiles in tqdm(molecular_optimization_sample_df['SMILES']):
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    feature_list.append(features.tolist())

feature_df = pd.DataFrame(feature_list)
feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
molecular_optimization_sample_df = pd.concat([molecular_optimization_sample_df,feature_df],axis=1)
# molecular_optimization_sample_df.to_csv(dragonfold_leadlike_save_sir, index=False)

screening_df = pd.concat([dragonfold_leadlike_df])
screening_df = screening_df.reset_index(drop=True)

##### Plot ##### 
# plot_df = pd.concat([training_df, screening_df])
screening_df['group'] = 'Screening Data'
embed_df = pd.concat([screening_df])
plot_df = pd.concat([screening_df, molecular_optimization_sample_df])
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
embedding_df = embedding_df[embedding_df['group'].isin(['Screening Data', 'Seed Molecule', 'Selected Molecule - Step 1', 'Final Optimized Molecule'])]
embedding_df['size'] = np.where(embedding_df['group'].isin(['Seed Molecule', 'Selected Molecule - Step 1', 'Final Optimized Molecule']),5,1)
print(embedding_df)

fig = px.scatter(embedding_df, 
                 x=0, 
                 y=1, 
                 color="group",
                 opacity=0.5,
                 size='size'
                 )

fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 30),
                              x=1.1,
                              xanchor="center",
                              y=0.5,
                              yanchor="bottom",))

fig.update_layout(xaxis=dict(showgrid=False,
                             showticklabels=False,
                             visible=False),
                  yaxis=dict(showgrid=False,
                             showticklabels=False,
                             visible=False),
                  plot_bgcolor = 'rgba(0, 0, 0, 0)',
                  legend_title=None)

fig.write_html('evaluation/molecular_optimization/output/chemical_space.html')
import pandas as pd 
import numpy as np
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import plotly.express as px


save_dir = 'evaluation/dataset/data_distribution/output/'

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

print(training_df)


##### Screening Data ##### 
## Dragonfold Lead-Like
dragonfold_leadlike_df = pd.read_csv("data/screen_dataset/processed/dragonfold/leadlike_49608_mols.csv")
dragonfold_fraglike_df = pd.read_csv("data/screen_dataset/processed/dragonfold/fraglike_49761_mols.csv")
dragonfold_inhouse_df = pd.read_csv("data/screen_dataset/processed/dragonfold/fraglike_49761_mols.csv")

dragonfold_df = pd.concat([dragonfold_leadlike_df, dragonfold_fraglike_df, dragonfold_inhouse_df])
dragonfold_df = dragonfold_df.reset_index(drop=True)
dragonfold_df['group'] = "dragonfold"

dragonfold_leadlike_result_df = pd.read_csv("data/output/dragonfold/leadlike/final_output.csv")
dragonfold_fraglike_result_df = pd.read_csv("data/output/dragonfold/fraglike/final_output.csv")
dragonfold_inhouse_result_df = pd.read_csv("data/output/dragonfold/am_inhouse/final_output.csv")

dragonfold_result_df = pd.concat([dragonfold_leadlike_result_df, dragonfold_fraglike_result_df, dragonfold_inhouse_result_df])
dragonfold_df['inhibition'] = np.where(dragonfold_df['SMILES'].isin(dragonfold_result_df['SMILES']),1,0)

dragonfold_save_sir = save_dir + 'dragonfold_morgan_with_prediction.csv'

if os.path.exists(dragonfold_save_sir):
    dragonfold_df= pd.read_csv(dragonfold_save_sir)
else:
    feature_list = []
    for smiles in tqdm(dragonfold_df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        feature_list.append(features.tolist())

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [str(item) for item in list(np.arange(0,2048))]
    dragonfold_df = pd.concat([dragonfold_df,feature_df],axis=1)
    dragonfold_df.to_csv(dragonfold_save_sir, index=False)

screening_df = pd.concat([dragonfold_df])
screening_df = screening_df.reset_index(drop=True)

print(screening_df)


##### Plot ##### 
# screening_df = screening_df[screening_df['inhibition']==1]
training_df['group'] = np.where(training_df['inhibition'] == 1, "Training Set - M.tb Inhibitory", "Training Set - M.tb NonInhibitory")
screening_df['group'] = np.where(screening_df['inhibition'] == 1, "Screening Molecules - Predicted Inhibitory", "Screening Molecules - Predicted NonInhibitory")

embed_df = pd.concat([training_df, screening_df])
plot_df = pd.concat([training_df, screening_df])
plot_df['size'] = np.where(plot_df['inhibition'] == 1, 0.2, 0.1)
plot_df = plot_df.reset_index(drop=True)

reducer = umap.UMAP(n_neighbors=30,
                    min_dist=0.0,
                    n_components=2,
                    metric="jaccard",
                    random_state=42)

feature_names = list(np.arange(0,2048))
feature_names = [str(item) for item in feature_names]
# print(feature_names)
feature_data = plot_df.loc[:,feature_names].values

# feature_data = StandardScaler().fit_transform(feature_data)
embed_feature_data = embed_df.loc[:,feature_names].values
generated_feature_data = plot_df.loc[:,feature_names].values

trans = reducer.fit(embed_feature_data)
embedding = trans.transform(generated_feature_data)
# embedding = reducer.fit_transform(feature_data)

embedding_df = pd.DataFrame(embedding)
embedding_df['inhibition'] = plot_df['inhibition']
embedding_df['group'] = plot_df['group']
embedding_df['size'] = plot_df['size']

embedding_df = embedding_df.sort_values(by=['inhibition'])

print(embedding_df)

fig = px.scatter(embedding_df, 
                 x=0, 
                 y=1, 
                 color="group",
                 opacity=0.5,
                 size = 'size',
                 color_discrete_map={"Training Set - M.tb NonInhibitory":'rgba(188,205,235,0.8)',
                                     "Training Set - M.tb Inhibitory":'rgba(39,245,69,0.8)',
                                     "Screening Molecules - Predicted NonInhibitory":'rgba(245,181,39,0.8)',
                                     "Screening Molecules - Predicted Inhibitory":'rgba(255,4,4,0.75)'})

fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 30),
                              x=1.2,
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

fig.write_html("evaluation/dataset/data_distribution/output/plotly_distribution.html")
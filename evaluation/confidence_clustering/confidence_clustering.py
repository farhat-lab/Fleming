import pandas as pd 
import numpy as np
import umap
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import tukey_hsd

def confidence_clustering(prediction_dir,
                          feature_dir,
                          molecular_features,
                          output_dir,
                          uncertainty_cutoff,
                          ):
    
    predictions = pd.read_csv(prediction_dir)

    ## Generate Features
    features = []
    try:
        feature_df = pd.read_csv(feature_dir)
    except:
        if molecular_features == 'rdkit':
            generator = rdNormalizedDescriptors.RDKit2DNormalized()
            for smiles in tqdm(predictions['SMILES']):
                features.append(generator.process(smiles)[1:])
    
            feature_df = pd.DataFrame(features)
            feature_df['SMILES'] = predictions['SMILES']
            feature_df = feature_df.fillna(0)
            feature_df.to_csv(feature_dir,
                            index = False)

    ## Generate UMAP Embeddings
    reducer = umap.UMAP(n_neighbors = 30,
                        min_dist = 0.0,
                        n_components = 2)
    feature_data = feature_df.loc[:,feature_df.columns!='SMILES'].values

    if molecular_features != 'rdkit':
        feature_data = StandardScaler().fit_transform(feature_data)
    embedding = reducer.fit_transform(feature_data)

    np.random.seed(15)
    clusters = HDBSCAN(min_cluster_size = 100,).fit_predict(embedding)

    embedding_df = pd.DataFrame(embedding)
    embedding_df['uncertainty'] = predictions['inhibition_dirichlet_uncal_uncertainty']
    embedding_df['cluster'] = clusters

    ## Plot Chemical Space Clustering using HDBSCAN
    cluster_sample_list = []
    plt.figure()
    for i in range(max(embedding_df['cluster']) + 1):

        plt.scatter(
            embedding_df[embedding_df['cluster'] == i][0],
            embedding_df[embedding_df['cluster'] == i][1],
            # c = embedding_df[embedding_df['cluster'] == i]['cluster'],
            s = 0.5,
            label = f'Cluster_{i}',
        )

        cluster_sample_list.append(embedding_df[embedding_df['cluster'] == i]['uncertainty'].to_list())

    plt.legend()
    plt.savefig(output_dir + 'HDBSCAN_Clustering_Plot.png')

    ## Plot Uncertainty against Chemical Space Clustering
    cluster_sample_df = pd.DataFrame(cluster_sample_list).transpose()
    cluster_sample_df.columns = np.arange(0,max(embedding_df['cluster']) + 1).tolist()

    plt.figure()
    sns.violinplot(data=cluster_sample_df)
    plt.savefig(output_dir + 'uncertainty_against_HDBSCAN_Clustering.png')

    ## Kukey_hsd Test
    group0 = embedding_df[embedding_df['cluster'] == 0]['uncertainty'].to_list()
    group1 = embedding_df[embedding_df['cluster'] == 1]['uncertainty'].to_list()
    group2 = embedding_df[embedding_df['cluster'] == 2]['uncertainty'].to_list()
    group3 = embedding_df[embedding_df['cluster'] == 3]['uncertainty'].to_list()
    group4 = embedding_df[embedding_df['cluster'] == 4]['uncertainty'].to_list()

    print(" >>>>> Tukey HSD Test on Model Uncertainty Clustering >>>>>")
    res = tukey_hsd(group0,
                    group1,
                    group2,
                    group3,
                    group4)
    print(res)              

    ## Plot Confidence Clustering
    plt.figure()
    cmap = sns.cubehelix_palette(as_cmap=True)
    plt.scatter(
        embedding_df.iloc[:, 0],
        embedding_df.iloc[:, 1],
        c = embedding_df['uncertainty'],
        cmap=cmap,
        s = 20,
        alpha = 0.5)
    plt.colorbar(label = 'Uncertainty')
    plt.title(f'UMAP projection of {molecular_features} Features against Uncertainty', 
              fontsize=18)    
    plt.savefig(output_dir + f'confidence_clustering_cut-off={uncertainty_cutoff}.png')

    ## Save Clustering Results
    embedding_df['SMILES'] = predictions['SMILES']
    embedding_df[['SMILES','uncertainty','cluster']].to_csv(output_dir + "Prediction_Clustering_Results.csv",
                        index = False)
    

confidence_clustering(prediction_dir = '/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/dragonfold_prediction_dir.csv',
                      feature_dir = "/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/dragonfold_RDKIT_features.csv",
                      molecular_features='rdkit',
                      output_dir="/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering",
                      uncertainty_cutoff=0.55)
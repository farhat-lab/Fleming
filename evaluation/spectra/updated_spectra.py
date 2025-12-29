import random
import networkx as nx
from networkx.algorithms.components import connected_components
from scipy import stats
import numpy as np
import pandas as pd

def is_integer(n):
    if isinstance(n, int):
        return True
    elif isinstance(n, float):
        return n.is_integer()
    else:
        return False
    
def is_clique(G):
    return G.size() == (G.order()*(G.order()-1))/2

def calculate_percentile_2d(array, target):
    total_count = 0
    less_than_or_equal_count = 0
    # Iterate over each row
    for row in array:
        total_count += len(row)
        less_than_or_equal_count += sum(x <= target for x in row)
    # Calculate percentile
    if total_count == 0:
        return None  # To handle the case when array is empty
    percentile = (less_than_or_equal_count / total_count) * 100
    return percentile
def run_independent_set(spectral_parameter, input_G,
                        seed = None,
                        debug=False,
                        distribution  = None,
                        binary = True,
                        numpy_array = False):
    total_deleted = 0
    independent_set = []
    if seed is not None:
        random.seed(seed)
    G = input_G.copy()
    if binary:
        #First check if any connected component of the graph is a clique, if so, add it as one unit to the independent set
        components = list(connected_components(G))
        deleted = 0
        for i, component in enumerate(components):
            subgraph = G.subgraph(component)
            if is_clique(subgraph):
                print(f"Component {i} is too densly connected, adding samples as a single unit to independent set and deleting them from the graph")
                independent_set.append(list(subgraph.nodes()))
                G.remove_nodes_from(subgraph.nodes())
            else:
                for node in list(subgraph.nodes()):
                    if subgraph.degree(node) == len(subgraph.nodes()) - 1:
                        deleted += 1
                        G.remove_node(node)
        print(f"Deleted {deleted} nodes from the graph since they were connected to all other nodes")
    iterations = 0
    if numpy_array:
        #This is for when you can't construct a graph and you are doing continuous so it's everything to everything
        #We just implicitly run it assuming the graph, assumes a 2D numpy array
        if binary:
            raise Exception("Can't use numpy array with binary edge weights")
        indices_to_scan = list(range(input_G.shape[0]))
        while len(indices_to_scan) > 0:
            print(len(indices_to_scan))
            chosen_node = random.sample(indices_to_scan, 1)[0]
            independent_set.append(chosen_node)
            indices_to_scan.remove(chosen_node)
            for index, edge_weight in enumerate(input_G[chosen_node]):
                if index != chosen_node:
                    if random.random() < spectral_parameter and (1-spectral_parameter)*100 < calculate_percentile_2d(input_G, edge_weight):
                        indices_to_scan.remove(index)
    else:
        while not nx.is_empty(G):
            chosen_node = random.sample(list(G.nodes()), 1)[0]
            independent_set.append(chosen_node)
            neighbors = G.neighbors(chosen_node)
            neighbors_to_delete = []
            for neighbor in neighbors:
                if not binary:
                    if spectral_parameter == 1.0:
                        neighbors_to_delete.append(neighbor)
                    else:
                        edge_weight = G[chosen_node][neighbor]['weight']
                        if distribution is None:
                            raise Exception("Distribution must be provided if binary is set to False, must precompute similarities")
                        if random.random() < spectral_parameter and (1-spectral_parameter)*100 < stats.percentileofscore(distribution, edge_weight):
                            neighbors_to_delete.append(neighbor)
                else:
                    if spectral_parameter == 1.0:
                        neighbors_to_delete.append(neighbor)
                    elif spectral_parameter != 0.0:
                        if random.random() < spectral_parameter:
                            neighbors_to_delete.append(neighbor)
            if debug:
                print(f"Iteration {iterations} Stats")
                print(f"Deleted {len(neighbors_to_delete)} nodes from {G.degree(chosen_node)} neighbors of node {chosen_node}")
                total_deleted += len(neighbors_to_delete)
            for neighbor in neighbors_to_delete:
                G.remove_node(neighbor)
            if chosen_node not in neighbors_to_delete:
                G.remove_node(chosen_node)
            iterations += 1
        for node in list(G.nodes()):
            #Append the nodes left to G
            independent_set.append(node)
        if debug:
            print(f"{len(input_G.nodes())} nodes in the original graph")
            print(f"Total deleted {total_deleted}")
            print(f"{len(independent_set)} nodes in the independent set")
    return independent_set

tb_inhibition_dataset = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')
pre_calculated_similarity = np.load('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933.npy')

print("Load Pre Calculate Similarity Files Successully")

smiles_list = list(tb_inhibition_dataset['SMILES'])
index_list = np.arange(0,len(tb_inhibition_dataset))

spectra_parameters = {'number_repeats': 3, 
                      'random_seed': [42, 44, 46],
                      'spectral_parameters': ["{:.2f}".format(i) for i in np.arange(0, 1.05, 0.05)],
                      'force_reconstruct': True,}

independent_set = run_independent_set(0.5,
                                      input_G=pre_calculated_similarity,
                                      binary=False,
                                      numpy_array=True)

print(independent_set)
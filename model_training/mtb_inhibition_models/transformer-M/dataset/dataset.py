from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Sampler
import random
import os
import sys
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from torch_geometric.data import Data
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from random import Random

from utils.smiles_utils import smiles2graph, mol2graph
import algos
from dataset.collator import collator_3d

class Mtb_Inhibition_Dataset(LightningDataModule):
    def __init__(self,
                 position_data_dir,
                 feature_dir = None,
                 ):
        
        super.__init__

        self.positional_data_dir = position_data_dir

        self.data_list = process_pos_data(position_data_dir, feature_dir=feature_dir)

        for i in tqdm(range(len(self.data_list))):
            self.data_list[i] = preprocess_item(self.data_list[i])
            self.data_list[i].idx = i
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # item.idx = idx
        return item

    def collate_fn(self, batch):       
        return collator_3d(batch)

def process_pos_data(position_data_dir, feature_dir):

    graph_pos_list = Chem.SDMolSupplier(position_data_dir,
                                        removeHs = False)
    inhibition_list = [mol.GetProp('_tb_inhibition') for mol in graph_pos_list]
    smiles_list = [mol.GetProp('_SMILES') for mol in graph_pos_list]

    if feature_dir is not None:
        feature_list = np.load(feature_dir, allow_pickle=True)
    else:
        feature_list = None

    print('Extracting 3D positions from SDF files for Training Data...')

    data_list = []

    with Pool(processes=120) as pool:
        iter = pool.imap(mol2graph, graph_pos_list)

        for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):

            data = Data()
            tb_inhibition = float(inhibition_list[i])
            smiles = smiles_list[i]

            if feature_list is not None:
                fingerprint = np.array(feature_list[i])
                if np.isnan(fingerprint).any():
                    print("Generated Features contains nan, replacing with 0")
                    fingerprint[np.isnan(fingerprint)] = 0
            else:
                fingerprint = None

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([tb_inhibition])
            data.pos = torch.from_numpy(graph['position']).to(torch.float32)
            data.smiles = smiles
            data.fingerprint = fingerprint

            data_list.append(data)

    return data_list   

class TB_Inhibition_Dataset(LightningDataModule):
    def __init__(self, 
                 raw_data_dir = None,
                 processed_data_dir = None,
                 raw_data = True,
                 batch_size = 128,
                 three_D_Channel = False,
                 sdf_data_dir = None,
                 # scaffold_split = True,
                 # split_size = [0.8,0.2,0.0],
                 seed = 15,):
        
        super.__init__

        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

        self.batch_size = batch_size
        self.three_D_Channel = three_D_Channel
        self.sdf_data_dir = sdf_data_dir

        # self.embed_fail_index = embed_fail_index

        if raw_data == True:

            print(raw_data_dir)
            
            if raw_data_dir is None:
                print('raw_data_dir is needed for input raw_data')
                exit()

            self.data_list = self.process_raw()

            for i in range(len(self.data_list)):
                self.data_list[i] = preprocess_item(self.data_list[i])
                self.data_list[i].idx = i
            
            if self.three_D_Channel:
                processed_file_name = f"tb_processed_3D_data_{len(self.data_list)}_molecules.pt"
            else:
                processed_file_name = f"tb_processed_data_{len(self.data_list)}_molecules.pt"

            if self.processed_data_dir is not None:
                print(f">>>>> Saving to file >>>>>")
                torch.save(self.data_list, self.processed_data_dir + processed_file_name)
        
        else:
            
            if self.processed_data_dir is None:
                print('processed_data_dir is needed for input processed_data')
                exit()

            print(f">>>>>>>>>>Loading Molecular Data from {processed_data_dir}>>>>>>>>>>")
            self.data_list = torch.load(self.processed_data_dir)



    def process_raw(self):
        data_df = pd.read_csv(self.raw_data_dir)
        smiles_list = data_df['SMILES']
        inhibition_list = data_df['inhibition']

        if self.three_D_Channel:
            graph_pos_list = Chem.SDMolSupplier(self.sdf_data_dir)
            inhibition_list = [mol.GetProp('_tb_inhibition') for mol in graph_pos_list]
            smiles_list = [mol.GetProp('_SMILES') for mol in graph_pos_list]

            print('Extracting 3D positions from SDF files for Training Data...')
            train_data_with_position_list = []
            with Pool(processes=120) as pool:
                iter = pool.imap(mol2graph, graph_pos_list)

                for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):

                    data = Data()
                    tb_inhibition = float(inhibition_list[i])
                    smiles = smiles_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([tb_inhibition])
                    data.pos = torch.from_numpy(graph['position']).to(torch.float32)
                    data.smiles = smiles
                    data.fingerprint = None

                    train_data_with_position_list.append(data)

            data_list = train_data_with_position_list
            print(len(data_list))
        
        else:
            print('Converting SMILES strings into graphs...')
            data_list = []
            with Pool(processes=120) as pool:
                iter = pool.imap(smiles2graph, smiles_list)

                for i, graph in tqdm(enumerate(iter), total=len(inhibition_list)):
                    try:
                        data = Data()

                        tb_inhibition = inhibition_list[i]

                        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                        assert (len(graph['node_feat']) == graph['num_nodes'])

                        data.__num_nodes__ = int(graph['num_nodes'])
                        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                        data.y = torch.Tensor([tb_inhibition])
                        data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

                        data_list.append(data)

                    except:
                        continue

        return data_list

    def __getitem__(self, idx):
        item = self.data_list[idx]
        item.idx = idx
        return item

    def print_data_list(self):
        print(self.data_list)

    def collate_fn(self, batch):       
        return collator_3d(batch)	
        

def convert_to_single_emb(x, offset :int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x
    N = x.size(0)
    # print(x)
    x = convert_to_single_emb(x)
    # print(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item

def scaffold_balance(data,
                     sizes = [0.8,0.2,0.0],
                     seed = 15,
                     balanced = True):
    
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    smiles = [item.smiles for item in data]

    scaffolds = defaultdict(set)

    for i, smile in tqdm(enumerate(smiles), total = len(smiles)):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles = smile, includeChirality = False)
        scaffolds[scaffold].add(i)
    
    random = Random()

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffolds.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        # random.seed()
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    print(f">>>>> Completed Scaffold Balancing, Split into {len(train)} training samples, {len(val)} validation samples, and {len(test)} test samples >>>>>")

    return {'train': train, 
            'valid': val, 
            'test': test}


from torch.utils.data import Sampler
import torch
import numpy as np
import pandas as pd


class TB_Inhibition_Data_Sampler():
    def __init__(self,
                 labels,
                 idx,
                 subset_index,
                 class_balance = True,
                 task = 'classification',
                 bins = 100,):

        self.class_balance = class_balance
        self.task = task
        self.bins = bins

        self.labels = [labels]

        self.labels = [labels[i] for i in subset_index]
        self.idx = [idx[i] for i in subset_index]

        self.pos_mol = []
        self.neg_mol = []

        if task == 'classification':

            self.pos_mol = []
            self.neg_mol = []

            for i in range(len(self.labels)):
                if self.labels[i] == 1:
                    self.pos_mol.append(self.idx[i])
                else:
                    self.neg_mol.append(self.idx[i])
                    
        elif task == 'regression':
            self.data_df = pd.DataFrame({
                'index': self.idx,
                'labels': self.labels
            })

            self.data_df['bin'] = pd.cut(self.data_df['labels'], 
                                         bins = np.arange(0.0, 1.0 + 1/bins, 1/bins, dtype=float).tolist(), 
                                         labels = np.arange(0, bins, 1, dtype = int).tolist(),
                                         )

            self.data_category = self.data_df.groupby(['bin'], observed=False)['index'].apply(list).to_dict()

    def __len__(self):
        return len(self.pos_mol) + len(self.neg_mol)

    def __iter__(self):
        
        if self.class_balance:
            if self.task == 'classification':
                pos = True
                while True:
                    if pos:
                        yield np.random.choice(self.pos_mol)
                        pos = False
                    else:
                        yield np.random.choice(self.neg_mol)
                        pos = True
            elif self.task == 'regression':
                cat_idx = 0
                while True:
                    yield np.random.choice(self.data_category[cat_idx])
                    if cat_idx == self.bins - 1:
                        cat_idx = 0
                    else:
                        cat_idx += 1

        else:
            while True:
                yield np.random.choice(self.idx)

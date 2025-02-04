import pandas as pd
import os
import ast
from rdkit import Chem

class ADMET_AI:
    def __init__(self, dataset):
        if dataset not in os.listdir('ADMET_predictions'):
            raise ValueError(f'{dataset} not in ADMET_predictions')
        self.dataset = dataset
        self.mapping = self.process()
    
    def process(self):
        self.smiles_to_prediction = {}

        for i in range(5):
            for num, row in pd.read_csv(f'ADMET_predictions/{self.dataset}/fold_{i}/test_preds.csv').iterrows():
                smile = ast.literal_eval(row['smiles'])[0]
                mol = Chem.MolFromSmiles(smile)
                self.smiles_to_prediction[Chem.MolToSmiles(mol, canonical=True)] = row[self.dataset]

        return self.smiles_to_prediction
    
    def __getitem__(self, smiles):
        return self.smiles_to_prediction.get(smiles, "Prediction not found")

test = ADMET_AI('Carcinogens_Lagunin')
print(test['Cc1ccc(S(=O)(=O)Oc2ccc(/N=N/c3ccc(-c4ccc(/N=N/c5c(O)ccc6cc(S(=O)(=O)O)cc(S(=O)(=O)O)c56)c(C)c4)cc3C)cc2)cc1'])
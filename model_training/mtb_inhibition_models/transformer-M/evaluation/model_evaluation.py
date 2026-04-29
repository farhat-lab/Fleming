from run.TrainingModule import TrainingModule
from dataset.dataset import TB_Inhibition_Dataset, scaffold_balance as sb
import numpy as np
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
from utils.samplers import TB_Inhibition_Data_Sampler
import pickle

def model_evaluation(dataset_dir: str,
                     checkpoint_dir: str,
                     output_dir:str,):

    dataset = TB_Inhibition_Dataset(processed_data_dir= dataset_dir,
                                    batch_size=batch_size,
                                    raw_data=False,
                                    three_D_Channel=True,
                                    # scaffold_split=scaffold_balance,
                                    )
    
    model = TrainingModule.load_from_checkpoint(model_checkpoint)
    model.eval()

    with open(checkpoint_dir + 'split_dict.pkl', 'rb') as f:
        split_dict = pickle.load(f)

    batch_size = model.hparams.batch_size

    labels = [item['y'].item() for item in dataset.data_list]
    idx = list(np.arange(0,len(dataset.data_list),1))

    val_sampler = TB_Inhibition_Data_Sampler(labels,
                                             idx = idx,
                                             subset_index=split_dict['test'],
                                             class_balance=False,
                                             task = 'classification' if classification else 'regression')

    with torch.no_grad():

        smiles = []
        pred = []
        uncertainty = []
        for batch in tqdm(dataloader):

            forward_output = model(batch)
            batch_alphas = torch.reshape(
                            forward_output,
                            (forward_output.shape[0], forward_output.shape[1] // 2, 2),
                        )
            forward_output = batch_alphas[:, :, 1] / torch.sum(
                            batch_alphas, axis=2
                        )
                    
            S = torch.sum(batch_alphas, axis = 2)
            num_classes = batch_alphas.shape[2]
            test_uncertainty = num_classes / S

            smiles = smiles + batch['smiles']
            pred = pred + forward_output.cpu().flatten().tolist()
            uncertainty = uncertainty + test_uncertainty.cpu().flatten().tolist()
    
    prediction_df = pd.DataFrame()
    prediction_df['SMILES'] = smiles
    prediction_df['prediction'] = pred
    prediction_df['uncertainty'] = uncertainty

    prediction_df.to_csv(output_dir,
                         index = False)


model_evaluation("CHHH",
                 "data/epoch=9-step=7820.ckpt")
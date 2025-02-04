from pytorch_lightning import LightningModule, Trainer
from run.TrainingModule import TrainingModule
from tb.data.dataset import EGNN_TB_Inhibition_Dataset
from tb.utils.samplers import TB_Inhibition_Data_Sampler
from argparse import Namespace
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import BatchSampler

import torch
import numpy as np
import pandas as pd
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from matplotlib import pyplot as plt

import pickle

def run_training(device: str = 'cpu',
                 lr: float = 5e-4,
                 batch_size: int = 64,
                 model_args: Namespace = None, 
                 position_data_dir: str = None,
                 output_dir: str = None,
                 max_epochs: int = 2500,
                 record_run: bool = False,
                 # split_dict: dict = None,
                 data_loader_num_workers: int = 1,
                 warmup_steps: int = 4000,
                 hidden_nf: int = 32,
                 n_layers: int = 4,
                 feature_dir: str = None,
                 gradient_clip: float = 0.0,
                 classification: bool = True,
                 weight_decay: float = None,
                 task: str = 'classification',
                 dropout: float = 0.0,
                 model_name: str = 'EGNN',
                 scaffold_balance: bool = True,
                 validation_data_balancing: bool = False,
                 train_data_balancing: bool = True,
                 loss_function: str = None,
                 use_lr_scheduler: bool = False,
                 evidential_lambda: float = 0.2,
                 split_dict: dict = None,
                 ):
    
    if feature_dir is not None:
        feature_list = np.load(feature_dir, allow_pickle=True)
    else: 
        feature_list = None
    
    config = {
        "scaffold_balance": scaffold_balance,
    }


    dataset = EGNN_TB_Inhibition_Dataset(position_data_dir,
                                         feature_list=feature_list)

    print(len(dataset))

    if split_dict is None:
        if scaffold_balance:
            split_dict = dataset.scaffold_balance(data = dataset.smiles,
                                                sizes = [0.8,0.1,0.1],
                                                balanced = True)
        else:
            split_dict = {}
            # generator = torch.Generator().manual_seed(15)
            generator = torch.Generator()
            split_dict['train'], split_dict['valid'], split_dict['test'] = random_split(range(len(dataset)),
                                                                                        [0.8,0.1,0.1],
                                                                                        generator)
            split_dict['train'] = split_dict['train'].indices
            split_dict['valid'] = split_dict['valid'].indices
            split_dict['test'] = split_dict['test'].indices

    model = TrainingModule(lr = lr,
                           batch_size=batch_size,
                           device = device,
                           hidden_nf = hidden_nf,
                           n_layers = n_layers,
                           mol_feature_len = feature_list.shape[1] if (feature_list is not None) else 0,
                           gradient_clip = gradient_clip,
                           classification = classification,
                           weight_decay=weight_decay,
                           dropout=dropout,
                           model_name = model_name,
                           task = task,
                           loss_function=loss_function,
                           train_size = len(split_dict['train']),
                           use_lr_scheduler=use_lr_scheduler,
                           out_node_nf=2 if loss_function == 'dirichlet' else 1,
                           output_dir = output_dir,
                           lam = evidential_lambda,
                           )
    
    print(model)

    ### Define Logging Run Name
    run_name = '(R)' if task == 'regression' else '(C)'
    run_name = run_name + model_name + "_" + str(hidden_nf)

    if loss_function == 'dirichlet':
        run_name = run_name + f'+evidential_{evidential_lambda}' 

    if feature_dir is not None:
        if 'avalon' in feature_dir:
            run_name = run_name + '+avalon_512'
        elif 'morgan' in feature_dir:
            run_name = run_name + '+morgan'
        elif ('rdkit' in feature_dir) and ('rdkit_2d_normalized' in feature_dir):
            run_name = run_name + '+rdkit_2d_normalized'
        elif ('rdkit' in feature_dir) and ('rdkit_2d_normalized' not in feature_dir):
            run_name = run_name + '+rdkit'

    if scaffold_balance:
        run_name = run_name + '_scaffold_balanced'

    with open(output_dir + run_name + '_split_dict.pkl', 'wb') as f:
        pickle.dump(split_dict, f)

    if record_run:
        wandb_logger = WandbLogger(save_dir = output_dir,
                               project = "EGNN_TB_Inhibition_Prediction",
                               name = run_name,
                               config = config,
                               )        
    else:
        wandb_logger = None

    if task == 'regression':
        checkpoint_callback = ModelCheckpoint(dirpath = output_dir,
                                            save_top_k = 1,
                                            monitor = "validation_epoch_L1_Loss",
                                            mode = 'min',
                                            filename=run_name + '-{epoch}-{validation_epoch_L1_Loss:.2f}-{validation_epoch_Pearson_Correlation:.2f}')
    elif task == 'classification':
        checkpoint_callback = ModelCheckpoint(dirpath = output_dir,
                                            save_top_k = 1,
                                            monitor = "validation_epoch_auroc",
                                            mode = 'max',
                                            filename=run_name + '-{epoch}-{validation_epoch_auroc:.2f}-{validation_epoch_BCE_loss:.2f}')

    trainer = Trainer(devices = 1,
                      accelerator=device,
                      max_epochs=max_epochs,
                      logger = wandb_logger,
                      default_root_dir=output_dir,
                      callbacks=[checkpoint_callback],)
    
    train_idx = split_dict['train']
    val_idx = split_dict['valid']
    test_idx = split_dict['test']
    
    if classification:
        labels = dataset.binary_labels
    else:    
        labels = dataset.labels

    train_sampler = TB_Inhibition_Data_Sampler(labels,
                                               idx = dataset.idx,
                                               subset_index=train_idx,
                                               class_balance=train_data_balancing,
                                               task = task,
                                               bins = 5,)

    val_sampler = TB_Inhibition_Data_Sampler(labels,
                                             idx = dataset.idx,
                                               subset_index=val_idx,
                                               class_balance=validation_data_balancing,
                                               task = task,
                                               bins = 5,)

    test_sampler = TB_Inhibition_Data_Sampler(labels,
                                             idx = dataset.idx,
                                               subset_index=test_idx,
                                               class_balance=False,
                                               task = task,
                                               bins = 5,)
    
    train_dataloader = DataLoader(dataset=dataset,
                                  num_workers=data_loader_num_workers,
                                  collate_fn=dataset.collate_fn,
                                  batch_sampler=BatchSampler(train_sampler, 
                                                       batch_size=batch_size, 
                                                       drop_last=False),)

    val_dataloader = DataLoader(dataset=dataset,
                                  num_workers=data_loader_num_workers,
                                  collate_fn=dataset.collate_fn,
                                  batch_sampler=BatchSampler(val_sampler,
                                                             batch_size=batch_size,
                                                             drop_last=False),)
    
    test_dataloader = DataLoader(dataset=dataset,
                                  num_workers=data_loader_num_workers,
                                  collate_fn=dataset.collate_fn,
                                  batch_sampler=BatchSampler(test_sampler,
                                                             batch_size=batch_size,
                                                             drop_last=False),)
    
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    
    trainer.test(model,
                 dataloaders=test_dataloader,
                 ckpt_path='best')

    


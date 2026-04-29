from pytorch_lightning import LightningModule, Trainer
from model.transformer_m_test import TransformerM
from argparse import Namespace
from run.TrainingModule import TrainingModule
from dataset.dataset import TB_Inhibition_Dataset, scaffold_balance as sb
import torch
import wandb
from torch.utils.data import DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, BatchSampler
from utils.samplers import TB_Inhibition_Data_Sampler
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import pickle

def run_training(device:str = 'cpu',
                   lr:float = 1e-5,
                   batch_size: int = 64,
                   model_name: str = 'Transformer-M',
                   model_args: Namespace = None,
                   data_dir: str = 'data/tb_data/Batch1and2_data.csv',
                   processed_data_dir: str = None,
                   checkpoint_dir = 'data/',
                   loss_name: str = 'bce',
                   gpu: bool = False,
                   max_epochs: int = 2500,
                   record_run: bool = False,
                   split_dict: dict = None,
                   data_loader_num_workers: int=1,
                   warmup_steps: int=4000,
                   patience: int=10,
                   channel_weight: list=[0.997, 0.003],
                   scale_loss: bool = True,
                   noise_scale: float = 0.2,
                   weight_decay: float = 0.0,
                   finetune: bool = False,
                   log: bool = False,
                   use_lr_scheduler: bool = True,
                   scaffold_balance: bool = True,
                   classification: bool = True,):

    model = TrainingModule(
        lr = lr,
        loss_name = loss_name,
        model_name = model_name,
        model_args = model_args,
        warmup_steps=warmup_steps,
        d_model=model_args.encoder_embed_dim,
        batch_size = batch_size,
        accelerator = device,
        channel_weight = channel_weight,
        scale_loss=scale_loss,
        noise_scale=noise_scale,
        weight_decay=weight_decay,
        finetune=finetune,
        use_lr_scheduler=use_lr_scheduler,
        classification = classification,
        checkpoint_dir = checkpoint_dir,
        )

    dataset = TB_Inhibition_Dataset(processed_data_dir= processed_data_dir,
                                    batch_size=batch_size,
                                    raw_data=False,
                                    three_D_Channel=True,
                                    # scaffold_split=scaffold_balance,
                                    )

    if scaffold_balance:
        split_dict = sb(data = dataset.data_list,
                        sizes = [0.8,0.1,0.1],
                        seed = 15,
                        balanced = True)
    else:
        split_dict = {}
        generator = torch.Generator().manual_seed(15)
        split_dict['train'], split_dict['valid'] = random_split(range(len(dataset)),
                                                                [0.8,0.2],
                                                                generator)
        split_dict['train'] = split_dict['train'].indices
        split_dict['valid'] = split_dict['valid'].indices
    
    with open(checkpoint_dir + 'split_dict.pkl', 'wb') as f:
        pickle.dump(split_dict, f)

    # Loggers
    if log:
        logger = WandbLogger(save_dir=checkpoint_dir,
                                # project = 'ML_Drug_Discovery',
                                # checkpoint_name = 
                                )
    else:
        logger = None

    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath = checkpoint_dir,
                                            save_top_k = 1,
                                            monitor = "validation_epoch_auroc",
                                            mode = 'max',
                                            filename = '{epoch}-{validation_epoch_auroc:.2f}-{validation_epoch_BCE_loss:.2f}')

    callbacks = [checkpoint_callback]

    if patience > 0:
        early_stop_callback = EarlyStopping(monitor="validation auroc",
                                            min_delta=0.0,
                                            patience=patience,
                                            mode = 'max')
        callbacks.append(early_stop_callback)


    ## TODO: Add callbacks
    trainer = Trainer(
        devices=1,
        accelerator=device,
        ## num_nodes = num_gpu,
        callbacks = callbacks,
        max_epochs=max_epochs,
        num_sanity_val_steps=1,
        logger=logger,
        default_root_dir=checkpoint_dir,
        log_every_n_steps=1,
        )
    
    # if split_dict is not None:
    #     train_idx = split_dict['train']
    #     val_idx = split_dict['valid']
    # else:
    #     train_idx = dataset.split_dict['train']
    #     val_idx = dataset.split_dict['valid']

    labels = [item['y'].item() for item in dataset.data_list]
    idx = list(np.arange(0,len(dataset.data_list),1))

    train_sampler = TB_Inhibition_Data_Sampler(labels = labels,
                                               idx = idx,
                                               subset_index=split_dict['train'],
                                               class_balance= True,
                                               task = 'classification' if classification else 'regression',
                                               bins = 5,)
    
    val_sampler = TB_Inhibition_Data_Sampler(labels,
                                             idx = idx,
                                             subset_index=split_dict['valid'],
                                             class_balance=False,
                                             task = 'classification' if classification else 'regression')
    
    # test_sampler = TB_Inhibition_Data_Sampler(labels,
    #                                          idx = idx,
    #                                          subset_index=split_dict['test'],
    #                                          class_balance=False,
    #                                          task = 'classification' if classification else 'regression')

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
    
    # test_dataloader = DataLoader(dataset=dataset,
    #                               num_workers=data_loader_num_workers,
    #                               collate_fn=dataset.collate_fn,
    #                               batch_sampler=BatchSampler(test_sampler,
    #                                                          batch_size=batch_size,
    #                                                          drop_last=False),)

    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,)

args = Namespace(num_atoms = 512 * 9,
            num_in_degree = 512,
            num_out_degree = 512,
            num_edges = 1536,
            num_spatial = 512,
            num_edge_dis = 128,
            edge_type = 'multi_hop',
            multi_hop_max_dist = 5,
            encoder_layers = 12, # Default 12
            encoder_embed_dim = 768,
            encoder_ffn_embed_dim = 768,
            encoder_attention_heads = 32, # Default 32
            dropout = 0.0, # Default = 0.0
            attention_dropout = 0.1, # Default = 0.1
            act_dropout = 0.1, # Default = 0.1
            max_positions = 512,
            num_segment = 2,
            no_token_positional_embeddings = False,
            encoder_normalize_before = True,
            apply_init = True,
            activation_fn = 'gelu', 
            encoder_learned_pos = 'True',
            sandwich_ln = 'False',
            droppath_prob = 0.1, # Default = 0.1
            add_3d = True, # Default = True
            num_3d_bias_kernel = 128,
            no_2d = False, # Default = False
            mode_prob = '0.2,0.5,0.3',
            remove_head = True,
            num_classes = 1,
            clip_norm = 5,
            )

run_training(processed_data_dir='data/inhibition_training_data/tb_inhibition_A&M/final_data/tb_processed_3D_data_114887_molecules.pt',
             checkpoint_dir = 'data/model_checkpoints/Transformer-M/fold_1/',
             batch_size=64,
             model_args=args,
             device='gpu',
             data_loader_num_workers=0,
             split_dict=None,
             warmup_steps=15000,
             patience=0,
             channel_weight=[1,0.01], ## If [1,0], 3D channel closed
             noise_scale=0.2,
             weight_decay=0,
             finetune = False, 
             scale_loss = False,
             lr = 1e-5,
             log=True,
             use_lr_scheduler = True,
             classification = True,
             scaffold_balance=True,
             )
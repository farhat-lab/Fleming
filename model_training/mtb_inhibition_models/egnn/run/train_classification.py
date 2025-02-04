from run.run_training import run_training
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for spectra_split in ['5.0','7.0','10.0','15.0']:

    for fold in range(0,5):

        run_training(device=device,
                    lr = 1e-4,
                    batch_size = 64,
                    # position_data_dir = '/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/final_tb_data_114933_3d.sdf',
                    position_data_dir = f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/{spectra_split}_3d_data.sdf',
                    # output_dir = '/n/data1/hms/dbmi/farhat/alex/Deep Learning Pipeline for Mtb Antibiotics Discovery/final_checkpoints/Inhibition_EGNN/fold_4/',
                    output_dir = f'/n/data1/hms/dbmi/farhat/alex/Deep_Learning_Pipeline_for_Mtb_Antibiotics_Discovery/final_checkpoints/Inhibition_EGNN/spectra/spectra_{spectra_split}/fold_{fold}/',
                    max_epochs = 50,
                    record_run = True,
                    data_loader_num_workers = 1,
                    hidden_nf = 64, # 256 for EGNN
                    n_layers = 3, # 4 For EGNN
                    gradient_clip = 0.2,
                    # feature_dir = '/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/rdkit_2d_normalized_features_114887_molecules.npy',
                    feature_dir = f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/{spectra_split}_rdkit_normalized_features.npy',
                    classification = True,
                    weight_decay = None,
                    dropout = 0.0,
                    task = 'classification',
                    model_name='EGNN', # 'EGNN' for EGNN
                    scaffold_balance=False,
                    validation_data_balancing=False,
                    train_data_balancing=True,
                    loss_function='dirichlet',
                    use_lr_scheduler=True,
                    warmup_steps=4000,
                    evidential_lambda=0.2,
                    split_dict = None,
                    )
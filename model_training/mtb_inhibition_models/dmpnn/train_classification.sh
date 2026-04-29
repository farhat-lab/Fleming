#!/bin/bash
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python train.py --data_path 'data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv' \
                --dataset_type 'classification' \
                --save_dir 'data/model_checkpoints/Full_TB_Dataset_traditional_classification_scaffold' \
                --split_type 'scaffold_balanced' \
                --num_folds '1' \
                --metric 'auc' \
                --loss_function 'binary_cross_entropy' \
                --wandb_group_name 'Full_TB_Dataset_traditional_classification' \
                --class_balance \
                --epochs 100\
                --features_generator 'rdkit_2d_normalized'\
                --no_features_scaling \
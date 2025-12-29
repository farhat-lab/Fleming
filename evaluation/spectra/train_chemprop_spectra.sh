#!/bin/bash
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

# spectra_splits=(5.0 7.0 10.0 15.0)
## Split 1,2,3 < 1000 mols
spectra_splits=(4.0,6.0,8.0,9.0,11.0,12.0,13.0,14.0,16.0,17.0,18.0,19.0,20.0) # Run for the remaining splits

cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/inhibition_model/

for split in "${spectra_splits[@]}"
do 
    python train.py --data_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_splits/'$split'_data.csv' \
                    --dataset_type 'classification' \
                    --save_dir '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/checkpoints/chemprop_spectra_'$split \
                    --split_type 'random' \
                    --num_folds '5' \
                    --metric 'auc' \
                    --loss_function 'binary_cross_entropy' \
                    --wandb_group_name 'TB_Spectra' \
                    --class_balance \
                    --epochs 100\
                    --features_generator 'rdkit_2d_normalized'\
                    --no_features_scaling 
done



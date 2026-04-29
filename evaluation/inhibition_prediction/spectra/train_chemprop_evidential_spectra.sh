#!/bin/bash
#SBATCH -c 4
#SBATCH -t 5-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:teslaV100s:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

# spectra_splits=(5.0 7.0 10.0 15.0)
## Split 1,2,3 < 1000 mols
spectra_splits=(4.0 6.0 8.0 9.0 11.0 12.0 13.0 14.0 16.0 17.0 18.0 19.0 20.0) # Run for the remaining splits

cd data/inhibition_model/

for split in "${spectra_splits[@]}"
do 

    python train.py --data_path 'evaluation/inhibition_prediction/spectra/output/spectra_splits/'$split'_data.csv' \
                    --dataset_type 'classification' \
                    --save_dir 'evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_evidential_'$split \
                    --split_type 'random'\
                    --num_folds '5' \
                    --metric 'auc' \
                    --extra_metrics 'binary_cross_entropy' \
                    --loss_function 'dirichlet' \
                    --evidential_regularization 0.2\
                    --wandb_group_name 'TB_Spectra_Evidential'\
                    --class_balance \
                    --epochs 100 \
                    --features_generator 'rdkit_2d_normalized'\
                    --no_features_scaling \
                    --save_preds 
                    
done

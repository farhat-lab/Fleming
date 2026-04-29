#!/bin/bash
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

# spectra_splits=(5.0 7.0 10.0 15.0)
spectra_splits=(15.0)

cd data/inhibition_model/

for split in "${spectra_splits[@]}"
    do
        for fold in $(seq 0 4)
            do 
                python predict_inhibition.py --test_path 'evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_'$split'/fold_'$fold'/test_data.csv' \
                                            --checkpoint_dir 'evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_'$split'/fold_'$fold'/model_0' \
                                            --preds_path 'evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_'$split'/fold_'$fold'/test_preds.csv' \
                                            --features_generator 'rdkit_2d_normalized' \
                                            --no_features_scaling \
                                            --uncertainty_method 'dirichlet'
            done
    done
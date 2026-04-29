#!/bin/bash
#SBATCH -c 1
#SBATCH -t 1-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

cd data/inhibition_model/

for evidential in $(seq 4.0 1.0 20.0); do
    for fold in $(seq 0 5); do
        echo "Running evidential_$evidential, fold_$fold..."

        target_screening_dataset="evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_evidential_${evidential}/fold_${fold}/test_data.csv"
        inhibition_model_checkpoint="evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_evidential_${evidential}/fold_${fold}/"
        inhibition_screening_output_dir="evaluation/inhibition_prediction/spectra/output/checkpoints/chemprop_spectra_evidential_${evidential}/fold_${fold}/test_preds_evi.csv"

        python predict_inhibition.py \
            --test_path "$target_screening_dataset" \
            --checkpoint_dir "$inhibition_model_checkpoint" \
            --preds_path "$inhibition_screening_output_dir" \
            --features_generator 'rdkit_2d_normalized' \
            --no_features_scaling \
            --uncertainty_method 'dirichlet'
    done
done

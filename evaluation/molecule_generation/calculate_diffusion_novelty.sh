#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python data/novelty_check/calculate_novelty_parallel.py --reference_dir 'data/inhibition_training_data/Binary_Inhibition_114850_mols_inclusive.csv' \
                                                                                          --query_dir "data/output/diffusion_generated/ensemble_pass_both.csv" \
                                                                                          --similarity_output_dir 'data/output/diffusion_generated/output_train_molecule_similarity.csv'
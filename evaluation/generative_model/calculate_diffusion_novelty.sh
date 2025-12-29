#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python /home/aw274/Mtb_Inhibition_DL_Pipeline/novelty_check/calculate_novelty_parallel.py --reference_dir '/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/Binary_Inhibition_114850_mols_inclusive.csv' \
                                                                                          --query_dir "/home/aw274/Mtb_Inhibition_DL_Pipeline/output/diffusion_generated/ensemble_pass_both.csv" \
                                                                                          --similarity_output_dir '/home/aw274/Mtb_Inhibition_DL_Pipeline/output/diffusion_generated/output_train_molecule_similarity.csv'
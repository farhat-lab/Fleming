#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python data/novelty_check/calculate_novelty_parallel.py --reference_dir 'data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv' \
                                                                                          --query_dir 'data/output/generative/synthemol_generated/ensemble_pass_both.csv' \
                                                                                          --similarity_output_dir 'evaluation/molecule_generation/output/synthemol_hit_generated_novelty.csv'
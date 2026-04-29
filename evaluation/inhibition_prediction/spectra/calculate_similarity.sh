#!/bin/bash
#SBATCH -c 20
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=128G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


for i in `seq 0 5`
do
    python evaluation/inhibition_prediction/spectra/calculate_similarity.py --target_molecule_dir 'data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv' \
                                                                                            --output_dir 'evaluation/inhibition_prediction/spectra/output/pairwise_similarity_114933_'$i'.npy' \
                                                                                            --start_idx $(($i * 20000))\
                                                                                            --end_idx $(($i * 20000 + 20000))
done
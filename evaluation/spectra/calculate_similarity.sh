#!/bin/bash
#SBATCH -c 20
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=128G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


for i in `seq 0 5`
do
    python /home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/calculate_similarity.py --target_molecule_dir '/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv' \
                                                                                            --output_dir '/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_'$i'.npy' \
                                                                                            --start_idx $(($i * 20000))\
                                                                                            --end_idx $(($i * 20000 + 20000))
done
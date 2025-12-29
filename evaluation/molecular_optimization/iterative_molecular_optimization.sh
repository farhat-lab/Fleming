#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

# Define Directories
output_folder_dir="/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/molecular_optimization/"
seed_dir="/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/molecular_optimization/step-wise_data/selected_molecules_step2.csv"
# seed_dir="/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/molecular_optimization/step-wise_data/sample_seed.csv"

improving_siblings_dir=$output_folder_dir"improving_siblings.csv"
best_siblings_dir=$output_folder_dir"best_siblings.csv"

cp $seed_dir $improving_siblings_dir

# Define Variables
max_iter=1

for i in $(seq 1 $max_iter);
do
    echo "Running Iteration: $i"
    if [ $(cat improving_siblings.csv | wc -l) != 1 ]; then
        bash run_molecular_optimization.sh $output_folder_dir $improving_siblings_dir $best_siblings_dir $i
    fi
done

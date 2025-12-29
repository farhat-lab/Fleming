#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=200G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python /home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/combine_similarity_file.py
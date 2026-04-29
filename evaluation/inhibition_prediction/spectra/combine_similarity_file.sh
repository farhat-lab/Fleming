#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=200G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python evaluation/inhibition_prediction/spectra/combine_similarity_file.py
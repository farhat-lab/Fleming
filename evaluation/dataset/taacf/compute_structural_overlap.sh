#!/bin/bash
#SBATCH -c 2
#SBATCH -t 00-12:00
#SBATCH -p short
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

/home/aw274/.conda/envs/chemprop/bin/python /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/dataset/taacf/compute_structural_overlap.py
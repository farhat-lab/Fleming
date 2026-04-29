#!/bin/bash
#SBATCH -c 2
#SBATCH -t 00-12:00
#SBATCH -p short
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python evaluation/dataset/taacf/compute_insample_sim.py
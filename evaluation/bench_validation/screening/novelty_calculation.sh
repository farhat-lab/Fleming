#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-6:00
#SBATCH -p short
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

python evaluation/bench_validation/screening/output/novelty_calculation.py
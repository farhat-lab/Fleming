#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-6:00
#SBATCH -p short
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

/home/aw274/.conda/envs/chemprop/bin/python /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/inhibition_prediction/bench_validation/novelty_calculation.py
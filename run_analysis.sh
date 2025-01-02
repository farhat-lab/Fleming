#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 00-10:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=50G                          # Memory total in MB (for all cores)
#SBATCH -o logs/bioavailability.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e logs/bioavailability.err


python run_analysis.py configs/bioavailability.yaml
#python -m run_analysis hERG "block hERG" "block hERG"
#python -m run_analysis DILI "drug-induced liver injury" "cause drug-induced liver injury"
#python -m run_analysis Carcinogens_Lagunin "ability to cause cancer or be a carcinogen"

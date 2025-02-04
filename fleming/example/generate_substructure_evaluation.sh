#!/bin/bash
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p short
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

# /n/data1/hms/dbmi/farhat/alex/conda_envs/paperqa_env/bin/python /n/data1/hms/dbmi/farhat/alex/AgentLite/example/generate_substructure_evaluation.py

/n/data1/hms/dbmi/farhat/alex/conda_envs/pqapi_env/bin/python /n/data1/hms/dbmi/farhat/alex/AgentLite/example/generate_substructure_evaluation.py

#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

source activate /n/data1/hms/dbmi/farhat/alex/conda_envs/synthemol

synthemol \
    --model_path '/n/data1/hms/dbmi/farhat/alex/model_checkpoints/synthemol_dmpnn_checkpoint' \
    --model_type chemprop \
    --save_dir '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/synthemol_generated_molecules/' \
    --building_blocks_path '/n/data1/hms/dbmi/farhat/alex/model_checkpoints/synthemol_dmpnn_checkpoint/building_blocks.csv' \
    --building_blocks_score_column inhibition \
    --n_rollout 100
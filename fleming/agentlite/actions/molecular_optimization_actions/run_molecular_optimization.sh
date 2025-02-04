#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:teslaV100s:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

########## Activating Pipeline ##########
# Activate part of pipeline
run_generate_siblings=true
run_inhibition_prediction=true
run_inhibition_filtering=true
run_admet_prediction=true
run_admet_filtering=true
run_final_output=true
run_get_best_sibling=true

########## Declaring Variables ##########
# Input & Output Directory
output_folder_dir=$2

## Molecular Optimization
seed_dir=$1
sibling_df_save_dir=$output_folder_dir"sibling_molecules.csv"
sibling_dict_save_dir=$output_folder_dir"sibling_dict.csv"
molecular_optimization_output_dir=$output_folder_dir"best_siblings.csv"

## Target Screnning Dataset Path
# target_screening_dataset=$output_folder_dir"sibling_molecules.csv"

## Inhibition Model Variables
# Model Checkpoints
inhibition_model_checkpoint_1='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_evidential_classification_0.2_output' ## 2 Batches + Fragment
inhibition_model_checkpoint_2='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output' ## 3 Batches + Fragment + Antibiotics
# Output Directories
inhibition_screening_output_dir_1=$output_folder_dir"raw_inhibition_prediction_1.csv"
inhibition_screening_output_dir_2=$output_folder_dir"raw_inhibition_prediction_2.csv"
inhibition_filtering_output_dir_1=$output_folder_dir"filtered_inhibition_prediction_1.csv"
inhibition_filtering_output_dir_2=$output_folder_dir"filtered_inhibition_prediction_2.csv"
ensemble_pass_both_dir=$output_folder_dir"ensemble_pass_both.csv"
ensemble_pass_one_dir=$output_folder_dir"ensemble_pass_one.csv"
# Thresholds
inhibition_threshold=0.7
confidence_threshold=0.55

## ADMET Model Variables
admet_EGNN_model_checkpoints=""
admet_output_dir=$output_folder_dir"raw_admet_prediction.csv"
admet_filtering_output_dir=$output_folder_dir"filtered_admet_prediction.csv"
antibiotic_admet_prediction_dir='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv'
ranking_admet_out_dir=$output_folder_dir"ranking_admet_prediction.csv"

## Agent Model Variables
agent_log_dir=$output_folder_dir"agent_log/"
agent_output_dir=$output_folder_dir"agent_output.csv"
num_agent=1

## Final Output
final_output_dir=$output_folder_dir"final_output.csv"

########## Running Pipeline ###########
## Diffusion Model

## Molecular Optimization
cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/molecule_optimization
source activate /home/aw274/.conda/envs/synspace

if $run_generate_siblings; then
    python generate_siblings.py --seed_smiles_dir $seed_dir \
                                --sibling_df_save_dir $sibling_df_save_dir \
                                --sibling_dict_save_dir $sibling_dict_save_dir

    conda activate /home/aw274/.conda/envs/chemprop
    python sibling_QC.py --sibling_df_save_dir $sibling_df_save_dir 
fi

## M.tb Inhibition Classifier
cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/inhibition_model
conda activate /home/aw274/.conda/envs/chemprop
    
if $run_inhibition_prediction; then
    python predict_inhibition.py --test_path $sibling_df_save_dir \
                                --checkpoint_dir $inhibition_model_checkpoint_1 \
                                --preds_path $inhibition_screening_output_dir_1 \
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'
    python predict_inhibition.py --test_path $sibling_df_save_dir \
                                --checkpoint_dir $inhibition_model_checkpoint_2 \
                                --preds_path $inhibition_screening_output_dir_2 \
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'
fi

if $run_inhibition_filtering; then
    python filter_inhibition.py --raw_prediction_dir $inhibition_screening_output_dir_1 \
                                --filtered_prediction_dir $inhibition_filtering_output_dir_1 \
                                --inhibition_threshold $inhibition_threshold \
                                --confidence_threshold $confidence_threshold
    python filter_inhibition.py --raw_prediction_dir $inhibition_screening_output_dir_2 \
                                --filtered_prediction_dir $inhibition_filtering_output_dir_2 \
                                --inhibition_threshold $inhibition_threshold \
                                --confidence_threshold $confidence_threshold
    python ensemble_inhibition.py --model_1_result_dir $inhibition_filtering_output_dir_1 \
                              --model_2_result_dir $inhibition_filtering_output_dir_2 \
                              --pass_both_result_dir $ensemble_pass_both_dir \
                              --pass_one_result_dir $ensemble_pass_one_dir
fi

## Toxicity Classifier
cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model
conda activate /home/aw274/.conda/envs/admet_ai

if $run_admet_prediction; then
    python -m predict_admet --inhibition_out_dir $ensemble_pass_both_dir \
                            --admet_out_dir $admet_output_dir
fi

if $run_admet_filtering; then
    python -m filter_admet --raw_admet_out_dir $admet_output_dir \
                           --filtered_admet_out_dir $admet_filtering_output_dir \
                           --antibiotic_admet_prediction_dir $antibiotic_admet_prediction_dir \
                           --ranking_admet_out_dir $ranking_admet_out_dir
fi

# cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/agent
# conda activate /home/aw274/.conda/envs/langchain_env

# if $run_agent_ranking; then
#     python -m agent --input_smiles_dir $ranking_admet_out_dir\
#                     --agent_log_dir $agent_log_dir\
#                     --agent_output_dir $agent_output_dir\
#                     --num_agent $num_agent
# fi

cd /n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/molecular_optimization_actions
conda activate /home/aw274/.conda/envs/chemprop

if $run_final_output; then
    python -m final_output --inhibition_output_dir $ensemble_pass_both_dir \
                           --admet_output_dir $ranking_admet_out_dir \
                           --agent_output_dir '' \
                           --final_output_dir $final_output_dir 
fi

cd /n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/molecular_optimization_actions
conda activate /home/aw274/.conda/envs/synspace

if $run_get_best_sibling; then
    python -m filter_siblings --sibling_screening_output_dir $final_output_dir \
                              --seed_screening_output_dir $seed_dir\
                              --sibling_filter_output $molecular_optimization_output_dir \
                              --sibling_dict_dir $sibling_dict_save_dir
fi




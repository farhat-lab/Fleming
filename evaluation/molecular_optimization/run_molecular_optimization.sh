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
run_diffusion_generating=false
run_diffusion_filtering=false
run_inhibition_prediction=true
run_inhibition_filtering=true
run_admet_prediction=true
run_admet_filtering=true
run_agent_ranking=false
run_final_output=true
run_get_best_sibling=true

echo $1

########## Declaring Variables ##########
# Output Directory
output_folder_dir=$1

## Molecular Optimization
seed_dir=$2
sibling_df_save_dir=$output_folder_dir"sibling_molecules.csv"
sibling_dict_save_dir=$output_folder_dir"sibling_dict.csv"
molecular_optimization_output_dir=$3
iter=$4

## Target Screnning Dataset Path
target_screening_dataset=$output_folder_dir"sibling_molecules.csv"

## Inhibition Model Variables
# Model Checkpoints
inhibition_model_checkpoint_1='data/model_checkpoints/classifier_checkpoints/CompoundFragment_evidential_classification_0.2_output' ## 2 Batches + Fragment
inhibition_model_checkpoint_2='data/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output' ## 3 Batches + Fragment + Antibiotics
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
antibiotic_admet_prediction_dir='data/admet_model/current_antibiotic_admet_predictions.csv'
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
cd data/molecule_optimization
source activate synspace

if $run_generate_siblings; then
    python generate_siblings.py --seed_smiles_dir $seed_dir \
                                --sibling_df_save_dir $sibling_df_save_dir \
                                --sibling_dict_save_dir $sibling_dict_save_dir
fi

## M.tb Inhibition Classifier
cd data/inhibition_model
source activate chemprop
    
if $run_inhibition_prediction; then
    python predict_inhibition.py --test_path $target_screening_dataset \
                                --checkpoint_dir $inhibition_model_checkpoint_1 \
                                --preds_path $inhibition_screening_output_dir_1 \
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'
    python predict_inhibition.py --test_path $target_screening_dataset \
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
cd data/admet_model
source activate admet_ai

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

cd data/agent
source activate langchain_env

if $run_agent_ranking; then
    python -m agent --input_smiles_dir $ranking_admet_out_dir\
                    --agent_log_dir $agent_log_dir\
                    --agent_output_dir $agent_output_dir\
                    --num_agent $num_agent
fi

cd data/run
source activate chemprop

if $run_final_output; then
    python -m final_output --inhibition_output_dir $ensemble_pass_both_dir \
                           --admet_output_dir $ranking_admet_out_dir \
                           --agent_output_dir '' \
                           --final_output_dir $final_output_dir 
fi

cd data/molecule_optimization
source activate synspace

if $run_get_best_sibling; then
    python -m filter_siblings --sibling_screening_output_dir $final_output_dir \
                              --seed_screening_output_dir $seed_dir\
                              --sibling_filter_output $molecular_optimization_output_dir \
                              --sibling_dict_dir $sibling_dict_save_dir
fi




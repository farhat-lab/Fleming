#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:teslaV100s:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

## M.tb Inhibition Classifier
cd /home/aw274/Mtb_Inhibition_DL_Pipeline/inhibition_model
source activate chemprop

inhibition_model_checkpoint='/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output'
# inhibition_model_checkpoint="/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/dmpnn_evidential_classification_0.2_rdkit_output"
save_preds_path='/home/aw274/Mtb_Inhibition_DL_Pipeline/prediction_evaluation/inhibition_prediction/new_outsample_prediction.csv' 

python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/validation_dataset/outsample_data.csv' \
                                --checkpoint_dir $inhibition_model_checkpoint \
                                --preds_path  $save_preds_path\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'
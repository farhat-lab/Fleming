## M.tb Inhibition Classifier
cd /home/aw274/Mtb_Inhibition_DL_Pipeline/inhibition_model
source activate chemprop

# inhibition_model_checkpoint='/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output' # 3 Batches + Fragment + Antibiotics
# inhibition_model_checkpoint="/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/dmpnn_evidential_classification_0.2_rdkit_output" # 2 Batches
# inhibition_model_checkpoint="/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/dmpnn_CompoundFragment_evidential_classification_0.2_output" ## 2 Batches + Fragment
inhibition_model_checkpoint="/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_evidential_classification_0.2_output"
save_preds_path='/home/aw274/Mtb_Inhibition_DL_Pipeline/prediction_evaluation/inhibition_prediction/tb_antibiotics_prediction.csv' 

python predict_inhibition.py --test_path '/home/aw274/Mtb_Inhibition_DL_Pipeline/inhibition_model/current_tb_antibiotic_SMILES.csv' \
                                --checkpoint_dir $inhibition_model_checkpoint \
                                --preds_path  $save_preds_path\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

cd /home/aw274/Mtb_Inhibition_DL_Pipeline/prediction_evaluation/inhibition_prediction

python current_tb_antibiotic_evaluation.py --antibiotics_inhibition_prediction_dir $save_preds_path \
                                        --inhibition_threshold 0.7 \
                                        --confidence_threshold 0.7 


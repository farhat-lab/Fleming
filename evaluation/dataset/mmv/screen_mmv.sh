cd data/inhibition_model

# python predict_inhibition.py  --test_path "evaluation/dataset/mmv/output/mmv_smiles.csv" \
#                               --checkpoint_dir "data/final_checkpoints/Full_TB_Dataset_scaffold_dropout=0.2_cv=5/" \
#                               --preds_path "evaluation/dataset/mmv/output/mmv_inhibition_prediction.csv"\
#                               --features_generator 'rdkit_2d_normalized'\
#                               --no_features_scaling \
#                               --uncertainty_method 'dirichlet'

python predict_inhibition.py  --test_path "evaluation/dataset/mmv/output/mmv_smiles.csv" \
                              --checkpoint_dir "data/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output" \
                              --preds_path "evaluation/dataset/mmv/output/mmv_inhibition_prediction_old_models.csv"\
                              --features_generator 'rdkit_2d_normalized'\
                              --no_features_scaling \
                              --uncertainty_method 'dirichlet'
## M.tb Inhibition Classifier
cd data/inhibition_model
conda activate chemprop
    
python predict_inhibition.py --test_path 'evaluation/bench_validation/designed_mols/verified_six.csv' \
                             --checkpoint_dir 'data/FlemingDev/model_checkpoints/final_model/' \
                             --preds_path 'evaluation/bench_validation/designed_mols/inhibition_model_prediction_final_all_data_model.csv' \
                             --features_generator 'rdkit_2d_normalized' \
                             --no_features_scaling \
                             --uncertainty_method 'dirichlet'
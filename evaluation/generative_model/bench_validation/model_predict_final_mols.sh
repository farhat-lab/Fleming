## M.tb Inhibition Classifier
cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/inhibition_model
conda activate /home/aw274/.conda/envs/chemprop
    
python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/verified_six.csv' \
                             --checkpoint_dir '/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/final_model/' \
                             --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/inhibition_model_prediction_final_all_data_model.csv' \
                             --features_generator 'rdkit_2d_normalized' \
                             --no_features_scaling \
                             --uncertainty_method 'dirichlet'
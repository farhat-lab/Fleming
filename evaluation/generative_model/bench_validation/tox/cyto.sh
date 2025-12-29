
cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/inhibition_model

# python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/verified_six.csv' \
#                              --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
#                              --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_hepg2.csv'\
#                              --features_generator 'rdkit_2d_normalized' \
#                              --no_features_scaling \
#                              --uncertainty_method 'dirichlet'

# python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/verified_six.csv' \
#                                 --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
#                                 --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_hskmc.csv'\
#                                 --features_generator 'rdkit_2d_normalized' \
#                                 --no_features_scaling \
#                                 --uncertainty_method 'dirichlet'

# python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/verified_six.csv' \
#                                 --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
#                                 --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_imr90.csv'\
#                                 --features_generator 'rdkit_2d_normalized' \
#                                 --no_features_scaling \
#                                 --uncertainty_method 'dirichlet'

# For current antibiotics
python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv' \
                             --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
                             --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/current_antibiotic_hepg2.csv'\
                             --features_generator 'rdkit_2d_normalized' \
                             --no_features_scaling \
                             --uncertainty_method 'dirichlet' \
                             --smiles_column 'SMILES'

python predict_inhibition.py --test_path '//n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/current_antibiotic_hskmc.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet' \
                                --smiles_column 'SMILES'

python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/current_antibiotic_imr90.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet' \
                                --smiles_column 'SMILES'
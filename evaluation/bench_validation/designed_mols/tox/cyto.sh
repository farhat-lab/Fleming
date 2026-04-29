
cd data/inhibition_model

# python predict_inhibition.py --test_path 'evaluation/bench_validation/designed_mols/verified_six.csv' \
#                              --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
#                              --preds_path 'evaluation/bench_validation/designed_mols/tox/verified_six_hepg2.csv'\
#                              --features_generator 'rdkit_2d_normalized' \
#                              --no_features_scaling \
#                              --uncertainty_method 'dirichlet'

# python predict_inhibition.py --test_path 'evaluation/bench_validation/designed_mols/verified_six.csv' \
#                                 --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
#                                 --preds_path 'evaluation/bench_validation/designed_mols/tox/verified_six_hskmc.csv'\
#                                 --features_generator 'rdkit_2d_normalized' \
#                                 --no_features_scaling \
#                                 --uncertainty_method 'dirichlet'

# python predict_inhibition.py --test_path 'evaluation/bench_validation/designed_mols/verified_six.csv' \
#                                 --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
#                                 --preds_path 'evaluation/bench_validation/designed_mols/tox/verified_six_imr90.csv'\
#                                 --features_generator 'rdkit_2d_normalized' \
#                                 --no_features_scaling \
#                                 --uncertainty_method 'dirichlet'

# For current antibiotics
python predict_inhibition.py --test_path 'data/admet_model/current_antibiotic_admet_predictions.csv' \
                             --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
                             --preds_path 'evaluation/bench_validation/designed_mols/tox/current_antibiotic_hepg2.csv'\
                             --features_generator 'rdkit_2d_normalized' \
                             --no_features_scaling \
                             --uncertainty_method 'dirichlet' \
                             --smiles_column 'SMILES'

python predict_inhibition.py --test_path '/data/admet_model/current_antibiotic_admet_predictions.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
                                --preds_path 'evaluation/bench_validation/designed_mols/tox/current_antibiotic_hskmc.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet' \
                                --smiles_column 'SMILES'

python predict_inhibition.py --test_path 'data/admet_model/current_antibiotic_admet_predictions.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
                                --preds_path 'evaluation/bench_validation/designed_mols/tox/current_antibiotic_imr90.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet' \
                                --smiles_column 'SMILES'
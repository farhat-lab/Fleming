cd data/inhibition_model

python predict_inhibition.py --test_path 'evaluation/fleming/before_mo_smiles.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
                                --preds_path 'evaluation/fleming/hepg2_before.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

python predict_inhibition.py --test_path 'evaluation/fleming/after_mo_smiles.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
                                --preds_path 'evaluation/fleming/hepg2_after.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'


python predict_inhibition.py --test_path 'evaluation/fleming/before_mo_smiles.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
                                --preds_path 'evaluation/fleming/hskmc_before.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

python predict_inhibition.py --test_path 'evaluation/fleming/after_mo_smiles.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
                                --preds_path 'evaluation/fleming/hskmc_after.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'


python predict_inhibition.py --test_path 'evaluation/fleming/before_mo_smiles.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
                                --preds_path 'evaluation/fleming/imr90_before.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

python predict_inhibition.py --test_path 'evaluation/fleming/after_mo_smiles.csv' \
                                --checkpoint_dir "data/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
                                --preds_path 'evaluation/fleming/imr90_after.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'
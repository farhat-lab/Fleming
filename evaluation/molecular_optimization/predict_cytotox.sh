cd /home/aw274/Mtb_Inhibition_DL_Pipeline/inhibition_model

python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/before_mo_smiles.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/hepg2_before.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/after_mo_smiles.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/hepg2_after.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'


python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/before_mo_smiles.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/hskmc_before.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/after_mo_smiles.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/hskmc_after.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'


python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/before_mo_smiles.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/imr90_before.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'

python predict_inhibition.py --test_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/after_mo_smiles.csv' \
                                --checkpoint_dir "/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/" \
                                --preds_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/imr90_after.csv'\
                                --features_generator 'rdkit_2d_normalized' \
                                --no_features_scaling \
                                --uncertainty_method 'dirichlet'
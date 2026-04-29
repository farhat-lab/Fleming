cd data/admet_model

# admet_predict --data_path 'evaluation/bench_validation/designed_mols/verified_six.csv' \
#               --save_path 'evaluation/bench_validation/designed_mols/tox/verified_six_admet.csv' \
#               --smiles_column 'SMILES'

python -m filter_admet --raw_admet_out_dir 'evaluation/bench_validation/designed_mols/tox/verified_six_admet.csv' \
                           --filtered_admet_out_dir 'evaluation/bench_validation/designed_mols/tox/verified_six_admet_filtered.csv' \
                           --antibiotic_admet_prediction_dir 'data/admet_model/current_antibiotic_admet_predictions.csv' \
                           --ranking_admet_out_dir 'evaluation/bench_validation/designed_mols/tox/verified_six_admet_ranking.csv'
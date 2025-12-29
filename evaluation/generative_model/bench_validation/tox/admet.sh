cd /n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model

# admet_predict --data_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/verified_six.csv' \
#               --save_path '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_admet.csv' \
#               --smiles_column 'SMILES'

python -m filter_admet --raw_admet_out_dir '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_admet.csv' \
                           --filtered_admet_out_dir '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_admet_filtered.csv' \
                           --antibiotic_admet_prediction_dir '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv' \
                           --ranking_admet_out_dir '/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/tox/verified_six_admet_ranking.csv'
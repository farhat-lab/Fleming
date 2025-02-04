#!/bin/bash
cd /n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/pharmacokinetics_prediction_actions
source activate admet_ai

output_folder_dir=$2

admet_output_dir=$output_folder_dir"raw_admet_prediction.csv"
admet_filtering_output_dir=$output_folder_dir"filtered_admet_prediction.csv"
antibiotic_admet_prediction_dir='/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/pharmacokinetics_prediction_actions/current_antibiotic_admet_predictions.csv'
ranking_admet_out_dir=$output_folder_dir"ranking_admet_prediction.csv"

if $run_admet_prediction; then
    python -m predict_admet --inhibition_out_dir $1 \
                            --admet_out_dir $admet_output_dir
fi

if $run_admet_filtering; then
    python -m filter_admet --raw_admet_out_dir $admet_output_dir \
                           --filtered_admet_out_dir $admet_filtering_output_dir \
                           --antibiotic_admet_prediction_dir $antibiotic_admet_prediction_dir \
                           --ranking_admet_out_dir $ranking_admet_out_dir
fi


#!/bin/bash
cd /n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/inhibition_prediction_actions/
source activate chemprop

python predict.py --test_path $1 \
                  --checkpoint_dir "/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output" \
                  --preds_path $2 \
                  --features_generator 'rdkit_2d_normalized' \
                  --no_features_scaling \
                  --uncertainty_method 'dirichlet'
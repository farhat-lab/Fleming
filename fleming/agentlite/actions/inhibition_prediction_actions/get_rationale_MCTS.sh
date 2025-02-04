#!/bin/bash
cd /n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/inhibition_prediction_actions/
source activate chemprop

python get_rationale_MCTS.py --smiles_dir $1 
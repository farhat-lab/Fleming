#!/bin/bash

source /n/data1/hms/dbmi/farhat/alex/conda_envs/pqapi_env/bin/activate

conda run --prefix /n/data1/hms/dbmi/farhat/alex/conda_envs/pqapi_env/ python /n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/biochemist_actions/paperqa_actions.py --molecule_description "$1" \
                                                                                                                                                                              --paper_directory $2 \
                                                                                                                                                                              --save_dir $3

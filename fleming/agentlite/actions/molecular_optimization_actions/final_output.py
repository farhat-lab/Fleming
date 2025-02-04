import pandas as pd
import numpy as np
import argparse

rank_weights = {
    'rank_inhibition':1,
    'rank_admet':3,
    'rank_LLM':1
}

def combined_results_from_filters(inhibition_output_dir:str,
                                  admet_output_dir:str,
                                  agent_output_dir:str,
                                  final_output_dir:str,):

    ## Reading and merging dataframes
    inhibition_df = pd.read_csv(inhibition_output_dir)
    admet_df = pd.read_csv(admet_output_dir)
    
    combined_df = inhibition_df.merge(admet_df, on=['SMILES'], how = 'inner')

    if agent_output_dir != '':
        agent_df = pd.read_csv(agent_output_dir)
        combined_df = combined_df.merge(agent_df, on=['SMILES'], how='inner')
        output_columns = ['SMILES','inhibition','confidence','fail_absorption','fail_distribution',
                          'fail_metabolism','fail_excretion','fail_toxicity','fail_sum','absorption_admet_score',
                          'distribution_admet_score','metabolism_admet_score', 'excretion_admet_score', 'toxicity_admet_score',
                          'mean_rate', 'std']
    else:
        output_columns = ['SMILES','inhibition','confidence','fail_absorption','fail_distribution',
                          'fail_metabolism','fail_excretion','fail_toxicity','fail_sum','absorption_admet_score',
                          'distribution_admet_score','metabolism_admet_score', 'excretion_admet_score', 'toxicity_admet_score']
    
    combined_df = combined_df[output_columns]

    ## Ranking
    combined_df['rank_inhibition'] = combined_df['inhibition'].rank(ascending=False)
    # combined_df['rank_absorption'] = combined_df['fail_absorption'].rank(ascending=True)
    # combined_df['rank_distribution'] = combined_df['fail_distribution'].rank(ascending=True)
    # combined_df['rank_metabolism'] = combined_df['fail_metabolism'].rank(ascending=True)
    # combined_df['rank_excretion'] = combined_df['fail_excretion'].rank(ascending=True)
    # combined_df['rank_toxicity'] = combined_df['fail_toxicity'].rank(ascending=True)
    # combined_df['rank_admet_score'] = combined_df['admet_score'].rank(ascending=False)
    # combined_df['rank_absorption'] = combined_df['absorption_admet_score'].rank(ascending=False)
    # combined_df['rank_distribution'] = combined_df['distribution_admet_score'].rank(ascending=False)
    # combined_df['rank_metabolism'] = combined_df['metabolism_admet_score'].rank(ascending=False)
    # combined_df['rank_excretion'] = combined_df['excretion_admet_score'].rank(ascending=False)
    # combined_df['rank_toxicity'] = combined_df['toxicity_admet_score'].rank(ascending=False)
    combined_df['admet_score'] = combined_df['absorption_admet_score'] + combined_df['distribution_admet_score'] + combined_df['metabolism_admet_score'] \
                                + combined_df['excretion_admet_score'] + combined_df['toxicity_admet_score']
    combined_df['rank_admet'] = combined_df['admet_score'].rank(ascending=False)

    if agent_output_dir != '':
        combined_df['rank_LLM'] = combined_df['mean_rate'].rank(ascending=False)
        # ranks = ['rank_inhibition','rank_absorption','rank_distribution','rank_metabolism','rank_excretion',
        #          'rank_toxicity','rank_LLM']
        ranks = ['rank_inhibition','rank_admet','rank_LLM']
    else:
        # ranks = ['rank_inhibition','rank_absorption','rank_distribution','rank_metabolism','rank_excretion',
        #          'rank_toxicity']
        ranks = ['rank_inhibition','rank_admet']

    combined_df['overall_rank'] = 0
    for category in ranks:
        if category in rank_weights:
            combined_df['overall_rank'] += rank_weights[category] * combined_df[category]
    combined_df['overall_rank'] = combined_df['overall_rank'].rank()

    # combined_df = combined_df.sort_values(by=['overall_rank'],
    #                                       ascending=True,)

    # combined_df['overall_rank'] = combined_df[['rank_inhibition', 'rank_admet_score']].sum(axis=1)
    combined_df = combined_df.sort_values(by=['overall_rank'],
                                          ascending=True,)

    combined_df.to_csv(final_output_dir,
                       index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inhibition_output_dir', type=str,
                        help = "Results for Inhibition Prediction")
    
    parser.add_argument('--admet_output_dir', type=str,
                        help = "Results for ADMET Prediction")
    
    parser.add_argument('--agent_output_dir', type=str,
                        help = "Results for Agent Prediction")
    
    parser.add_argument('--final_output_dir', type=str,
                        help = "Final Output Save Directory")
    
    args = parser.parse_args()
    
    combined_results_from_filters(inhibition_output_dir=args.inhibition_output_dir,
                                  admet_output_dir=args.admet_output_dir,
                                  agent_output_dir=args.agent_output_dir,
                                  final_output_dir = args.final_output_dir)
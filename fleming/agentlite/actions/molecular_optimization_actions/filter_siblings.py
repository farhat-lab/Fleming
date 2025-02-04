import pandas as pd
import numpy as np
import argparse
import os.path

admet_weights = {
    'absorption':1,
    'distribution':1,
    'metabolism':1,
    'excretion':1,
    'toxicity':1
}

def filter_siblings(sibling_screening_output_dir:str,
                    seed_screening_output_dir:str,
                    sibling_dict_dir:str,
                    output_dir:str):
    
    print("Getting best siblings in this step...")

    sibling_screening_output = pd.read_csv(sibling_screening_output_dir)
    seed_screening_output = pd.read_csv(seed_screening_output_dir)
    sibling_dict = pd.read_csv(sibling_dict_dir)

    comparing_columns = ['SMILES', 'inhibition','fail_absorption','fail_distribution','fail_metabolism',
                         'fail_excretion','fail_toxicity','fail_sum','absorption_admet_score','distribution_admet_score',
                         'metabolism_admet_score','excretion_admet_score', 'toxicity_admet_score']

    best_siblings = pd.DataFrame(columns=comparing_columns)

    seed_screening_output = seed_screening_output[comparing_columns]
    sibling_screening_output = sibling_screening_output[comparing_columns]

    seed_screening_output['admet_score'] = 0
    sibling_screening_output['admet_score'] = 0
    for category in admet_weights:
        seed_screening_output['admet_score'] += seed_screening_output[f'{category}_admet_score'] * admet_weights[category]
        sibling_screening_output['admet_score'] += sibling_screening_output[f'{category}_admet_score'] * admet_weights[category]
        
    sibling_dict = sibling_dict[['SMILES','seed']]

    # improving_df = pd.DataFrame(columns=comparing_columns+['tolerance'])

    for seed in seed_screening_output['SMILES']:
        seed_output = seed_screening_output[seed_screening_output['SMILES']==seed]
        # seed_fail_sum = seed_output['fail_sum'].values[0]
        seed_admet_score = seed_output['admet_score'].values[0]

        siblings = sibling_dict[sibling_dict['seed']==seed]['SMILES']
        sibling_output = sibling_screening_output[sibling_screening_output['SMILES'].isin(siblings)]
        # sibling_output = sibling_output[sibling_output['fail_sum']<seed_fail_sum]
        sibling_output = sibling_output[sibling_output['admet_score'] >= seed_admet_score]
        if len(sibling_output) == 0:
            sibling_output = seed_output.copy()
            sibling_output['seed'] = seed
            # sibling_output['improved_ADMET_by'] = 0
            sibling_output['improved_admet_score_by'] = 0

        else:
            # sibling_output = sibling_output.sort_values(by=['fail_sum','inhibition'], ascending=[True,False])
            sibling_output = sibling_output.sort_values(by=['admet_score','inhibition'], ascending=[False,False])
            sibling_output = sibling_output.reset_index(drop=True)
            sibling_output = sibling_output.iloc[[0],:]
            sibling_output['seed'] = seed
            # sibling_output['improved_ADMET_by'] = seed_fail_sum - sibling_output['fail_sum'] 
            sibling_output['improved_admet_score_by'] = sibling_output['admet_score'] - seed_admet_score 

        best_siblings = pd.concat([best_siblings, sibling_output])  

    best_siblings = best_siblings.sort_values(by=['admet_score','inhibition'],
                                              ascending=False)

    best_siblings.to_csv(output_dir,
                         index = False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sibling_screening_output_dir', type=str,
                        help = "The directory of sibling screening results")

    parser.add_argument('--seed_screening_output_dir', type=str,
                        help = "The directory of seed screening results")

    parser.add_argument('--sibling_filter_output', type=str,
                        help = 'Output directory of filtering sibling')

    parser.add_argument('--sibling_dict_dir', type=str,
                        help = 'The directory of sibling dict')

    args = parser.parse_args()

    filter_siblings(sibling_screening_output_dir=args.sibling_screening_output_dir,
                    seed_screening_output_dir=args.seed_screening_output_dir,
                    output_dir=args.sibling_filter_output,
                    sibling_dict_dir=args.sibling_dict_dir,)
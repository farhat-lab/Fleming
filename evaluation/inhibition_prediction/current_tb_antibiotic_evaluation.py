import pandas as pd 
import numpy as np
import argparse

def current_antibiotic_evaluation(current_tb_antibiotic_inhibition_prediction_dir:str,
                                  inhibition_threshold:float,
                                  confidence_threshold:float):

    tb_antibiotic_prediction_df = pd.read_csv(current_tb_antibiotic_inhibition_prediction_dir)

    uncertainty_threshold = 1-confidence_threshold

    high_inhibition_high_confidence = tb_antibiotic_prediction_df[(tb_antibiotic_prediction_df['inhibition'] > inhibition_threshold) & (tb_antibiotic_prediction_df['inhibition_dirichlet_uncal_uncertainty'] < uncertainty_threshold)]
    high_inhibition_low_confidence = tb_antibiotic_prediction_df[(tb_antibiotic_prediction_df['inhibition'] > inhibition_threshold) & (tb_antibiotic_prediction_df['inhibition_dirichlet_uncal_uncertainty'] > uncertainty_threshold)]
    low_inhibition_high_confidence = tb_antibiotic_prediction_df[(tb_antibiotic_prediction_df['inhibition'] < inhibition_threshold) & (tb_antibiotic_prediction_df['inhibition_dirichlet_uncal_uncertainty'] < uncertainty_threshold)]
    low_inhibition_low_confidence = tb_antibiotic_prediction_df[(tb_antibiotic_prediction_df['inhibition'] < inhibition_threshold) & (tb_antibiotic_prediction_df['inhibition_dirichlet_uncal_uncertainty'] > uncertainty_threshold)]

    print(f"High Prediction High Confidence: {len(high_inhibition_high_confidence)}")
    print(f"High Prediction Low Confidence: {len(high_inhibition_low_confidence)}")
    print(f"Low Prediction High Confidence: {len(low_inhibition_high_confidence)}")
    print(f"Low Prediction Low Confidence: {len(low_inhibition_low_confidence)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--antibiotics_inhibition_prediction_dir', type=str,
                        help = "The directory of current in-use antibiotics inhibition prediction output")
    parser.add_argument('--inhibition_threshold', type=float,
                        help = 'Inhibition Threshold')
    parser.add_argument('--confidence_threshold', type=float,
                        help = 'Confidence Threshold')

    args = parser.parse_args()

    current_antibiotic_evaluation(current_tb_antibiotic_inhibition_prediction_dir=args.antibiotics_inhibition_prediction_dir,
                                  inhibition_threshold=args.inhibition_threshold,
                                  confidence_threshold=args.confidence_threshold)

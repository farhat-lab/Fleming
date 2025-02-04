import pandas as pd 
import numpy as np

toy_data = pd.read_csv('/n/scratch/users/a/aw274/data/tb_data/raw_data/Batch1and2_data.csv')

positive_case = toy_data[toy_data['inhibition'] == 0].iloc[0:1]
negative_case = toy_data[toy_data['inhibition'] == 1].iloc[0:1]

toy_data = pd.concat([positive_case,negative_case])
toy_data = toy_data.reset_index(drop=True)

toy_data.to_csv('/n/scratch/users/a/aw274/data/tb_data/raw_data/toy_data.csv',
                index=False)
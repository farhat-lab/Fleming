import pandas as pd

before = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/molecular_optimization_result_v2/top_generated_molecules.csv')
before[['SMILES']].to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/before_mo_smiles.csv', index=False)
after = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/molecular_optimization_result_v2/step_5/best_siblings.csv')
after[['SMILES']].to_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/fleming/after_mo_smiles.csv', index=False)
import pandas as pd 
import numpy

seed_data = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/top_100_output.csv')
seed_data['group'] = 'Seed Data'

step1_seed_data = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/molecular_optimization_result/sibling_dict_step1.csv')
step1_seed_data['group'] = 'Step 1 Generated Siblings'

step2_seed_data = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/molecular_optimization_result/sibling_dict_step2.csv')
step2_seed_data['group'] = 'Step 2 Generated Siblings'

step3_seed_data = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/molecular_optimization_result/sibling_dict_step3.csv')
step3_seed_data['group'] = 'Step 3 Generated Siblings'

step4_seed_data = pd.read_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/molecular_optimization_result/sibling_dict_step4.csv')
step4_seed_data['group'] = 'Step 4 Generated Siblings'

selected_molecule = 'CN(C)Cc1ccc(N(C)CCN(C)CC(=O)Nc2nc3ccc(OCCCN4CCCC4)cc3s2)nc1'
selected_molecule = pd.DataFrame([selected_molecule], columns = ['SMILES'])
selected_molecule['group'] = 'Final Optimized Molecule'
num_step = 4

try:
    step_4_seed = step4_seed_data[step4_seed_data['SMILES'] == selected_molecule['SMILES'].item()]
    step_4_seed['group'] = 'Selected Molecule - Step 3'
    step_4_siblings = step4_seed_data[step4_seed_data['seed'] == step_4_seed['seed'].item()]
except: 
    step_4_seed = selected_molecule
    step_4_seed['seed'] = step_4_seed['SMILES']
    num_step -= 1

try:
    print(step_4_seed['seed'].item())
    step_3_seed = step3_seed_data[step3_seed_data['SMILES'] == step_4_seed['seed'].item()]
    step_3_seed['group'] = 'Selected Molecule - Step 2'
    step_3_siblings = step3_seed_data[step3_seed_data['seed'] == step_3_seed['seed'].item()]
except: 
    step_3_seed = selected_molecule
    step_3_seed['seed'] = step_3_seed['SMILES']
    num_step -= 1

step_2_seed = step2_seed_data[step2_seed_data['SMILES'] == step_3_seed['seed'].item()]
print(step_2_seed['seed'].item())
step_2_seed['group'] = 'Selected Molecule - Step 1'
step_2_siblings = step2_seed_data[step2_seed_data['seed'] == step_2_seed['seed'].item()]

step_1_seed = step1_seed_data[step1_seed_data['SMILES'] == step_2_seed['seed'].item()]
print(step_1_seed['seed'].item())
step_1_seed['group'] = 'Seed Molecule'
step_1_siblings = step1_seed_data[step1_seed_data['seed'] == step_1_seed['seed'].item()]

if num_step == 4:
    total_df = pd.concat([step_1_seed[['seed','group']], step_1_siblings[['SMILES','group']],
                        step_2_seed[['seed','group']], step_2_siblings[['SMILES','group']],
                        step_3_seed[['seed','group']], step_3_siblings[['SMILES','group']],
                        step_4_seed[['seed','group']], step_4_siblings[['SMILES','group']],
                        selected_molecule])

elif num_step == 3:
    step_1_seed['SMILES'] = step_1_seed['seed']
    step_2_seed['SMILES'] = step_2_seed['seed']
    step_3_seed['SMILES'] = step_3_seed['seed']
    total_df = pd.concat([step_1_seed[['SMILES','group']], step_1_siblings[['SMILES','group']],
                        step_2_seed[['SMILES','group']], step_2_siblings[['SMILES','group']],
                        step_3_seed[['SMILES','group']], step_3_siblings[['SMILES','group']],
                        selected_molecule[['SMILES','group']]])

elif num_step == 2:
    total_df = pd.concat([step_1_seed[['seed','group']], step_1_siblings[['SMILES','group']],
                        step_2_seed[['seed','group']], step_2_siblings[['SMILES','group']],
                        selected_molecule])

total_df.to_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/molecular_optimization/sample_output.csv',
                index = False)



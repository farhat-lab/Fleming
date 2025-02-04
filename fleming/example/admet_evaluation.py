import pandas as pd 
import numpy as np
from langchain_openai import ChatOpenAI
from DladAgent import run_agent
from tqdm import tqdm

def test_molecules(test_dir:str,
                   test_num:int = 10,
                   admet_file:bool = True):

    test_df = pd.read_csv(test_dir)
    admet_df = pd.read_csv(admet_file)
    # test_smiles = list(test_df['SMILES'])[:test_num]
    smiles_list = list(test_df['SMILES'])

    test_smiles = []
    for i in range(int(test_num / 5)):
        
        start_idx = i * int(len(smiles_list) / int(test_num / 5))
        end_idx = start_idx + 2 # originally +5

        if end_idx > len(smiles_list):
            end_idx = len(smiles_list)
            start_idx = end_idx - 2 # originally -5
        
        print(start_idx, '-->', end_idx)
        
        test_smiles = test_smiles + smiles_list[start_idx:end_idx]

    file = open('/n/data1/hms/dbmi/farhat/alex/AgentLite/pqapi_toxicity_evaluation_output.txt','w+')

    rank = 1
    for smiles in tqdm(test_smiles):

        try:
            task = f'''You're an expert medicinal chemistry, you are accessing Toxicity of an input molecules. Please do a literature search based on only existing medicincal chemistry knowledge regarding Toxicity for the molecule {smiles}.'''
            response = run_agent(task = task)
        except:
            response = 'Fail to Generate Response'

        # molecule_row = admet_df[admet_df['SMILES'] == smiles]
        # binary_columns = [x for x in admet_df.columns if x.startswith('pass_')]
        
        # failed_properties = []
        # for column in binary_columns:
        #     if list(molecule_row[column])[0] == 0:
        #         failed_properties.append(column)

        file.write(f"Molecule-{rank}: {smiles} \n")
        file.write("Fleming Output: \n")
        file.write(response)
        file.write('\n')
        # file.write("ADMET Model Output: \n")
        # for item in failed_properties:
        #     file.write(item)
        #     file.write('    ')
        file.write('\n')
        file.write('\n')

        rank += 1

    file.close()

def compare_results(agent_output_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/admet_evaluation_output.txt',
                    save_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agent_admet_result_comparison_classification.txt'):

    file = open(agent_output_dir, 'r+')
    text = file.read()
    file.close()

    llm = ChatOpenAI(model_name="gpt-4o")

    prompt = '''You're an experience medicinal biochemist. Now you are working on finding an antibiotics for TB, and testing for molecule's ADMET properties. 
                I'll provide you a list of SMILES and it's corresponding ADMET analysis from another agent that could read from literatures. 
                Now I only want to focus on Toxicity. 
                Please divide the molecules into two classes: favorable or medium Toxicity and unfavorable Toxicity, regarding its potential to be an antibiotics. 
                Please feel free to skip the molecules that you don't have enough information about, but give the list of molecules you skip at the end. 
                Please think carefully, do not miss any molecule, do not output SMILES string, do not output the reasoning in the response. 
                ''' + text

    # prompt = '''You're an experience medicinal biochemist. Now you are working on finding an antibiotics for TB, and testing for molecule's ADMET properties. 
    #             I'll provide you a list of SMILES and it's corresponding ADMET analysis from another agent that could read from literatures. 
    #             Please rank the molecules based on the described ADMET properties, from most favorable to least favorable for antibiotic discovery. 
    #             Please feel free to skip the molecules that you don't have enough information about, but give the list of molecules you skip at the end. 
    #             Please think carefully, do not miss any molecule, do not output the reasoning in the response. Following the format:
    #             1. molecule x
    #             2. molecule y
    #             etc.
    #             ''' + text

    response = llm.invoke(prompt).content

    file = open(save_dir, 'w+')
    file.write(response)
    file.close()

def get_ADMET_ranking(test_dir:str,
                      admet_file:str,
                      test_num:int = 50):
    
    test_df = pd.read_csv(test_dir)
    admet_df = pd.read_csv(admet_file)
    smiles_list = list(test_df['SMILES'])

    test_smiles = []
    for i in range(int(test_num / 5)):
        
        start_idx = i * int(len(smiles_list) / int(test_num / 5))
        end_idx = start_idx + 5 # originally + 5

        if end_idx > len(smiles_list):
            end_idx = len(smiles_list)
            start_idx = end_idx - 5 # orinally -5
        
        test_smiles = test_smiles + smiles_list[start_idx:end_idx]

    admet_labels = admet_df[admet_df['SMILES'].isin(test_smiles)]
    test_smiles = pd.DataFrame(test_smiles,columns=['SMILES'])
    admet_labels = admet_labels[['SMILES', 'fail_absorption','fail_distribution','fail_metabolism','fail_excretion','fail_toxicity','fail_sum']]
    admet_labels = test_smiles.merge(admet_labels,
                                     how = 'left',
                                     on = 'SMILES')
    
    admet_labels.to_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/admet_top50_label.csv',
                        index = False)
    
def evaluate(comparison_file:str):

    comparison_df = pd.read_csv(comparison_file)

    for task in ['Absorption','Distribution','Metabolism','Excretion','Toxicity']:

        agent_fav_df = comparison_df[comparison_df[f'Agent {task}'] == 1]
        agent_unfav_df = comparison_df[comparison_df[f'Agent {task}'] == 0]

        fav_mean_failed = np.mean(agent_fav_df[f"Model {task} Failed Sum"])
        unfav_mean_failed = np.mean(agent_unfav_df[f"Model {task} Failed Sum"])
        
        print(f"Favorable {task} Mean Failed by ADMET-AI model: {fav_mean_failed}")
        print(f"Unfavorable {task} Mean Failed by ADMET-AI model: {unfav_mean_failed}")

        

if __name__ == '__main__':

    test_molecules(test_dir="/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/final_output.csv",
                   test_num=50,
                   admet_file="/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/ranking_admet_prediction.csv")

    # compare_results()

    # get_ADMET_ranking(test_dir="/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/final_output.csv",
    #                   admet_file="/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/ranking_admet_prediction.csv")

    # evaluate('/n/data1/hms/dbmi/farhat/alex/AgentLite/admet_top50.csv')


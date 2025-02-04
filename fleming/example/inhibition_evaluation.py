import pandas as pd 
import numpy as np
from langchain_openai import ChatOpenAI
from DladAgent import run_agent
# import pqapi
from random import sample

def test_molecules(test_dir:str,
                   test_num:int = 10):

    test_df = pd.read_csv(test_dir)
    pos_df = test_df[test_df['inhibition'] >= 0.7]
    pos_smiles = list(pos_df['SMILES'])
    neg_df = test_df[test_df['inhibition'] < 0.7]
    neg_smiles = list(neg_df['SMILES'])

    # test_smiles = sample(pos_smiles,int(test_num/2)) + sample(neg_smiles,int(test_num/2))
    test_smiles = sample(neg_smiles,int(test_num/2))

    file = open('/n/data1/hms/dbmi/farhat/alex/AgentLite/pqapi_inhibition_evaluation_output.txt','w+')

    rank = 1
    for smiles in test_smiles:
        
        task = f'''Please do a literature search only based on existing medicincal chemistry knowledge regarding its potential to be inhibitory to TB for the molecule {smiles}, 
                    tell me if you think there are potential, if so what substructure may lead to that. '''
        response = run_agent(task = task)
        # response = str(response.answer)

        # try:
        #     task = f'''You're an expert medicinal chemistry looking for antibiotics against TB. 
        #             You are accessing whether the molecule is worth bench testing for it's inhibition against TB. 
        #             Please do a literature search only based on existing medicincal chemistry knowledge regarding its potential to be inhibitory to TB for the molecule {smiles}, 
        #             tell me if you think there are potential, if so what substructure may lead to that. '''
        #     response = pqapi.agent_query(task)
        # except:
        #     response = 'Fail to Generate Response'

        file.write(f"Molecule-{rank}: {smiles} \n")
        file.write("Fleming Output: \n")
        file.write(response)
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

    response = llm.invoke(prompt).content

    file = open(save_dir, 'w+')
    file.write(response)
    file.close()

if __name__ == '__main__':

    test_molecules(test_dir="/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv",
                   test_num=2)

    # compare_results()

    # get_ADMET_ranking(test_dir="/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/final_output.csv",
    #                   admet_file="/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/ranking_admet_prediction.csv")

    # evaluate('/n/data1/hms/dbmi/farhat/alex/AgentLite/admet_top50.csv')


import pandas as pd 
import numpy as np
from DladAgent import run_agent
from tqdm import tqdm
from langchain_openai import ChatOpenAI

def test_molecules(test_dir:str,
                   pred_dir:str,):

    test_df = pd.read_csv(test_dir)
    pred_df = pd.read_csv(pred_dir)
    pred_df['binary_pred'] = np.where(pred_df['Y'] >= 0.5, 1, 0)
    pred_df['smiles'] = pred_df['smiles'].str[2:-2]

    combine_df = test_df.merge(pred_df,
                               left_on = 'SMILES',
                               right_on = 'smiles')

    agree_pos_df = combine_df[(combine_df['inhibition'] == combine_df['binary_pred']) & (combine_df['inhibition'] == 1)]
    agree_neg_df = combine_df[(combine_df['inhibition'] == combine_df['binary_pred']) & (combine_df['inhibition'] == 0)]
    disagree_df =  combine_df[(combine_df['inhibition'] != combine_df['binary_pred'])]

    test_smiles = list(agree_pos_df['SMILES'])[:5] + list(agree_neg_df['SMILES'])[:5] + list(disagree_df['SMILES'])[:5]

    file = open('/n/data1/hms/dbmi/farhat/alex/AgentLite/pqapi_bbb_evaluation_output.txt','w+')

    for smiles in tqdm(test_smiles):

        # print(smiles)

        # task = f'''You're an expert medicinal chemistry, you are accessing the ability of an input molecules to penetrate blood brain barrier. Please do a literature search based on only existing medicincal chemistry knowledge regarding its ability to penetrate blood brain barrier for the molecule O=C(Cc1ccc(Cl)c(Cl)c1)N1CCc2occc2[C@H]1CN1CC[C@@H](O)C1 
        #         tell me what substructures that lead you think it can or cannot penetrate the blood brain barrier, if not enough information, just say not enough information.'''
        # response = run_agent(task = task)

        try:
            task = f'''You're an expert medicinal chemistry, you are accessing the ability of an input molecules to penetrate blood brain barrier. Please do a literature search based on only existing medicincal chemistry knowledge regarding its ability to penetrate blood brain barrier for the molecule O=C(Cc1ccc(Cl)c(Cl)c1)N1CCc2occc2[C@H]1CN1CC[C@@H](O)C1 
                tell me what substructures that lead you think it can or cannot penetrate the blood brain barrier, if not enough information, just say not enough information.'''
            response = run_agent(task = task)
        except:
            response = 'Fail to Generate Response'

        file.write(f"{smiles} \n")
        # file.write("Fleming Output: \n")
        file.write(response)
        file.write('\n')
        file.write('\n')

    file.close()

def compare_results(agent_output_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/pqapi_bbb_evaluation_output.txt',
                    save_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/pqapi_bbb_result_comparison_classification.txt'):

    file = open(agent_output_dir, 'r+')
    text = file.read()
    file.close()

    llm = ChatOpenAI(model_name="gpt-4o")

    prompt = '''You're an experience medicinal biochemist. Now you are testing for molecule's ability to penetrate Blood Brain Barrier. 
                I'll provide you a list of SMILES and it's corresponding Blood Brain Barrier penetration analysis from another agent that could read from literatures. 
                Please label the molecules into three classes: penetratable, unpentratable and no enough information. 
                Please think carefully, output SMILES string, do not miss any molecule, do not output the reasoning in the response. 
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

if __name__ == '__main__':
    # test_molecules(test_dir='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/single_task_DMPNN/bbb/checkpoints/fold_0/test_data.csv',
    #                pred_dir='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/single_task_DMPNN/bbb/checkpoints/fold_0/test_preds.csv')
    compare_results()
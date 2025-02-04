import pandas as pd 
from DladAgent import run_agent
from tqdm import tqdm
from langchain_openai import ChatOpenAI

def test_molecules(test_dir:str):

    test_df = pd.read_csv(test_dir)
    test_smiles = list(test_df['SMILES'])

    file = open('/n/data1/hms/dbmi/farhat/alex/AgentLite/local_paperqa_bioavailability_evaluation_output.txt','w+')

    for smiles in tqdm(test_smiles):

        print(smiles)

        try:
            task = f'''You're an expert medicinal chemistry, you are accessing bioavailability of an input molecules. Please do a literature search based on only existing medicincal chemistry knowledge regarding its bioavailability for the molecule {smiles} 
                    tell me what substructures that lead you think it has good or bad bioavailability, if not enough information, just say not enough information.'''
            response = run_agent(task = task)
        except:
            response = 'Fail to Generate Response'

        file.write(f"{smiles} \n")
        file.write("Fleming Output: \n")
        file.write(response)
        file.write('\n')
        file.write('\n')

    file.close()

def compare_results(agent_output_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/local_paperqa_bioavailability_evaluation_output.txt',
                    save_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/local_paperqa_bioavailbility_result_comparison_classification.txt'):

    file = open(agent_output_dir, 'r+')
    text = file.read()
    file.close()

    llm = ChatOpenAI(model_name="gpt-4o")

    prompt = '''You're an experience medicinal biochemist. Now you are working on finding an antibiotics for TB, and testing for molecule's bioavailability. 
                I'll provide you a list of SMILES and it's corresponding bioavailability analysis from another agent that could read from literatures. 
                Please label the molecules into four classes: favorable bioavailability, medium bioavailability, unfavorable bioavailability, and no enough information, regarding its potential to be an antibiotics. 
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
    # test_molecules(test_dir='/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/single_task_DMPNN/bioavailability/checkpoints/fold_0/test_data.csv',)
    compare_results()
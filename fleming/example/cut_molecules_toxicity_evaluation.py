import pandas as pd 
import numpy as np
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from DladActions import DescribeMolecule
# import pqapi

def get_top_and_bottom_toxicity_molecules(test_dir):
    test_df = pd.read_csv(test_dir)
    test_df = test_df.sort_values(by=['fail_toxicity'], ascending = False)

    test_smiles = list(test_df['SMILES'])
    worst_toxicity_smiles = test_smiles[5:6]
    best_toxicity_smiles = test_smiles[-6:-5]

    return worst_toxicity_smiles, best_toxicity_smiles

def parse_description(description):

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    prompt = '''Please parse the description of the molecules to contain only important substructures for molecule property. 
                Separate each substructure with a comma. Molecule Description: ''' + description

    output = llm.invoke(prompt).content

    return output

def get_descriptions(worst_toxicity_smiles, best_toxicity_smiles):
    
    molecule_describer = DescribeMolecule()

    description_dict = []
    for molecule in worst_toxicity_smiles:
        description = molecule_describer(smiles=molecule)
        description_dict.append([molecule, 'bad_toxicity', description])
    for molecule in best_toxicity_smiles:
        description = molecule_describer(smiles=molecule)
        description_dict.append([molecule, 'good_toxicity', description])

    description_df = pd.DataFrame(description_dict, columns = ['SMILES','toxicity_label','description'])

    description_df['substructures'] = description_df['description'].apply(parse_description)

    return description_df

# def generate_paperqa_evaluation(substructure_list):

#     with open('/n/data1/hms/dbmi/farhat/alex/AgentLite/substructure_evaluation.csv', 'w+') as file:
#         for substructure in tqdm(substructure_list):
#             response = pqapi.agent_query(f"How does a {substructure} structure influence toxicity of a molecule as a drug?")
#             response = str(response.answer)
#             file.write(substructure)
#             file.write('\n')
#             file.write(response)
#             file.write('\n')
#             file.write('\n')
#             file.flush()

if __name__ == '__main__':
    worst_toxicity_smiles, best_toxicity_smiles = get_top_and_bottom_toxicity_molecules('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/ranking_admet_prediction.csv')
    description_df = get_descriptions(worst_toxicity_smiles, best_toxicity_smiles)
    print(description_df)
    # description_df.to_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/paperqa_toxicity_evaluation/best_worst_toxicity_descriptions.csv',
    #                       index = False)

    # substructure_list = ['1,3,4-thiadiazole core']
    # generate_paperqa_evaluation(substructure_list)
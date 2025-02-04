import pqapi
# from paperqa import Settings, ask
from tqdm import tqdm
import pandas as pd 
import re

def pqapi_generate_paperqa_evaluation(substructure_list):

    with open('/n/data1/hms/dbmi/farhat/alex/AgentLite/substructure_evaluation.txt', 'a+') as file:
        for substructure in tqdm(substructure_list):
            response = pqapi.agent_query(f"How does a {substructure} structure influence toxicity of a molecule as a drug?")
            response = str(response.answer)
            file.write(substructure)
            file.write('\n')
            file.write(response)
            file.write('\n')
            file.write('\n')
            file.flush()

def local_pqa_generate_paperqa_evaluation(substructure_list):

    with open('/n/data1/hms/dbmi/farhat/alex/AgentLite/local_pqa_substructure_evaluation.txt', 'w+') as file:
        for substructure in tqdm(substructure_list):
            prompt = f"How does a {substructure} structure influence toxicity of a molecule as a drug?"
            response = ask(
                prompt, 
                settings=Settings(temperature=0.5, paper_directory='/n/data1/hms/dbmi/farhat/alex/AgentLite/paper_repo/admet/'),
            )
            response = str(response.answer)
            file.write(substructure)
            file.write('\n')
            file.write(response)
            file.write('\n')
            file.write('\n')
            file.flush()

def structuralize_substructure_evaluation(txt_file = '/n/data1/hms/dbmi/farhat/alex/AgentLite/substructure_evaluation.txt'):

    with open(txt_file, 'r+') as file:
        evaluation_txt = file.read()

    evaluation_block_list = evaluation_txt.split('\n\n\n')

    substructure_info_list = []
    for evaluation_block in evaluation_block_list:

        parts = evaluation_block.split('\n')

        parts = [x for x in parts if x != '']

        if len(parts) > 2:
            substructure_name = re.sub('"', '', parts[0])
        else:
            print(parts)
            continue
        
        try:
            reference_start = parts.index('References')
            substructure_information = parts[2:reference_start]

            substructure_info_list.append({'substructure':substructure_name,
                                        'evaluation':' '.join(substructure_information)})
        except:
            substructure_info_list.append({'substructure':substructure_name,
                                        'evaluation':"No Information"})
        
    substructure_info_df = pd.DataFrame(substructure_info_list)

    substructure_info_df.to_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/paperqa_toxicity_evaluation/substructure_info_df.csv',
                                index = False)


if __name__ == '__main__':
    # worst_toxicity_smiles, best_toxicity_smiles = get_top_and_bottom_toxicity_molecules('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/ranking_admet_prediction.csv')
    # description_df = get_descriptions(worst_toxicity_smiles, best_toxicity_smiles)
    # description_df.to_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/best_worse_toxicity_descriptions.csv',
    #                       index = False)

    description_file = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/paperqa_toxicity_evaluation/[5,10]_and_[-10,-5]_toxicity_descriptions.csv')

    

    exit()

    # [:5] and [-5:]
    substructure_list = ['Tetracyclic azatetracyclic framework','Pyrrolidinyl side group','Pyridazine','Dimethylamino','2,3-Difluorophenyl', 'Pyridinyl',
                         'Benzamide core','Dimethylamino group','Thiazolo-benzothiazole moiety','Methyl substitution','1,3,4-thiadiazole core',
                         'chlorophenyl group','nitrophenyl group','amine nitrogen','Benzoxazole','bromophenyl group',
                         'Nitro group','[1,2,4]triazolo[4,3-a]pyrazine','propyl group','4-chlorophenylmethyl','ketone group',
                         '1,2,4-oxadiazole','para-methylphenyl group','morpholine group','propanone chain','Pyridopyridazine core',
                         'dihydro substitution','cyclopentyl group','ethylamino side group','Oxadiazole core','methylfuran',
                         'thioether linkage','acetamide group','cyclohexyl group','methyl group on nitrogen atom','tetrahydro-2H-pyran-4-ylsulfanyl']
    
    # [5:10] and [-10:-5]
    substructure_list = ['thiazole core','pyridine ring','amine linkage','fluorophenyl group','octahydrocyclopenta[c]pyrrole',
                         'chiral methylene bridge','quinoxaline','carboxamide','chlorophenyl','trifluoromethylphenyl',
                         'Quinoline core','oxadiazolyl substituent','acetamide linkage','isopropylphenyl group','phenyl substitution',
                         'pyridinyl substitution', 'Chloro substitution','Amino linkage','Hydroxyl group','Urea derivative',
                         'ethyl','isopropyl','oxazole ring','methoxyphenyl','Oxazolopyridine core', 
                         'dimethyl substitution','cyclohexylethyl','sulfonamide','sulfonyl',' N,N-diisopropyl',
                         'sulfanyl linkage','methoxy substitution','furan-2-ylmethyl group','oxyacetamide linkage','"Pyridopyrimidinone',
                         'methyl group','cyclopropyloxadiazole','thiomethyl linkage']

    # local_pqa_generate_paperqa_evaluation(substructure_list)
    # pqapi_generate_paperqa_evaluation(substructure_list)

    structuralize_substructure_evaluation()
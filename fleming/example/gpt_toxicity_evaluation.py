from DladActions import DescribeMolecule
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import pandas as pd

llm = ChatOpenAI(model_name="gpt-4o")
molecule_describer = DescribeMolecule()

test_df = pd.read_csv("/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/final_output.csv")
smiles_list = list(test_df['SMILES'])

description = molecule_describer(smiles='Cc1ccc(Nc2c(F)cccc2Cl)c(CC(=O)O)c1 ')
print(description)
exit()

test_num = 50 # 20
test_smiles = []
for i in range(int(test_num / 5)):
        
    start_idx = i * int(len(smiles_list) / int(test_num / 5))
    end_idx = start_idx + 2 # originally +5

    if end_idx > len(smiles_list):
        end_idx = len(smiles_list)
        start_idx = end_idx - 2 # originally -5
        
    print(start_idx, '-->', end_idx)
        
    test_smiles = test_smiles + smiles_list[start_idx:end_idx]

print(len(test_smiles))

file = open('/n/data1/hms/dbmi/farhat/alex/AgentLite/gpt_toxicity_evaluation_output.txt','w+')
for smiles in tqdm(test_smiles):

    description = molecule_describer(smiles=smiles)

    prompt = "Please evaluate the molecule with the following descriptions on its Toxicity. \n Molecule Description:"
    instruction = "\n Please respond in: [Favorable / Unfavorable Toxicity], Reasoning"
    prompt = prompt + description + instruction

    response = llm.invoke(prompt).content

    file.write(smiles)
    file.write('\n')
    file.write(response)
    file.write('\n')
    file.write('\n')

file.close()

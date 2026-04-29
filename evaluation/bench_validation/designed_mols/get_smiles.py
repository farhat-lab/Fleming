import pandas as pd 
import numpy as np
import rdkit.Chem
from rdkit.Chem import AllChem
from bioservices import KEGG
import pubchempy
import re
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

def get_number_after_word(text, word):
    # Regular expression to find the number after the word
    pattern = rf'{word}\s*(\d+)'  # Looks for the word followed by any number of spaces and then a number
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))  # Returns the number after the word
    else:
        return None  # Return None if no match is found


generated_mols = pd.read_excel('data/final_gen_molecule_selection.xlsx')
kegg = KEGG()

# ## Bioactive Compounds
# bioactive_compounds = kegg.get('br:br08001')
# pattern = r'C\d{5}'
# bioactive_compounds = re.findall(pattern, bioactive_compounds)

# fail_compound_id = []
# selected_compounds = []
# selected_smiles = []
# for compound in tqdm(bioactive_compounds):

#     success = False
#     attempt = 0

#     while not success:

#         compound_info = kegg.get(compound)

#         if isinstance(compound_info, str):
#             success = True
#             sid = get_number_after_word(compound_info,'PubChem:')
#             try:
#                 substance = pubchempy.Substance.from_sid(sid)
#                 smiles = substance.standardized_compound.isomeric_smiles
#                 selected_smiles.append(smiles)
#                 selected_compounds.append(compound)
#             except:
#                 fail_compound_id.append(compound)
            
#         else:
#             print("api error, try again")
#             print(compound_info)

#         if attempt >= 2:
#             fail_compound_id.append(compound)
#             success = True

#     if len(selected_compounds) % 10 == 0:
#         selected_compound_df = pd.DataFrame()
#         selected_compound_df['compound_id'] = selected_compounds
#         selected_compound_df['smiles'] = selected_smiles
#         selected_compound_df.to_csv('bioactive_mols.csv',
#                                     index = False)    

## Metabolites
url = 'https://www.genome.jp/kegg/tables/br08011.html'
response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

text = soup.get_text()
pattern = r'C\d{5}'
metabolites = re.findall(pattern, text)

fail_compound_id = []
selected_compounds = []
selected_smiles = []
for compound in tqdm(metabolites):

    success = False
    attempt = 0

    while not success:

        compound_info = kegg.get(compound)

        if isinstance(compound_info, str):
            success = True
            sid = get_number_after_word(compound_info,'PubChem:')
            try:
                substance = pubchempy.Substance.from_sid(sid)
                smiles = substance.standardized_compound.isomeric_smiles
                selected_smiles.append(smiles)
                selected_compounds.append(compound)
            except:
                fail_compound_id.append(compound)
            
        else:
            print("api error, try again")
            print(compound_info)
            attempt += 1

        if attempt >= 2:
            fail_compound_id.append(compound)
            success = True

    if len(selected_compounds) % 10 == 0:
        selected_compound_df = pd.DataFrame()
        selected_compound_df['compound_id'] = selected_compounds
        selected_compound_df['smiles'] = selected_smiles
        selected_compound_df.to_csv('evaluation/bench_validation/designed_mols/output/metabolites.csv',
                                    index = False)


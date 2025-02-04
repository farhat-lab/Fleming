import pandas as pd
import pubchempy as pcp
import re
import requests
from langchain_openai import ChatOpenAI
from agentlite.actions.biochemist_actions.molecular_description_image import generate_message

def get_safety_info(cid):

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    response = requests.get(url)

    smiles = None
    property_list = []
    if response.status_code == 200:
        data = response.json()
        for section in data['Record']['Section']:
            if section['TOCHeading'] == 'Safety and Hazards':
                safety_list = []
                safety_info = section['Section']
                for section in safety_info:
                    for subsection in section:
                        if subsection == 'Section':
                            for subsubsection in section[subsection]:
                                if subsubsection['TOCHeading'] == 'Hazard Classes and Categories':
                                    categories = subsubsection['Information']
                                    for category in categories:
                                        for prop in category['Value']['StringWithMarkup']:
                                            prop_string = prop['String']
                                            match = re.search(r"(.*?)(\b(?:Category |[0-9]+[A-Za-z]*\b).*)", prop_string, re.IGNORECASE)
                                            if match:
                                                property_part = match.group(1).strip()
                                                if property_part.endswith('-'):
                                                    property_part = property_part[:-1].strip()
                                                category_part = match.group(2).strip()
                                                if '1' in category_part:
                                                    safety_list.append(property_part)

                safety_information = f'The molecule have category 1 safety hazard for {safety_list}.'
                
                property_list.append(safety_information)
                    
            elif section['TOCHeading'] == 'Toxicity':
                toxicity_info = section['Section']
                for section in toxicity_info:
                    for subsection in section:
                        if subsection == 'Section':
                            for subsubsection in section[subsection]:
                                if subsubsection['TOCHeading'] == 'Toxicity Summary':
                                    categories = subsubsection['Information']
                                    for category in categories:
                                        for prop in category['Value']['StringWithMarkup']:
                                            prop_string = prop['String']
                                            property_list.append(prop_string)
                                        
            elif section['TOCHeading'] == 'Names and Identifiers':
                for section in section['Section']:
                    for subsection in section:
                        if subsection == 'Section':
                            for subsubsection in section[subsection]:
                                if subsubsection['TOCHeading'] == 'SMILES':
                                    categories = subsubsection['Information']
                                    for category in categories:
                                        for prop in category['Value']['StringWithMarkup']:
                                            smiles = prop['String']
                                            break

        return smiles, property_list
    
    else:
        return None, None
    

def describe_molecule(smiles):

        message = generate_message(smiles)
    
        llm = ChatOpenAI(model_name="gpt-4o")


        response = llm.invoke(message).content

        return response


def molecule_info_generation(cid):

    molecule_string = ""
    smiles, safety_data = get_safety_info(cid)
    if (smiles is None) | (safety_data is None) | (len(safety_data) == 0):
        return None
    else:
        description = describe_molecule(smiles)

        molecule_string = molecule_string + f"The following description belongs to the molecule {smiles}."

        molecule_string = molecule_string + description
        
        for property in safety_data:
            molecule_string = molecule_string + property

    return molecule_string

if __name__ == '__main__':

    # aid = 1188  # Replace this with your specific AID
    # url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/cids/JSON"
    # response = requests.get(url)

    # cids = response.json()['InformationList']['Information'][0]['CID']

    save_file = '/n/data1/hms/dbmi/farhat/alex/AgentLite/paper_repo/pubchem.txt'

    cids = [135398658,7628,1118,2244,33613]

    f = open(save_file, "w+")

    for cid in cids:
        molecule_information = molecule_info_generation(cid = cid)
        if molecule_information is not None:
            f.write(molecule_information)
            f.write('\n')

    f.close()

    
import os 
# import wikipedia
from agentlite.actions.BaseAction import BaseAction
from agentlite.actions.biochemist_actions.prompt_engineer_description import prompt_generator
from langchain_openai import ChatOpenAI
import subprocess
import pandas as pd
import random
import numpy as np
from tqdm import tqdm

class InhibitionPrediction(BaseAction):
    def __init__(self) -> None:
        action_name = "Mtb_Inhibition_Prediction"
        action_desc = "Using this action to predict the molecule's inhibition against M.tb."
        params_doc = {"smiles": "this is the parameter smiles string that passed into the action."}

        self.predict = get_inhibition_prediction()

        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc,
        )

    def __call__(self, **kwargs):
        params = {
            "smiles": kwargs["smiles"],
        }
        return self.predict.predict(params['smiles'])

class get_inhibition_prediction():
    def __init__(self) -> None:
        pass
    def predict(self, smiles):

        inhibition_prediction_script_path = "/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/inhibition_prediction_actions/predict_evidential_classification.sh"
        inhibition_prediction_input_path = "/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/ip_input_molecules.csv"
        inhibition_prediction_save_path = "/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/inhibition_prediction_output.csv"
    

        input_molecule_df = pd.DataFrame()
        input_molecule_df['SMILES'] = [smiles]
        input_molecule_df.to_csv(inhibition_prediction_input_path,
                                 index = False)
        try: 
            process = subprocess.Popen([inhibition_prediction_script_path, 
                            inhibition_prediction_input_path, 
                            inhibition_prediction_save_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
        except subprocess.CalledProcessError as e:
            output = e.output
            print(output)
        
        inhibition_prediction_output = pd.read_csv(inhibition_prediction_save_path)

        return (str(round(inhibition_prediction_output['inhibition'].values[0],3)), str(round(inhibition_prediction_output['inhibition_dirichlet_uncal_uncertainty'].values[0],3)))

class PharmacokineticsPrediction(BaseAction):
    def __init__(self) -> None:
        action_name = "Pharmacokinetics_Prediction"
        action_desc = "Using this action to predict the molecule's pharmacokinetics property."
        params_doc = {"smiles": "this is the parameter smiles string that passed into the action."}

        self.predict = get_pharmacokinetics_prediction()

        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc,
        )

    def __call__(self, **kwargs):
        params = {
            "smiles": kwargs["smiles"],
        }
        return self.predict.predict(params['smiles'])
    
class get_pharmacokinetics_prediction():
    def __init__(self) -> None:
        # self.property_list = property_list
        pass

    def predict(self, smiles):
        
        pharma_prediction_script_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/pharmacokinetics_prediction_actions/predict_pharmacokinetics.sh'
        pharma_prediction_input_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/pk_input_molecules.csv'
        pharma_prediction_save_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/'

        input_molecule_df = pd.DataFrame()
        input_molecule_df['SMILES'] = [smiles]
        input_molecule_df.to_csv(pharma_prediction_input_path,
                                 index = False)

        try: 
            process = subprocess.Popen([pharma_prediction_script_path, 
                                        pharma_prediction_input_path, 
                                        pharma_prediction_save_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
        except subprocess.CalledProcessError as e:
            output = e.output
            print(output)

        pk_output_df = pd.read_csv(pharma_prediction_save_path + 'ranking_admet_prediction.csv')

        all_property = ['AMES','BBB_Martins','Bioavailability_Ma','CYP1A2_Veith','CYP2C19_Veith',
                        'CYP2C9_Substrate_CarbonMangels','CYP2C9_Veith','CYP2D6_Substrate_CarbonMangels','CYP2D6_Veith','CYP3A4_Substrate_CarbonMangels',
                        'CYP3A4_Veith', 'Carcinogens_Lagunin' ,'ClinTox' ,'DILI','HIA_Hou',
                        'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                        'NR-ER','NR-PPAR-gamma','PAMPA_NCATS','Pgp_Broccatelli','SR-ARE',
                        'SR-ATAD5','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                        'hERG','Caco2_Wang','Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach',
                        'HydrationFreeEnergy_FreeSolv','LD50_Zhu','Lipophilicity_AstraZeneca','PPBR_AZ','Solubility_AqSolDB',
                        'VDss_Lombardo']
        
        all_property = ['pass_'+item for item in all_property]
        all_property = [item for item in all_property if item in pk_output_df.columns]

        pk_output_df = pk_output_df[['SMILES'] + all_property]
        pk_output_df = pk_output_df.rename(columns={"pass_LD50_Zhu": "pass_LD50"})

        output_dict = pk_output_df.to_dict(orient='records')

        return output_dict

class MolecularOptimization(BaseAction):
    def __init__(self) -> None:
        action_name = "Molecular_Optimization"
        action_desc = "Using this action to modify the molecular slightly to optimize the pharmacokinetic properties of the molecule"
        params_doc = {"smiles": "this is the smiles string that act as the starting seed of the molecular modification."}

        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc,
        )

    def molecular_optimization_step(self, smiles):
        mo_script_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/molecular_optimization_actions/run_molecular_optimization.sh'
        seed_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/mo_seed.csv'
        output_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/mo_output/'
        pharma_prediction_save_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/ranking_admet_prediction.csv'
        inhibition_prediction_save_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/inhibition_prediction_output.csv'

        try:
            inhibition_df = pd.read_csv(inhibition_prediction_save_path)
            pharmacokinetics_df = pd.read_csv(pharma_prediction_save_path)
        except:
            _ = get_inhibition_prediction().predict
            # print(_)
            _ = get_pharmacokinetics_prediction().predict(smiles)
            inhibition_df = pd.read_csv(inhibition_prediction_save_path)
            pharmacokinetics_df = pd.read_csv(pharma_prediction_save_path)

        if (smiles in list(inhibition_df['SMILES'])) & (smiles in list(pharmacokinetics_df['SMILES'])):
            input_molecule_df = pharmacokinetics_df.copy()
            input_molecule_df['inhibition'] = inhibition_df['inhibition']
        else:
            _ = get_inhibition_prediction().predict(smiles)
            _ = get_pharmacokinetics_prediction().predict(smiles)
            inhibition_df = pd.read_csv(inhibition_prediction_save_path)
            pharmacokinetics_df = pd.read_csv(pharma_prediction_save_path)
            input_molecule_df = pharmacokinetics_df.copy()
            input_molecule_df['inhibition'] = inhibition_df['inhibition']

        input_molecule_df.to_csv(seed_dir, index = False)
        
        try: 
            process = subprocess.Popen([mo_script_path, 
                                        seed_dir, 
                                        output_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
        except subprocess.CalledProcessError as e:
            output = e.output
            print(output)

        mo_result = pd.read_csv(output_dir + 'best_siblings.csv')
        if smiles not in list(mo_result['seed']):
            raise FileNotFoundError()
        mo_result = mo_result[['SMILES','improved_admet_score_by']]
        output_dict = mo_result.to_dict(orient='records')

        print(output_dict)

        return output_dict

        
    def __call__(self, **kwargs):
        params = {
            "smiles": kwargs["smiles"],
        }

        return  self.molecular_optimization_step(params['smiles'])

class MoleculeGeneration(BaseAction):
    def __init__(self) -> None:
        action_name = "MoleculeGeneration"
        action_desc = "Using this action to generate a molecule that has potential inhibition to M.tb"
        params_doc = {"None": "No input required"}

        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc,
        )
    
    def molecular_generation(self):

        synthemol_generation_script_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/molecule_generation_actions/synthemol_generate.sh'

        try:
            generated_molecules = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/synthemol_generated_molecules/molecules.csv')
        except:
            try: 
                process = subprocess.Popen([synthemol_generation_script_path], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = process.communicate()
            except subprocess.CalledProcessError as e:
                output = e.output
                print(output)
            
            generated_molecules = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/synthemol_generated_molecules/molecules.csv')

        return generated_molecules['smiles'][random.randint(0,len(generated_molecules)-1)]

        
    def __call__(self, **kwargs):
        params = {
            "None": "No input required"
        }

        return  self.molecular_generation()

class DescribeMolecule(BaseAction):
    def __init__(self) -> None:
        action_name = 'DescribeMolecule'
        action_desc = "Using this action to generate a description of the molecule"
        params_doc = {"smiles": "this is the smiles string of the molecule that needs to be described"}

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def describe_molecule(self, smiles):

        prompt,iupac_success = prompt_generator(smiles)

        llm = ChatOpenAI(model_name="gpt-4o")

        wrong_format = True
        limit = 10
        attempt = 0

        while wrong_format and (attempt < limit):
            response = llm.invoke(prompt).content
            response = response.replace('*','')

            if iupac_success:
                return response
            else:
                try:
                    response = response.split("Output:\n",1)[1]
                    wrong_format = False
                except:
                    try: 
                        response = response.split("Output: ",1)[1]
                        wrong_format = False
                    except:
                        wrong_format = False
                        # print(response)

        return response

    def __call__(self, **kwargs):

        params = {
            'smiles': kwargs['smiles'],
        }

        return self.describe_molecule(params['smiles'])

class SearchLiterature(BaseAction):
    def __init__(self) -> None:
        action_name = 'SearchLiterature'
        action_desc = "Using this action to search literatures for the described molecule"
        params_doc = {"description": "This is the molecular description of the molecule that needs to search for domain knowledge"}

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def search_literature(self, description):

        save_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/paperqa_output.txt'
        paper_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/paper_repo/admet/'

        paperqa_script = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/biochemist_actions/paperqa.sh'

        try: 
            process = subprocess.Popen([paperqa_script, 
                            description, 
                            paper_dir,
                            save_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
        except subprocess.CalledProcessError as e:
            output = e.output
            print(output)
            exit()

        with open(save_dir, 'r') as file:
            paperqa_output = file.read()  # Read the entire file

        return paperqa_output

    def __call__(self, **kwargs):

        params = {
            'description': kwargs['description'],
        }

        return self.search_literature(params['description'])

class RationaleExtraction(BaseAction):
    def __init__(self) -> None:
        action_name = 'RationaleExtraction'
        action_desc = "Using this action to extract the rationales that makes the inhibition prediction model think the molecule is inhibitory."
        params_doc = {"smiles": "The SMILES structure of the pending testing molecule"}

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def rationale_extraction(self, smiles):

        rationale_extraction_script_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/inhibition_prediction_actions/get_rationale_MCTS.sh'
        rationale_input_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/rationale_input.csv'
        rationale_output_path = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/rationale_list.npy'

        input_molecule_df = pd.DataFrame()
        input_molecule_df['SMILES'] = [smiles]
        input_molecule_df.to_csv(rationale_input_path,
                                 index = False)
        
        try: 
            process = subprocess.Popen([rationale_extraction_script_path, 
                                        rationale_input_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
        except subprocess.CalledProcessError as e:
            output = e.output
            print(output)
        
        rationale_output = np.load(rationale_output_path)
        
        return rationale_output

    def rationale_evaluation(self,
                             rationale_description,
                             paper_qa=False):
        

        prompt = "You are an experienced medicinal biochemist and you're looking for novel antibiotics for M.tb. \
                    You'll be provided a molecule with its description, please evaluate if the molecule has potential to become a \
                    novel antibiotic based on the molecule description and the rationales description, please also consider the stability and molecular size. \
                    You'll also be provided a list of descriptions of rationales / substructures of that molecule \
                    predicted by a separate inhibition prediction model that leads to the inhibitory property of the molecule. \
                    The prediction may be correct or wrong, please evaluate if the predicted rationales are reasonable or not reasonable."

        instruction = "Please respond in the following format \
                        Molecule: [Potential / No Potential] [Reasoning] \
                        Rationale: [Reasonable / Unreasonable] [Reasoning]"

        prompt = prompt + rationale_description + instruction

        if paper_qa:
            
            print("Using Paper QA for In-silico Biochemist")

            save_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/paperqa_output_rationale_extraction.txt'
            paper_dir = '/n/data1/hms/dbmi/farhat/alex/AgentLite/paper_repo/inhibition_stability/'

            paperqa_script = '/n/data1/hms/dbmi/farhat/alex/AgentLite/agentlite/actions/biochemist_actions/paperqa.sh'

            try: 
                process = subprocess.Popen([paperqa_script, 
                                prompt, 
                                paper_dir,
                                save_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = process.communicate()
            except subprocess.CalledProcessError as e:
                output = e.output
                print(output)

            with open(save_dir, 'r') as file:
                response = file.read()  # Read the entire file
            
        else:
            llm = ChatOpenAI(model_name="gpt-4o")

            response = llm.invoke(prompt).content

        return response

    def __call__(self, **kwargs):
        params = {
            "smiles": kwargs["smiles"],
        }

        rationales =  self.rationale_extraction(params['smiles'])
        output_file = '/n/data1/hms/dbmi/farhat/alex/AgentLite/intermediate_output/rationale_description.txt'
        describer = DescribeMolecule()

        rationale_descriptions = ''
        with open(output_file, 'w+') as f:

            molecule_description = describer(smiles = params['smiles'])

            rationale_descriptions = rationale_descriptions + f"Molecule SMILES: {params['smiles']} \n"
            rationale_descriptions = rationale_descriptions + molecule_description
            rationale_descriptions = rationale_descriptions + '\n'

            for rationale in rationales[:1]:

                # f.write(f"Rationale SMILES: {rationale} \n")
                # f.write(describer(smiles = rationale))
                # f.write('\n')
                # f.write('\n')

                rationale_descriptions = rationale_descriptions + f"Rationale SMILES: {rationale} \n"
                rationale_descriptions = rationale_descriptions + describer(smiles = rationale)
                rationale_descriptions = rationale_descriptions + '\n \n'
                f.write(rationale_descriptions)

        evaluation = self.rationale_evaluation(rationale_descriptions,
                                               paper_qa=True)

        return evaluation


if __name__ == "__main__":
    # action = MolecularOptimization()
    # print(action(smiles = 'CCCc1ccc(CN2CCc3cc(CCCN4CCN(C)CC4)sc3C2)cc1'))
    # action = DescribeMolecule()
    # print(action(smiles = 'CCCc1ccc(CN2CCc3cc(CCCN4CCN(C)CC4)sc3C2)cc1'))
    # action = SearchLiterature()
    # print(action(description = 'This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group.'))

    action = RationaleExtraction()

    evaluation_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/AgentLite/top_100_Inna.csv')

    with open('/n/data1/hms/dbmi/farhat/alex/AgentLite/biochemist_agent_evaluation_paperqa.txt','w+') as f:
        for smiles in tqdm(list(evaluation_df['SMILES'])[:1]):
            f.write(smiles)
            f.write('\n')
            try:
                f.write(action(smiles = smiles))
            except:
                f.write("Fail to Process")
            f.write('\n')
            f.write('\n')
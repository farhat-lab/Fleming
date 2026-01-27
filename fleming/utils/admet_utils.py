from utils.utils import *
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
#from utils.admet_model import ADMETModel
from admet_ai import ADMETModel
import sys
import os

# Add external_tools to path to allow importing sascore and scscore
current_dir = os.path.dirname(os.path.abspath(__file__))
external_tools_path = os.path.join(current_dir, 'external_tools')
if external_tools_path not in sys.path:
    sys.path.append(external_tools_path)

try:
    from sascore.sascore import calculateScore as calculate_sascore
except ImportError:
    print("Warning: Could not import sascore. SAscore calculation will not be available.")
    calculate_sascore = None

from tqdm import tqdm

try:
    from scscore.scscore.standalone_model_numpy import SCScorer
except ImportError:
    print("Warning: Could not import SCScorer. SCScore calculation will not be available.")
    SCScorer = None

'''
Dictionary to map ADMET properties to their respective thresholds
'''
lower_the_better = ['AMES', 'BBB_Martins','CYP1A2_Veith','CYP2C19_Veith','PPBR_AZ',
                    'CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith','Carcinogens_Lagunin',
                    'ClinTox','DILI','NR-AR-LBD','NR-AR','NR-AhR',
                    'NR-Aromatase','NR-ER-LBD','NR-ER','NR-PPAR-gamma','SR-ATAD5',
                    'SR-ARE','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                    'hERG','Pgp_Broccatelli','LD50_Zhu','Lipophilicity_AstraZeneca',]

higher_the_better = ['Bioavailability_Ma', 'HIA_Hou','PAMPA_NCATS','Caco2_Wang','Clearance_Hepatocyte_AZ',
                     'Clearance_Microsome_AZ','Half_Life_Obach','Solubility_AqSolDB',
                     'VDss_Lombardo']

desired_property = ['AMES','BBB_Martins','Bioavailability_Ma','CYP1A2_Veith','CYP2C19_Veith',
                        'CYP2C9_Substrate_CarbonMangels','CYP2C9_Veith','CYP2D6_Substrate_CarbonMangels','CYP2D6_Veith','CYP3A4_Substrate_CarbonMangels',
                        'CYP3A4_Veith', 'Carcinogens_Lagunin' ,'ClinTox' ,'DILI','HIA_Hou',
                        'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                        'NR-ER','NR-PPAR-gamma','PAMPA_NCATS','Pgp_Broccatelli','SR-ARE',
                        'SR-ATAD5','SR-HSE','SR-MMP','SR-p53','Skin_Reaction',
                        'hERG','Caco2_Wang','Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach',
                        'HydrationFreeEnergy_FreeSolv','LD50_Zhu','Lipophilicity_AstraZeneca','PPBR_AZ','Solubility_AqSolDB',
                        'VDss_Lombardo']

task_weight_dict = {
    ## Absorption Task:
    'Bioavailability_Ma': 1,
    'HIA_Hou': 1,
    'PAMPA_NCATS': 1,
    'Pgp_Broccatelli': 1,
    'Caco2_Wang': 1,
    'Lipophilicity_AstraZeneca': 1,
    'Solubility_AqSolDB': 1,

    ## Distribution Tasks:
    'BBB_Martins': 1,
    'PPBR_AZ': 1,
    'VDss_Lombardo': 1,

    ## Metabolism Tasks:
    'CYP1A2_Veith': 1, # 2
    'CYP2C19_Veith':  1, # 2
    # 'CYP2C9_Substrate_CarbonMangels': 1,
    'CYP2C9_Veith': 1, # 3
    # 'CYP2D6_Substrate_CarbonMangels': 1,
    'CYP2D6_Veith': 1, # 2
    # 'CYP3A4_Substrate_CarbonMangels': 1,
    'CYP3A4_Veith': 1, #2

    ## Excretion Tasks:
    'Clearance_Hepatocyte_AZ': 1, # 2
    'Clearance_Microsome_AZ': 1, #2
    'Half_Life_Obach': 1, #2

    ## Toxicity Tasks:
    'AMES': 1, #3
    'Carcinogens_Lagunin': 1, #3
    'ClinTox': 1, #3
    'DILI': 1,
    'NR-AR-LBD': 1,
    'NR-AR': 1,
    'NR-AhR': 1,
    'NR-Aromatase': 1,
    'NR-ER-LBD': 1,
    'NR-ER': 1,
    'NR-PPAR-gamma': 1,
    'SR-ARE': 1,
    'SR-ATAD5': 1,
    'SR-HSE': 1,
    'SR-MMP': 1,
    'SR-p53': 1,
    'Skin_Reaction': 1,
    'hERG': 1,
    'LD50_Zhu': 1, #3    
}




from utils.env_manager import run_chemmcp_script

class ADMETUtils(Basic_Agent):
    def __init__(self, config, 
                 agent_name: str = "ADMET Agent", 
                 gradio: bool = False, 
                 prompts: dict = None,
                 special_csv_file: str = None,):
    
            
        super().__init__(config=config, agent_name=agent_name, gradio=gradio, prompts=prompts)

        if special_csv_file is not None:
            self.special_csv_data = pd.read_csv(special_csv_file)
        else:
            self.special_csv_data = None
        
        # Initialize ADMETModel with better error handling and CUDA memory management
        _original_torch_load = torch.load
        def _patched_torch_load_admet(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load_admet
        
        # Store original functions
        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        original_cuda_is_available = torch.cuda.is_available
        
        # Force CPU by hiding CUDA and monkey-patching torch.cuda.is_available
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.cuda.is_available = lambda: False
        
        try:
            print(f"\033[1;34;40mInitializing ADMETModel for {agent_name} on CPU (forced)...\033[0m")
            self.admet_predictor = ADMETModel()
            print(f"\033[1;32;40mADMETModel initialized successfully for {agent_name} on CPU\033[0m")

        except Exception as e:
            print(f"\033[1;31;40mError initializing ADMETModel in ADMETUtils: {e}\033[0m")
            self.admet_predictor = None
        finally:
            # Restore original functions and environment
            torch.cuda.is_available = original_cuda_is_available
            if original_cuda_visible_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
            torch.load = _original_torch_load
        
    def get_binarized_prediction(self, smiles:list):
        '''
        Function to get binarized ADMET prediction of a list of SMILES based on currenet antibiotic ADMET prediction
        '''

        threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = self.get_admet_threshold('model_training/current_antibiotic_admet_predictions.csv')

        preds = self.get_admet_prediction(smiles)

        for key in threshold_value_dict:  
            if threshold_type_dict[key] == 'lower':
                preds['pass_'+key] = np.where(preds[key] <= threshold_value_dict[key],1,0)
            else:
                preds['pass_'+key] = np.where(preds[key] >= threshold_value_dict[key],1,0)

        absorption_tasks = ['Bioavailability_Ma','HIA_Hou', 'PAMPA_NCATS','Pgp_Broccatelli','Solubility_AqSolDB',
                            'Caco2_Wang','Lipophilicity_AstraZeneca']
        absorption_tasks = [f"pass_{item}" for item in absorption_tasks]

        distribution_tasks = ['BBB_Martins','PPBR_AZ','VDss_Lombardo']
        distribution_tasks = [f"pass_{item}" for item in distribution_tasks]

        metabolism_tasks = ['CYP1A2_Veith','CYP2C19_Veith','CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith',]
        metabolism_tasks = [f"pass_{item}" for item in metabolism_tasks]

        excretion_tasks = ['Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach']
        excretion_tasks = [f"pass_{item}" for item in excretion_tasks]

        toxicity_tasks = ['AMES','Carcinogens_Lagunin','ClinTox','DILI',
                        'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                        'NR-ER','NR-PPAR-gamma','SR-ATAD5','SR-HSE','SR-MMP','SR-p53',
                        'Skin_Reaction','hERG','LD50_Zhu','SR-ARE']
        toxicity_tasks = [f"pass_{item}" for item in toxicity_tasks]

        total_task_num = len(absorption_tasks) + len(distribution_tasks) + len(metabolism_tasks) + len(excretion_tasks) + len(toxicity_tasks)

        preds['fail_absorption'] = len(absorption_tasks) - preds[absorption_tasks].sum(axis=1)
        preds['fail_distribution'] = len(distribution_tasks) - preds[distribution_tasks].sum(axis=1)
        preds['fail_metabolism'] = len(metabolism_tasks) - preds[metabolism_tasks].sum(axis=1)
        preds['fail_excretion'] = len(excretion_tasks) - preds[excretion_tasks].sum(axis=1)
        preds['fail_toxicity'] = len(toxicity_tasks) - preds[toxicity_tasks].sum(axis=1)
        preds['fail_sum'] = total_task_num - preds[preds.columns[pd.Series(preds.columns).str.startswith('pass_')]].sum(axis=1)
        

        ## Calculating ADMET Scores
        preds['absorption_admet_score'] = 0
        for task in absorption_tasks:
            preds['absorption_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['distribution_admet_score'] = 0
        for task in distribution_tasks:
            preds['distribution_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['metabolism_admet_score'] = 0
        for task in metabolism_tasks:
            preds['metabolism_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['excretion_admet_score'] = 0
        for task in excretion_tasks:
            preds['excretion_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['toxicity_admet_score'] = 0
        for task in toxicity_tasks:
            preds['toxicity_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]

        # Calculate overall ADMET score (this was missing!)
        preds['admet_score'] = preds['absorption_admet_score'] + preds['distribution_admet_score'] + preds['metabolism_admet_score'] \
                              + preds['excretion_admet_score'] + preds['toxicity_admet_score']

        return preds

    def get_admet_prediction_special(self, smile:str):
        if self.special_csv_data is not None:
            filtered_data = self.special_csv_data[self.special_csv_data['smiles'] == smile]
            if not filtered_data.empty:
                return self.special_csv_data[self.special_csv_data['smiles'] == smile]['target'].values[0]
            else:
                print(f"SMILES {smile} not found in special CSV data. Proceeding with model prediction.")
        else:
            raise Exception("This should not be possible")

    def get_admet_prediction_string_summary(self, smiles_list):
        """
        Get ADMET prediction summary for a list of SMILES
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
            
        Returns:
            String with ADMET summary for all molecules
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        all_summaries = []
        
        for i, smiles in enumerate(smiles_list):
            preds = self.get_binarized_prediction([smiles])
            
            # Get the first row since we're dealing with a single molecule
            pred_row = preds.iloc[0]
            
            summary = f"""ADMET Prediction Summary for SMILES {i+1}: {smiles}

Overall ADMET Score: {pred_row['admet_score']:.2f} (higher is better, perfect score is 0.)

Absorption Properties:
- Score: {pred_row['absorption_admet_score']:.2f}
- Failed tasks: {pred_row['fail_absorption']:.0f}/7

Distribution Properties:
- Score: {pred_row['distribution_admet_score']:.2f}
- Failed tasks: {pred_row['fail_distribution']:.0f}/3

Metabolism Properties:
- Score: {pred_row['metabolism_admet_score']:.2f}
- Failed tasks: {pred_row['fail_metabolism']:.0f}/5

Excretion Properties:
:- Score: {pred_row['excretion_admet_score']:.2f}
- Failed tasks: {pred_row['fail_excretion']:.0f}/3

Toxicity Properties:
- Score: {pred_row['toxicity_admet_score']:.2f}
- Failed tasks: {pred_row['fail_toxicity']:.0f}/19

Total Failed ADMET Tasks: {pred_row['fail_sum']:.0f}/37"""
            
            all_summaries.append(summary)
        
        # Join all summaries with separators
        if len(all_summaries) > 1:
            return "\n\n" + "="*80 + "\n\n".join(all_summaries)
        else:
            return all_summaries[0]

    def get_binarized_prediction_batch(self, smiles_list:list):
        threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = self.get_admet_threshold('model_training/current_antibiotic_admet_predictions.csv')

        preds = self.get_admet_prediction_batch(smiles_list)

        for key in threshold_value_dict:  
            if threshold_type_dict[key] == 'lower':
                preds['pass_'+key] = np.where(preds[key] <= threshold_value_dict[key],1,0)
            else:
                preds['pass_'+key] = np.where(preds[key] >= threshold_value_dict[key],1,0)

        absorption_tasks = ['Bioavailability_Ma','HIA_Hou', 'PAMPA_NCATS','Pgp_Broccatelli','Solubility_AqSolDB',
                            'Caco2_Wang','Lipophilicity_AstraZeneca']
        absorption_tasks = [f"pass_{item}" for item in absorption_tasks]

        distribution_tasks = ['BBB_Martins','PPBR_AZ','VDss_Lombardo']
        distribution_tasks = [f"pass_{item}" for item in distribution_tasks]

        metabolism_tasks = ['CYP1A2_Veith','CYP2C19_Veith','CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith',]
        metabolism_tasks = [f"pass_{item}" for item in metabolism_tasks]

        excretion_tasks = ['Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach']
        excretion_tasks = [f"pass_{item}" for item in excretion_tasks]

        toxicity_tasks = ['AMES','Carcinogens_Lagunin','ClinTox','DILI',
                        'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                        'NR-ER','NR-PPAR-gamma','SR-ATAD5','SR-HSE','SR-MMP','SR-p53',
                        'Skin_Reaction','hERG','LD50_Zhu','SR-ARE']
        toxicity_tasks = [f"pass_{item}" for item in toxicity_tasks]

        total_task_num = len(absorption_tasks) + len(distribution_tasks) + len(metabolism_tasks) + len(excretion_tasks) + len(toxicity_tasks)

        preds['fail_absorption'] = len(absorption_tasks) - preds[absorption_tasks].sum(axis=1)
        preds['fail_distribution'] = len(distribution_tasks) - preds[distribution_tasks].sum(axis=1)
        preds['fail_metabolism'] = len(metabolism_tasks) - preds[metabolism_tasks].sum(axis=1)
        preds['fail_excretion'] = len(excretion_tasks) - preds[excretion_tasks].sum(axis=1)
        preds['fail_toxicity'] = len(toxicity_tasks) - preds[toxicity_tasks].sum(axis=1)
        preds['fail_sum'] = total_task_num - preds[preds.columns[pd.Series(preds.columns).str.startswith('pass_')]].sum(axis=1)
        

        ## Calculating ADMET Scores
        preds['absorption_admet_score'] = 0
        for task in absorption_tasks:
            preds['absorption_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['distribution_admet_score'] = 0
        for task in distribution_tasks:
            preds['distribution_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['metabolism_admet_score'] = 0
        for task in metabolism_tasks:
            preds['metabolism_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['excretion_admet_score'] = 0
        for task in excretion_tasks:
            preds['excretion_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]
        
        preds['toxicity_admet_score'] = 0
        for task in toxicity_tasks:
            preds['toxicity_admet_score'] += (preds[task] - 1) * task_weight_dict[task[5:]]

        # Calculate overall ADMET score
        preds['admet_score'] = preds['absorption_admet_score'] + preds['distribution_admet_score'] + preds['metabolism_admet_score'] \
                              + preds['excretion_admet_score'] + preds['toxicity_admet_score']

        return preds
    
    @staticmethod
    def get_admet_threshold(current_antibiotic_admet_prediction_dir: str):
        '''
        Function to get ADMET thresholds from the current antibiotic ADMET prediction file
        '''
        # Resolve path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, current_antibiotic_admet_prediction_dir)
        
        threshold_value_dict = {}
        threshold_antibiotic_dict = {}
        threshold_type_dict = {}
        
        if not os.path.exists(full_path):
             # Fallback or error
             print(f"Warning: Threshold file not found at {full_path}")
             # Return empty or defaults to avoid crash
             return {}, {}, {}

        admet_prediction = pd.read_csv(full_path)

        for feature in lower_the_better:
            if feature in admet_prediction.columns:
                threshold_value_dict[feature] = admet_prediction[feature].quantile(0.9)
                threshold_antibiotic_dict[feature] = admet_prediction['Name']
                threshold_type_dict[feature] = 'lower'
        for feature in higher_the_better:
            if feature in admet_prediction.columns:
                threshold_value_dict[feature] = admet_prediction[feature].quantile(0.1)
                threshold_antibiotic_dict[feature] = admet_prediction['Name']
                threshold_type_dict[feature] = 'higher'
        
        return threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict

    def get_admet_prediction(self, smiles:list):
        '''
        Function to get ADMET prediction of a list of SMILES
        '''
        if self.admet_predictor is None:
            raise RuntimeError("ADMETModel (self.admet_predictor) is not initialized in ADMETUtils.")

        if isinstance(smiles, str):
            smiles = [smiles]

        preds = self.admet_predictor.predict(smiles)

        preds = preds[desired_property]
        preds.insert(0,'SMILES',smiles)

        return preds

    def get_admet_prediction_batch(self, smiles_list:list):
        if self.admet_predictor is None:
            raise RuntimeError("ADMETModel (self.admet_predictor) is not initialized in ADMETUtils.")

        preds = self.admet_predictor.predict(smiles_list)

        preds = preds[desired_property]
        preds.insert(0,'SMILES',smiles_list)

        return preds
    
    def get_cytotoxicity_copilot_mode(self, smiles_list):
        """
        Cytotoxicity Prediction for a list of SMILES
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
            
        Returns:
            String with cytotoxicity predictions for all molecules
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Use config or defaults for model paths, resolvable
        # Currently hardcoded in original file:
        # '/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/fold_0/model_0/model.pt'
        # I should change this to use relative paths if possible, or config.
        # Since I don't have these paths in config, I'll use placeholders relative to root.
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_paths = [
            'model_checkpoints/cytotox_models/hepg2_tox/fold_0/model_0/model.pt',
            'model_checkpoints/cytotox_models/hskmc_tox/fold_0/model_0/model.pt',
            'model_checkpoints/cytotox_models/imr90_tox/fold_0/model_0/model.pt'
        ]
        
        model_paths = [os.path.join(project_root, p) for p in default_paths]
        
        # Check if they exist
        if not all(os.path.exists(p) for p in model_paths):
            return "Cytotoxicity models not found."

        model_list = self.load_models(model_paths=model_paths, device='cpu')
        
        all_results = []
        
        for i, smiles in enumerate(smiles_list):
            preds_list = []
            uncertainties_list = []
            for c in model_list:
                preds, uncertainties = self.make_prediction([c], [smiles])
                preds_list.append(preds)
                uncertainties_list.append(uncertainties)

            result = f"For molecule {i+1} ({smiles}), the predicted hepatocyte toxicity prediction is {preds_list[0]} with an uncertainty of {uncertainties_list[0]}. The predicted skeletal muscle toxicity prediction is {preds_list[1]} with an uncertainty of {uncertainties_list[1]}. The predicted lung (imr90) toxicity prediction is {preds_list[2]} with an uncertainty of {uncertainties_list[2]}."
            all_results.append(result)
        
        return "\n\n".join(all_results)
    

    # calculate_novelty and get_novelty_and_sascore removed (inherited from Basic_Agent)
    
    def get_admet_prediction_raw(self, smiles_list):
        """
        Get raw ADMET predictions for all tasks for a list of SMILES
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Get raw predictions for all molecules at once (more efficient)
        preds_df = self.get_admet_prediction(smiles_list)
        
        all_results = []
        
        for i, smiles in enumerate(smiles_list):
            # Get the row for this molecule
            mol_preds = preds_df.iloc[i]
            
            result = f"Raw ADMET Predictions for Molecule {i+1}: {smiles}\n"
            result += "=" * 60 + "\n"
            
            # Group tasks by category for better organization
            absorption_tasks = ['Bioavailability_Ma', 'HIA_Hou', 'PAMPA_NCATS', 
                               'Pgp_Broccatelli', 'Caco2_Wang', 'Lipophilicity_AstraZeneca', 
                               'Solubility_AqSolDB']
            distribution_tasks = ['BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo']
            metabolism_tasks = ['CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 
                               'CYP2D6_Veith', 'CYP3A4_Veith']
            excretion_tasks = ['Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach']
            toxicity_tasks = ['AMES', 'Carcinogens_Lagunin', 'ClinTox', 'DILI',
                             'NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD',
                             'NR-ER', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 
                             'SR-MMP', 'SR-p53', 'Skin_Reaction', 'hERG', 'LD50_Zhu']
            
            # Add predictions by category
            result += "\nABSORPTION PROPERTIES:\n"
            for task in absorption_tasks:
                if task in mol_preds:
                    better_direction = "(lower is better)" if task in lower_the_better else "(higher is better)"
                    result += f"  {task}: {mol_preds[task]:.4f} {better_direction}\n"
            
            result += "\nDISTRIBUTION PROPERTIES:\n"
            for task in distribution_tasks:
                if task in mol_preds:
                    better_direction = "(lower is better)" if task in lower_the_better else "(higher is better)"
                    result += f"  {task}: {mol_preds[task]:.4f} {better_direction}\n"
            
            result += "\nMETABOLISM PROPERTIES:\n"
            for task in metabolism_tasks:
                if task in mol_preds:
                    better_direction = "(lower is better)" if task in lower_the_better else "(higher is better)"
                    result += f"  {task}: {mol_preds[task]:.4f} {better_direction}\n"
            
            result += "\nEXCRETION PROPERTIES:\n"
            for task in excretion_tasks:
                if task in mol_preds:
                    better_direction = "(lower is better)" if task in lower_the_better else "(higher is better)"
                    result += f"  {task}: {mol_preds[task]:.4f} {better_direction}\n"
            
            result += "\nTOXICITY PROPERTIES:\n"
            for task in toxicity_tasks:
                if task in mol_preds:
                    better_direction = "(lower is better)" if task in lower_the_better else "(higher is better)"
                    result += f"  {task}: {mol_preds[task]:.4f} {better_direction}\n"
            
            all_results.append(result)
        
        # Join all results with separators
        if len(all_results) > 1:
            return "\n\n" + "\n" + "="*100 + "\n".join(all_results)
        else:
            return all_results[0]

    def get_admet_prediction_raw_table(self, smiles_list):
        """
        Get raw ADMET predictions in a tabular format for a list of SMILES
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Get raw predictions for all molecules at once
        preds_df = self.get_admet_prediction(smiles_list)
        
        # Create markdown table
        lines = []
        
        # Header
        lines.append("| SMILES | Property | Value | Direction |")
        lines.append("|--------|----------|-------|-----------|")
        
        for i, smiles in enumerate(smiles_list):
            mol_preds = preds_df.iloc[i]
            
            # Add rows for each property
            for prop in desired_property:
                if prop in mol_preds:
                    value = mol_preds[prop]
                    direction = "Lower Better" if prop in lower_the_better else "Higher Better"
                    lines.append(f"| {smiles} | {prop} | {value:.4f} | {direction} |")
        
        return "\n".join(lines)
    
    def get_side_effect_prediction(self, smiles_list):
        answer = "The following is the side effect prediction profile, trained on 22820 molecules, for the following molecules: \n"
        
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        smiles_batch = ','.join(smiles_list)
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_tools', 'chemmcp_tools', 'sider.py')
        
        output = run_chemmcp_script(
            script_path,
            ['--smiles', smiles_batch]
        )
        
        if "===RESULT===" in output.stdout:
            answer += output.stdout.split("===RESULT===")[1].strip()
        else:
            answer += output.stdout.strip()

        return answer
    
    def get_solubility_prediction(self, smiles_list):
        answer = "The following is the solubility prediction, trained on 888 molecules, for the following molecules: \n"
        
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        smiles_batch = ','.join(smiles_list)
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_tools', 'chemmcp_tools', 'solubility.py')
        
        output = run_chemmcp_script(
            script_path,
            ['--smiles', smiles_batch]
        )
        
        if "===RESULT===" in output.stdout:
            answer += output.stdout.split("===RESULT===")[1].strip()
        else:
            answer += output.stdout.strip()

        return answer
    
    def get_toxicity_prediction(self, smiles_list): 
        answer = "The following is the toxicity prediction, trained on 1114 molecules, for the following molecules: \n"
        
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        smiles_batch = ','.join(smiles_list)
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_tools', 'chemmcp_tools', 'toxicity.py')
        
        output = run_chemmcp_script(
            script_path,
            ['--smiles', smiles_batch]
        )
        
        if "===RESULT===" in output.stdout:
            answer += output.stdout.split("===RESULT===")[1].strip()
        else:
            answer += output.stdout.strip()

        return answer
    
    def get_bbbp_prediction(self, smiles_list):
        answer = "The following is the bbbp prediction, trained on 1569 molecules, for the following molecules: \n"
        
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        smiles_batch = ','.join(smiles_list)
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_tools', 'chemmcp_tools', 'bbbp.py')
        
        output = run_chemmcp_script(
            script_path,
            ['--smiles', smiles_batch]
        )
        
        if "===RESULT===" in output.stdout:
            answer += output.stdout.split("===RESULT===")[1].strip()
        else:
            answer += output.stdout.strip() 
        return answer

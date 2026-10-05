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

ADMET_MODEL_CONTEXT = (
    "Model evidence context: ADMET-AI provides one quantitative line of evidence, to be weighed together with the other model outputs and the literature rather than used as a default answer, across heterogeneous TDC-style "
    "ADMET tasks. The pass/fail thresholds are calibrated against current antibiotic predictions when that file "
    "is available; they are not direct clinical safety calls. Robustness is higher when several related tasks agree "
    "and the value is clearly on the favorable side of the threshold, and lower when predictions are close to a "
    "threshold, unsupported by task-specific training coverage, or contradicted by literature on similar scaffolds."
)

CHEMMCP_MODEL_CONTEXT = {
    "side_effect": (
        "Model evidence context: side-effect profile model trained on 22,820 molecules. Treat as a relatively broad "
        "screening prior, but reconcile with known adverse-event literature for the exact scaffold or close analogs."
    ),
    "solubility": (
        "Model evidence context: solubility model trained on 888 molecules. This is a small dataset, so literature or "
        "measured solubility for similar ionization states/scaffolds should substantially affect the final judgment."
    ),
    "toxicity": (
        "Model evidence context: toxicity model trained on 1,114 molecules. This is a modest dataset; if literature "
        "reports toxicity for close analogs or reactive/toxicophores, down-weight reassuring predictions."
    ),
    "bbbp": (
        "Model evidence context: BBBP model trained on 1,569 MoleculeNet molecules. Use as a modest prior and adjust "
        "for literature on CNS penetration, polarity, charge state, P-gp liability, and close scaffold behavior."
    ),
}

CYTOTOX_MODEL_CONTEXT = (
    "Model evidence context: cytotoxicity predictions are Chemprop model outputs with Dirichlet uncertainty "
    "estimates. Both numbers run 0-1. For the prediction, HIGHER means MORE predicted cytotoxicity (more likely to "
    "kill that cell line). For the uncertainty, HIGHER means LESS confident - it is the Dirichlet vacuity K/S, which "
    "is 1.0 when the model saw no evidence either way and falls toward 0 as evidence accumulates, so a low "
    "uncertainty is a confident call and a high one is close to a coin flip. Treat the predictions as screening "
    "evidence rather than definitive safety results. These three predictions are for HepG2, HSKMC and IMR90 "
    "specifically. They are NOT a prediction for any other cell line or species."
)

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

    @staticmethod
    def _format_bayesian_context():
        return (
            f"{ADMET_MODEL_CONTEXT}\n"
            "Synthesis guidance: do not treat any single model output as the starting point and everything else "
            "as an adjustment to it. Report every quantitative result you obtained, each with its uncertainty, "
            "and every literature finding, then weigh them together into one integrated judgement. Where several "
            "models cover the same endpoint, use all of them: agreement across models trained on different "
            "assays is stronger evidence than any one of them, and disagreement means the endpoint is not well "
            "determined by the models at all. A model that was not trained on the exact endpoint being asked "
            "about is evidence, not an answer. Say which sources agree, which conflict, how you resolved the "
            "conflict, and what carried the most weight."
        )

    def _get_cytotoxicity_model_paths(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_paths = [
            'model_checkpoints/hepg2_model.pt',
            'model_checkpoints/hskmc_model.pt',
            'model_checkpoints/imr90_model.pt',
        ]
        configured_paths = self.config.get('cytotox_model_paths', default_paths)
        return [
            path if os.path.isabs(path) else os.path.join(project_root, path)
            for path in configured_paths
        ]
        
    def get_binarized_prediction(self, smiles:list):
        '''
        Function to get binarized ADMET prediction of a list of SMILES based on currenet antibiotic ADMET prediction
        '''

        threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = self.get_admet_threshold('model_training/current_antibiotic_admet_predictions.csv')

        preds = self.get_admet_prediction(smiles)

        # If thresholds are unavailable (e.g., threshold file missing), return raw predictions with
        # placeholder columns so callers don't crash.
        if not threshold_value_dict:
            for col in [
                'fail_absorption', 'fail_distribution', 'fail_metabolism', 'fail_excretion', 'fail_toxicity', 'fail_sum',
                'absorption_admet_score', 'distribution_admet_score', 'metabolism_admet_score', 'excretion_admet_score', 'toxicity_admet_score',
                'admet_score',
            ]:
                preds[col] = np.nan
            preds['thresholds_available'] = False
            return preds

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

        preds['thresholds_available'] = True
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

            if not bool(pred_row.get('thresholds_available', True)):
                raw_table = self.get_admet_prediction_raw_table([smiles])
                summary = (
                    f"ADMET Prediction Summary for SMILES {i+1}: {smiles}\n\n"
                    "Note: pass/fail thresholds are unavailable (missing `model_training/current_antibiotic_admet_predictions.csv`), "
                    "so task-level pass/fail counts and aggregate scores are not computed.\n\n"
                    f"{self._format_bayesian_context()}\n\n"
                    "Raw predicted task values (model outputs):\n"
                    f"{raw_table}"
                )
                all_summaries.append(summary)
                continue
            
            summary = f"""ADMET Prediction Summary for SMILES {i+1}: {smiles}

Overall ADMET Score: {pred_row['admet_score']:.2f} (0 is best; more negative means more predicted failures across tasks.)

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

Total Failed ADMET Tasks: {pred_row['fail_sum']:.0f}/37

{self._format_bayesian_context()}"""
            
            all_summaries.append(summary)
        
        # Join all summaries with separators
        if len(all_summaries) > 1:
            return "\n\n" + "="*80 + "\n\n".join(all_summaries)
        else:
            return all_summaries[0]

    def get_binarized_prediction_batch(self, smiles_list:list):
        threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = self.get_admet_threshold('model_training/current_antibiotic_admet_predictions.csv')

        preds = self.get_admet_prediction_batch(smiles_list)

        if not threshold_value_dict:
            for col in [
                'fail_absorption', 'fail_distribution', 'fail_metabolism', 'fail_excretion', 'fail_toxicity', 'fail_sum',
                'absorption_admet_score', 'distribution_admet_score', 'metabolism_admet_score', 'excretion_admet_score', 'toxicity_admet_score',
                'admet_score',
            ]:
                preds[col] = np.nan
            preds['thresholds_available'] = False
            return preds

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

        preds['thresholds_available'] = True
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
        
        model_paths = self._get_cytotoxicity_model_paths()
        
        # Check if they exist
        if not all(os.path.exists(p) for p in model_paths):
            missing_paths = [p for p in model_paths if not os.path.exists(p)]
            return f"Cytotoxicity models not found: {missing_paths}"

        model_list = self.load_models(model_paths=model_paths, device='cpu')
        
        all_results = []

        def _scalar(value):
            """
            One number out of whatever make_prediction returned.

            It returns a length-1 numpy array, which an f-string renders as "[0.31]".
            Handing the model an array literal makes the value harder to read and to quote
            back, and it is the same number either way.
            """
            try:
                array = np.asarray(value).ravel()
                return f"{float(array[0]):.3f}" if array.size else str(value)
            except Exception:
                return str(value)
        
        for i, smiles in enumerate(smiles_list):
            preds_list = []
            uncertainties_list = []
            for c in model_list:
                preds, uncertainties = self.make_prediction([c], [smiles])
                preds_list.append(preds)
                uncertainties_list.append(uncertainties)

            # Both scales are spelled out on every number. The inhibition tool has always
            # done this ("0-1, higher = less confident"); this tool used to emit a bare
            # "with an uncertainty of X", leaving the agent to guess whether a high value
            # meant high confidence or low - and a guess that lands the wrong way inverts
            # every cautionary adjustment the prompt asks for.
            result = (
                f"For molecule {i+1} ({smiles}): "
                f"hepatocyte (HepG2) cytotoxicity score (0-1, higher = more predicted cytotoxic): "
                f"{_scalar(preds_list[0])}; "
                f"Dirichlet uncertainty (0-1, higher = less confident): {_scalar(uncertainties_list[0])}. "
                f"Skeletal muscle (HSKMC) cytotoxicity score (0-1, higher = more predicted cytotoxic): "
                f"{_scalar(preds_list[1])}; "
                f"Dirichlet uncertainty (0-1, higher = less confident): {_scalar(uncertainties_list[1])}. "
                f"Lung (IMR90) cytotoxicity score (0-1, higher = more predicted cytotoxic): "
                f"{_scalar(preds_list[2])}; "
                f"Dirichlet uncertainty (0-1, higher = less confident): {_scalar(uncertainties_list[2])}. "
                f"{CYTOTOX_MODEL_CONTEXT} Report all three raw model outputs, then synthesize across them and the "
                f"literature into one judgement rather than adjusting a single value."
            )
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
        answer = (
            "The following is the side effect prediction profile, trained on 22820 molecules, for the following molecules: \n"
            f"{CHEMMCP_MODEL_CONTEXT['side_effect']}\n"
        )
        
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
        answer = (
            "The following is the solubility prediction, trained on 888 molecules, for the following molecules: \n"
            f"{CHEMMCP_MODEL_CONTEXT['solubility']}\n"
        )
        
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
        answer = (
            "The following is the toxicity prediction, trained on 1114 molecules, for the following molecules: \n"
            f"{CHEMMCP_MODEL_CONTEXT['toxicity']}\n"
        )
        
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
        answer = (
            "The following is the bbbp prediction, trained on 1569 molecules, for the following molecules: \n"
            f"{CHEMMCP_MODEL_CONTEXT['bbbp']}\n"
        )
        
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

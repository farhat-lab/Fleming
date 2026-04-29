import os
import sys
import pandas as pd
import pickle
import random
import torch
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
import numpy as np
from os.path import join

# Adjust sys.path to include molecule_diffusion and external tools
# Use absolute paths relative to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

molecule_diffusion_path = os.path.join(project_root, 'molecule_diffusion', 'e3_diffusion_for_molecules')
external_tools_path = os.path.join(current_dir, 'external_tools')

if molecule_diffusion_path not in sys.path:
    sys.path.append(molecule_diffusion_path)

if external_tools_path not in sys.path:
    sys.path.append(external_tools_path)

# Import molecule diffusion modules
try:
    from molecule_diffusion.e3_diffusion_for_molecules.eval_sample import get_dataset_info, compute_mean_mad, sample_different_sizes_and_save
    from qm9.models import get_model
    from qm9 import dataset
except ImportError as e:
    print(f"Warning: Could not import molecule_diffusion modules: {e}")

from utils.utils import Basic_Agent

class MolecularGenerationUtils(Basic_Agent):
    def __init__(self, config, agent_name: str = "Molecular Generation Utils", gradio: bool = False, prompts: dict = None):
        super().__init__(config, agent_name, gradio, prompts)
        
        # Load paths from config or use defaults
        self.diffusion_model_path = self.config.get('diffusion_model_path', "molecule_diffusion/e3_diffusion_for_molecules/outputs/TB_Diffusion_resume")
        self.diffusion_cache_path = self.config.get('diffusion_cache_path', "data/diffusion_mols_cache.pkl")
        self.chemprop_model_path = self.config.get('chemprop_model_path', "model_checkpoints/final_model/fold_0/model_0/model.pt")
        self.last_generation_source = None
        self.last_generated_smiles = []
        
    # Note: calculate_novelty and get_novelty_and_sascore are inherited from Basic_Agent

    def generate_molecule(self, model_name: str = "diffusion", 
                         model_path: str = None,
                         device: str = 'cuda'):
        """
        Generate a molecule using the specified model.
        
        Args:
            model_name: Name of the model to use for generation ("diffusion" or "synthemol")
            model_path: Path to the diffusion model checkpoint (optional, defaults to config)
            device: Device to run models on
            
        Returns:
            Generated molecule SMILES string
        """
        requested_device = (device or 'cuda').strip()
        self.last_generation_source = None
        self.last_generated_smiles = []

        if model_path is None:
            model_path = os.path.join(project_root, self.diffusion_model_path)

        result = None
        # Diffusion inference is expensive and not a good CPU path; prefer the cache when CUDA is unavailable.
        if requested_device != 'cuda' or not torch.cuda.is_available():
            cached = self.get_molecule_from_diffusion_cache(num_to_select=1, cache_path=None)
            if not (isinstance(cached, str) and cached.startswith("Error")):
                self.last_generation_source = "diffusion_cache"
                self.last_generated_smiles = cached if isinstance(cached, list) else [cached]
                return cached

        if model_name == "diffusion":
            result = self._sample_diffusion(model_path, requested_device)
        else:
            result = self._sample_diffusion(model_path, requested_device)

        # Check if result indicates an error or failure
        if isinstance(result, str) and (result.startswith("Error") or "not found" in result):
            print(f"Warning: Molecular generation failed ({result}). Falling back to diffusion cache.")
            cached = self.get_molecule_from_diffusion_cache(num_to_select=1, cache_path=None)
            if not (isinstance(cached, str) and cached.startswith("Error")):
                self.last_generation_source = "diffusion_cache_fallback"
                self.last_generated_smiles = cached if isinstance(cached, list) else [cached]
                return cached
            self.last_generation_source = "generation_error"
            return cached

        self.last_generation_source = "diffusion_model"
        self.last_generated_smiles = result if isinstance(result, list) else [result]
        return result
        
        
    def get_molecule_from_diffusion_cache(self, num_to_select, cache_path: str = None):
        """
        Load a cached molecule from the diffusion model.
        
        Args:
            cache_path: Path to the cached diffusion molecules (optional, defaults to config)
        
        Returns:
            SMILES string of the cached molecule
        """
        if cache_path is None:
            cache_path = os.path.join(project_root, self.diffusion_cache_path)
            
        try:
            with open(cache_path, 'rb') as f:
                cached_molecules = pickle.load(f)
                if isinstance(cached_molecules, dict):
                    cached_molecules = list(cached_molecules.values())
                elif not isinstance(cached_molecules, list):
                    cached_molecules = list(cached_molecules)

                if not cached_molecules:
                    return "Error loading cached molecules: cache is empty"

                selected_mol = random.sample(cached_molecules, min(num_to_select, len(cached_molecules)))
            return selected_mol
        except Exception as e:
            return f"Error loading cached molecules: {e}"

    def generate_molecule_string(self, model_name: str = "diffusion", 
                                model_path: str = None,
                                device: str = 'cuda'):
        """
        Generate a molecule and return a formatted string description.
        
        Args:
            model_name: Name of the model to use for generation ("diffusion" or "synthemol")
            model_path: Path to the diffusion model checkpoint (optional)
            device: Device to run models on
            
        Returns:
            String description of the generated molecule
        """
        try:
            generated_smiles = self.generate_molecule(model_name, model_path, device)
            
            # Handle list output (from cache or successful generation)
            if isinstance(generated_smiles, list):
                if len(generated_smiles) == 1:
                    generated_smiles = generated_smiles[0]
                else:
                    generated_smiles = ", ".join(generated_smiles)

            if model_name == "diffusion":
                if self.last_generation_source == "diffusion_cache":
                    return f"Generated molecule from diffusion cache (no GPU available): {generated_smiles}"
                if self.last_generation_source == "diffusion_cache_fallback":
                    return f"Generated molecule from diffusion cache after diffusion sampling failed: {generated_smiles}"
                return f"Generated molecule using Diffusion model: {generated_smiles}"
            elif model_name == "synthemol":
                return f"Generated molecule using Synthemol model: {generated_smiles}"
            else:
                return f"Error: {generated_smiles}"
                
        except Exception as e:
            return f"Error generating molecule: {e}"

    def _sample_diffusion(self, model_path: str, device: str = 'cuda'):
        """Generate a molecule using the diffusion model."""
        try:
            # Set up device
            torch_device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
            
            # Parse arguments
            parser = argparse.ArgumentParser()
            parser.add_argument('--model_path', type=str, default=model_path)
            parser.add_argument('--n_tries', type=int, default=10)
            parser.add_argument('--n_nodes', type=int, default=19)
            
            eval_args, unparsed_args = parser.parse_known_args()
            
            # Load model arguments
            args_path = join(eval_args.model_path, 'args.pickle')
            if not os.path.exists(args_path):
                 return f"Error: Model args not found at {args_path}"
                 
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            # Handle missing attributes
            if not hasattr(args, 'normalization_factor'):
                args.normalization_factor = 1
            if not hasattr(args, 'aggregation_method'):
                args.aggregation_method = 'sum'

            # Get dataset info and prepare model
            dataset_info = get_dataset_info(args.dataset, args.remove_h)
            dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
            property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
            mean, mad = property_norms['tb_inhibition']['mean'], property_norms['tb_inhibition']['mad'] 

            # Load model
            flow, nodes_dist, prop_dist = get_model(args, torch_device, dataset_info, dataloaders['train'])
            flow.to(torch_device)
            
            fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
            model_file_path = join(eval_args.model_path, fn)
            if not os.path.exists(model_file_path):
                return f"Error: Model file not found at {model_file_path}"
                
            flow_state_dict = torch.load(model_file_path, map_location=torch_device)
            flow.load_state_dict(flow_state_dict)
            
            # Generate molecule
            result_list = []
            for c in range(5):
                result = sample_different_sizes_and_save(
                    args, eval_args, torch_device, flow, nodes_dist,
                    dataset_info=dataset_info, n_samples=30, full_sample=10000, 
                    mean=mean, mad=mad, return_single_molecule=True
                )
                result_list.append(result)
            
            return result_list
                    
        except Exception as e:
            return f"Error in diffusion sampling: {e}"

    def get_chemprop_prediction(self, smiles:list,
                                model_paths:list = ['/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/final_model/fold_0/model_0/model.pt'],
                                device:str='cpu'):
        '''
        Wrapper function to get Chemprop prediction of MTB inhibition on a list of SMILES
        '''
        if model_paths is None:
             model_paths = [os.path.join(project_root, self.chemprop_model_path)]
             
        # Check if models exist
        valid_models = [p for p in model_paths if os.path.exists(p)]
        if not valid_models:
             return ["Model file not found"], ["N/A"]

        try:
            models = self.load_models(valid_models, device)

            if isinstance(smiles,str):
                smiles = [smiles]

            preds, unc = self.make_prediction(models, smiles)

            return preds, unc 
        except AttributeError:
             return ["Prediction functionality not available"], ["N/A"]
        except Exception as e:
             return [f"Error: {e}"], ["N/A"]

    
    def get_chemprop_prediction_string(self, smiles_list):
        '''
        Function to get Chemprop prediction string of MTB inhibition for a list of SMILES for Inhibition Agent
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
            
        Returns:
            String with inhibition predictions for all molecules
        '''
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        all_results = []
        
        for i, smiles in enumerate(smiles_list):
            preds, unc = self.get_chemprop_prediction([smiles])
            result = f"Molecule {i+1} ({smiles}) - Model Prediction (0 ~ 1): {preds}, Model Dirichlet Uncertainty (0 ~ 1): {unc}"
            all_results.append(result)
        
        return "\n".join(all_results)

from utils.utils import *
from utils.mcts_utils import MCTSNode, find_clusters, extract_subgraph, prepare_mcts_structures, MIN_ATOMS
from typing import Dict, List, Set
import numpy as np
from rdkit import Chem

class InhibitionUtils(Basic_Agent):
    
    def get_chemprop_prediction(self, smiles:list,
                                model_paths:list = ['/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/final_model/fold_0/model_0/model.pt'],
                                device:str='cpu'):
        '''
        Wrapper function to get Chemprop prediction of MTB inhibition on a list of SMILES
        '''
        models = self.load_models(model_paths, device)

        if isinstance(smiles,str):
            smiles = [smiles]

        preds, unc = self.make_prediction(models, smiles)

        return preds, unc 

    
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
    
    def _mcts_rollout(self, node: MCTSNode, state_map: Dict[str, MCTSNode], 
                     orig_smiles: str, clusters: List[Set[int]], 
                     atom_cls: List[Set[int]], nei_cls: List[Set[int]], 
                     models) -> float:
        """MCTS rollout using loaded models."""
        cur_atoms = node.atoms
        if len(cur_atoms) <= MIN_ATOMS:
            return node.P

        # Expand if this node has never been visited
        if len(node.children) == 0:
            cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])
            
            for i in cur_cls:
                leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
                if len(nei_cls[i] & cur_cls) == 1 or (len(clusters[i]) == 2 and len(leaf_atoms) == 1):
                    new_atoms = cur_atoms - set(leaf_atoms)
                    if len(new_atoms) > MIN_ATOMS:
                        new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                        if new_smiles and new_smiles != orig_smiles:
                            if new_smiles in state_map:
                                new_node = state_map[new_smiles]
                            else:
                                new_node = MCTSNode(new_smiles, new_atoms)
                                state_map[new_smiles] = new_node
                            node.children.append(new_node)

            if len(node.children) == 0:
                return node.P

            # Score all children using the models directly
            child_smiles = [child.smiles for child in node.children]
            try:
                preds, _ = self.make_prediction(models, child_smiles)
                if isinstance(preds, np.ndarray):
                    scores = preds.flatten().tolist()
                else:
                    scores = [float(p) for p in preds]
                
                for child, score in zip(node.children, scores):
                    child.P = score
            except Exception as e:
                print(f"Error scoring children: {e}")
                for child in node.children:
                    child.P = 0.0

        if len(node.children) == 0:
            return node.P

        sum_count = sum(c.N for c in node.children)
        selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self._mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, models)
        selected_node.W += v
        selected_node.N += 1

        return v

    def _interpret_single_molecule_mcts(self, smiles: str, models, n_rollout: int, 
                                      max_atoms: int, prop_delta: float) -> str:
        """Helper function to interpret a single molecule using MCTS."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Invalid SMILES: {smiles}"
                
            if mol.GetNumAtoms() > 50:
                n_rollout = max(1, n_rollout // 5)  # Reduce rollouts for large molecules

            # Get original molecule score
            original_preds, _ = self.make_prediction(models, [smiles])
            original_score = original_preds[0][0] if isinstance(original_preds[0], (list, np.ndarray)) else original_preds[0]

            # Prepare MCTS data structures using the helper function
            clusters, atom_cls, nei_cls = prepare_mcts_structures(mol)

            # Run MCTS
            root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
            root.P = original_score
            state_map = {smiles: root}
            
            for _ in range(n_rollout):
                try:
                    self._mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, models)
                except Exception as e:
                    print(f"Error in rollout for {smiles}: {e}")
                    continue

            # Filter and format rationales
            rationales = [
                node for _, node in state_map.items() 
                if len(node.atoms) <= max_atoms and node.P >= prop_delta and node.smiles != smiles
            ]

            # Format results as string
            output = f"MCTS Interpretation Results for {smiles}:\n"
            output += f"Original molecule score: {original_score:.3f}\n"
            output += f"Found {len(rationales)} rationales\n"
            
            if rationales:
                rationales.sort(key=lambda x: x.P, reverse=True)
                output += "Top rationales (important substructures):\n"
                for i, rat in enumerate(rationales[:3]):  # Top 3 rationales per molecule
                    output += f"  {i+1}. SMILES: {rat.smiles}, Score: {rat.P:.3f}, "
                    output += f"Atoms: {len(rat.atoms)}, Visits: {rat.N}\n"
            else:
                output += "No significant rationales found\n"
            
            return output
            
        except Exception as e:
            return f"Error in MCTS interpretation for {smiles}: {e}\n"

    def interpret_molecule_mcts(self, smiles_list, 
                               model_paths: list = None,
                               n_rollout: int = 5,
                               max_atoms: int = 20,
                               prop_delta: float = 0.5,
                               device: str = 'cpu'):
        """
        Interpret molecules using MCTS to find important substructures.
        
        Args:
            smiles_list: List of SMILES strings or a single SMILES string to interpret
            model_paths: List of model checkpoint paths (uses default if None)
            n_rollout: Number of MCTS rollouts per molecule
            max_atoms: Maximum atoms in rationale
            prop_delta: Minimum property value for rationale
            device: Device to run models on
            
        Returns:
            String description of the interpretation results for all molecules
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        if model_paths is None:
            model_paths = ['/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/final_model/fold_0/model_0/model.pt']
        
        try:
            # Load models once for all molecules
            models = self.load_models(model_paths, device)
            
            # Process each molecule
            all_results = []
            for i, smiles in enumerate(smiles_list):
                print(f"Processing molecule {i+1}/{len(smiles_list)}: {smiles}")
                result = self._interpret_single_molecule_mcts(
                    smiles, models, n_rollout, max_atoms, prop_delta
                )
                all_results.append(result)
            
            # Combine all results
            final_output = f"MCTS Interpretation Results for {len(smiles_list)} molecules:\n"
            final_output += "=" * 60 + "\n\n"
            
            for i, result in enumerate(all_results):
                final_output += f"Molecule {i+1}:\n"
                final_output += result + "\n"
                if i < len(all_results) - 1:  # Add separator except for last molecule
                    final_output += "-" * 40 + "\n\n"
            
            return final_output
            
        except Exception as e:
            return f"Error in MCTS interpretation: {e}"
    
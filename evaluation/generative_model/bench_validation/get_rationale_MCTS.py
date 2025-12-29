import pandas as pd
import torch
from rdkit import Chem
import chemprop
import chemprop.data as data
import os
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import math
from typing import Callable, Union, Iterable, List, Tuple, Set, Dict
from dataclasses import dataclass, field
import numpy as np
import argparse
import re


#functions/ classes
@dataclass
class MCTSNode:
    """Represents a node in a Monte Carlo Tree Search.

    Parameters
    ----------
    smiles : str
        The SMILES for the substructure at this node.
    atoms : list
        A list of atom indices in the substructure at this node.
    W : float
        The total action value, which indicates how likely the deletion will lead to a good rationale.
    N : int
        The visit count, which indicates how many times this node has been visited. It is used to balance exploration and exploitation.
    P : float
        The predicted property score of the new subgraphs' after the deletion, shown as R in the original paper.
    """

    smiles: str
    atoms: Iterable[int]
    W: float = 0
    N: int = 0
    P: float = 0
    children: List['MCTSNode'] = field(default_factory=list)


    def __post_init__(self):
        self.atoms = set(self.atoms)

    def Q(self) -> float:
        """
        Returns
        -------
        float
            The mean action value of the node.
        """
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int, c_puct: float = 10.0) -> float:
        """
        Parameters
        ----------
        n : int
            The sum of the visit count of this node's siblings.
        c_puct : float
            A constant that controls the level of exploration.
        
        Returns
        -------
        float
            The exploration value of the node.
        """
        return c_puct * self.P * math.sqrt(n) / (1 + self.N)

def extract_subgraph_from_mol(mol: Chem.Mol, selected_atoms: Set[int]) -> Tuple[Chem.Mol, List[int]]:
    """Extracts a subgraph from an RDKit molecule given a set of atom indices.

    Parameters
    ----------
    mol : RDKit molecule
        The molecule from which to extract a subgraph.
    selected_atoms : list of int
        The indices of atoms which form the subgraph to be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - RDKit molecule: The subgraph.
        - list of int: Root atom indices from the selected indices.
    """

    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [
            bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]
        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms
            and bond.GetEndAtom().GetIdx() in selected_atoms
        ]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [
        atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms
    ]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots
def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[str, List[int]]:
    """Extracts a subgraph from a SMILES given a set of atom indices.

    Parameters
    ----------
    smiles : str
        The SMILES string from which to extract a subgraph.
    selected_atoms : list of int
        The indices of atoms which form the subgraph to be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - str: SMILES representing the subgraph.
        - list of int: Root atom indices from the selected indices.
    """
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
    try:
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)
    except Exception:
        subgraph = None

    mol = Chem.MolFromSmiles(smiles)  # de-kekulize
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    # If fails, try without kekulization
    subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)

    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), roots
    else:
        return None, None
def mcts_rollout(
    node: MCTSNode,
    state_map: Dict[str, MCTSNode],
    orig_smiles: str,
    clusters: List[Set[int]],
    atom_cls: List[Set[int]],
    nei_cls: List[Set[int]],
    scoring_function: Callable[[List[str]], List[float]],
    min_atoms: int,
    c_puct: float = 10.0,
) -> float:
    """A Monte Carlo Tree Search rollout from a given MCTSNode.

    Parameters
    ----------
    node : MCTSNode
        The MCTSNode from which to begin the rollout.
    state_map : dict
        A mapping from SMILES to MCTSNode.
    orig_smiles : str
        The original SMILES of the molecule.
    clusters : list
        Clusters of atoms.
    atom_cls : list
        Atom indices in the clusters.
    nei_cls : list
        Neighboring cluster indices.
    scoring_function : function
        A function for scoring subgraph SMILES using a Chemprop model.
    min_atoms : int
        The minimum number of atoms in a subgraph.
    c_puct : float
        The constant controlling the level of exploration.

    Returns
    -------
    float
        The score of this MCTS rollout.
    """
    # Return if the number of atoms is less than the minimum
    cur_atoms = node.atoms
    if len(cur_atoms) <= min_atoms:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        # Cluster indices whose all atoms are present in current subgraph
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])

        for i in cur_cls:
            # Leaf atoms are atoms that are only involved in one cluster.
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]

            # This checks
            # 1. If there is only one neighbor cluster in the current subgraph (so that we don't produce unconnected graphs), or
            # 2. If the cluster has only two atoms and the current subgraph has only one leaf atom.
            # If either of the conditions is met, remove the leaf atoms in the current cluster.
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves
        scores = []
        for x in node.children: 
            scores.append(scoring_function([x.smiles]))
            #scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score

    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count, c_puct=c_puct))
    v = mcts_rollout(
        selected_node,
        state_map,
        orig_smiles,
        clusters,
        atom_cls,
        nei_cls,
        scoring_function,
        min_atoms=min_atoms,
        c_puct=c_puct,
    )
    selected_node.W += v
    selected_node.N += 1

    return v


def make_prediction(
    model,
    smiles: List[str],
) -> np.ndarray:
    """Makes predictions on a list of SMILES.

    Parameters
    ----------
    model : list
        A model to make predictions with.
    smiles : list
        A SMILES to make predictions on.

    Returns
    -------
    list[list[float]]
       
    """


    test_data = [data.MoleculeDatapoint(smiles, features_generator = ["rdkit_2d_normalized"])]
    test_dset = data.MoleculeDataset(test_data)
    test_loader = data.MoleculeDataLoader(
        test_dset, batch_size=1, num_workers=0, shuffle=False
    )

    

    with torch.inference_mode():
        
        predss = chemprop.train.predict(model[0], test_loader)
        predss = torch.tensor(predss)
        preds = torch.cat(tuple(predss), dim=0)
        preds = preds.numpy()



    return preds

def mcts(
    smiles: str,
    scoring_function: Callable[[List[str]], List[float]],
    n_rollout: int,
    max_atoms: int,
    prop_delta: float,
    min_atoms: int,
    c_puct: int = 10,
) -> List[MCTSNode]:
    """Runs the Monte Carlo Tree Search algorithm.

    Parameters
    ----------
    smiles : str
        The SMILES of the molecule to perform the search on.
    scoring_function : function
        A function for scoring subgraph SMILES using a Chemprop model.
    n_rollout : int
        The number of MCTS rollouts to perform.
    max_atoms : int
        The maximum number of atoms allowed in an extracted rationale.
    prop_delta : float
        The minimum required property value for a satisfactory rationale.
    min_atoms : int
        The minimum number of atoms in a subgraph.
    c_puct : float
        The constant controlling the level of exploration.

    Returns
    -------
    list
        A list of rationales each represented by a MCTSNode.
    """

    mol = Chem.MolFromSmiles(smiles)

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(
            root,
            state_map,
            smiles,
            clusters,
            atom_cls,
            nei_cls,
            scoring_function,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

    rationales = [
        node
        for _, node in state_map.items()
        if len(node.atoms) <= max_atoms and node.P >= prop_delta
    ]

    return rationales


def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
 

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls






def scoring_function(smiles: List[str]) -> List[float]:

    return make_prediction(
        model=model,
        smiles=smiles,
    )[0]




###### here's the crucial code that runs the rationale search 

def run_rationale_search(smiles:str,
                         rollout:int,
                         c_puct:int, 
                         max_atoms:int,
                         min_atoms:int,
                         prop_delta:int,
                         num_rationales_to_keep:int,
                         ):

    property_for_interpretation= "inhibition"

    results_df = {"smiles": [], property_for_interpretation: []}
    
    for i in range(num_rationales_to_keep):
        results_df[f"rationale_{i}"] = []
        results_df[f"rationale_{i}_score"] = []
    
  
    score = scoring_function([smiles])
    if score > prop_delta:
        rationales = mcts(
            smiles=smiles,
            scoring_function=scoring_function,
            n_rollout=rollout,
            max_atoms=max_atoms,
            prop_delta=prop_delta,
            min_atoms=min_atoms,
            c_puct=c_puct,
        
        )
    else:
        rationales = []
    rationales_smiles = []
    results_df["smiles"].append(smiles)
    results_df[property_for_interpretation].append(score)
    #rationales = []
    if len(rationales) == 0:
        for i in range(num_rationales_to_keep):
            results_df[f"rationale_{i}"].append(None)
            results_df[f"rationale_{i}_score"].append(None)
    else:
        rats = sorted(rationales, key=lambda x: x.P, reverse=True)

        for i in range(num_rationales_to_keep):
            if i < len(rats):
                results_df[f"rationale_{i}"].append(rats[i].smiles)
                rationales_smiles.append(rats[i].smiles) 
                results_df[f"rationale_{i}_score"].append(rats[i].P)
            else:
                results_df[f"rationale_{i}"].append(None)
                results_df[f"rationale_{i}_score"].append(None)

    
    
    to_save = pd.DataFrame(results_df)
    to_save.to_csv(save_dir+"molecule_rationale.csv",
                   index = False)

    rationales_smiles = [re.sub(r':[0-9]+', '', x) for x in rationales_smiles]
    
    if rationales_smiles!=[]:
        print("The top rationales are: " + str(set(rationales_smiles[:3])))
    else:
        print("There are no rationales found") 

    np.save(save_dir+"rationale_list.npy",
            np.array(rationales_smiles))

    return rationales_smiles


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--smiles_dir', type=str,
                        help = "The directory of seed smiles")

    args = parser.parse_args()

    # Get smiles
    smiles_df = pd.read_csv(args.smiles_dir)
    smiles = list(smiles_df['SMILES'])[0]

    # This finds rationales for molecules based on MCTS. This is adapted from chemprop 
    ### variables
    model_path = "/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output/fold_0/model_0/model.pt"
    save_dir = "/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/pipeline_prediction/" 
    # smiles = "CCCc1ccc(Oc2ccc(C(=O)N3C4CCC3CC(N(C)C)C4)cc2)cc1" 

    ### params for rationale search

    rollout = 5 # number of MCTS rollouts to perform. 
    c_puct = 10.0  # constant that controls the level of exploration
    max_atoms = 22  # maximum number of atoms allowed in an extracted rationale
    min_atoms = 8  # minimum number of atoms in an extracted rationale
    prop_delta = 0.5 # Minimum score to count as positive.
    num_rationales_to_keep = 10

    # In this algorithm, if the predicted property from the substructure if larger than prop_delta, the substructure is considered satisfactory
    ### Load the model 

    if torch.cuda.is_available():
        mpnn= chemprop.utils.load_checkpoint(model_path)
    else:
        mpnn= chemprop.utils.load_checkpoint(model_path, device = torch.device('cpu'))
    model = [mpnn]
    ### 

    run_rationale_search(smiles = smiles,
                         rollout = rollout,
                         c_puct = c_puct, 
                         max_atoms = max_atoms, 
                         min_atoms = min_atoms, 
                         prop_delta = prop_delta, 
                         num_rationales_to_keep = num_rationales_to_keep)



# model_path = "model_0/model.pt"
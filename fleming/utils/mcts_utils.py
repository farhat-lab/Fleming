import math
from typing import Callable, Dict, List, Set, Tuple
import numpy as np
from rdkit import Chem

# MCTS constants
MIN_ATOMS = 15
C_PUCT = 10

class MCTSNode:
    """A simplified MCTS node for molecular interpretation."""
    def __init__(self, smiles: str, atoms: List[int], W: float = 0, N: int = 0, P: float = 0) -> None:
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int) -> float:
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)


def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
    """Finds clusters within the molecule."""
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
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


def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[str, List[int]]:
    """Extracts a subgraph from a SMILES given a set of atom indices."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
            
        selected_atoms = set(selected_atoms)
        roots = []
        
        for idx in selected_atoms:
            atom = mol.GetAtomWithIdx(idx)
            bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
            if len(bad_neis) > 0:
                roots.append(idx)

        new_mol = Chem.RWMol(mol)
        for atom_idx in roots:
            if atom_idx < new_mol.GetNumAtoms():
                atom = new_mol.GetAtomWithIdx(atom_idx)
                atom.SetAtomMapNum(1)

        remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
        remove_atoms = sorted(remove_atoms, reverse=True)
        
        for atom in remove_atoms:
            if atom < new_mol.GetNumAtoms():
                new_mol.RemoveAtom(atom)

        result_mol = new_mol.GetMol()
        if result_mol is None:
            return None, None
            
        result_smiles = Chem.MolToSmiles(result_mol)
        return result_smiles, roots
        
    except Exception as e:
        print(f"Error extracting subgraph: {e}")
        return None, None


def prepare_mcts_structures(mol: Chem.Mol):
    """Prepare data structures needed for MCTS."""
    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
        
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])
    
    return clusters, atom_cls, nei_cls 
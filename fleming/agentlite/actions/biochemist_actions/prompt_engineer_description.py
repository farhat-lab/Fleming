from rdkit.Chem import Descriptors
from rdkit.Chem import rdqueries
import numpy as np
import pandas as pd
import requests
import pubchempy as pcp
from rdkit.Chem import rdqueries

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
import time
import glob
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs
from rdkit.Chem import AllChem
from STOUT import translate_forward

## use canonnical, no stereochemistry. 
smiles =["CCCc1ccc(CN2CCc3cc(CCCN4CCN(C)CC4)sc3C2)cc1", "CNCC(C)COC(CNCC(C)C)Cc1noc(CCOc2ccc(C(C)C)cc2)n1", "CN1CCCCC1CNCCc1coc(CCCC2CCCCC2)c1", "CC(C)Cc1ccc(CCCc2nc(CC(C)N3CCCC3CCCCOCCN3CCCC3)no2)cc1"]

#### important inputs 




case_example_no_iupac = """
Example Prompt, chain of thought, and output.  
User Input: You are an expert medicinal chemist. I will give you the smiles string and an LLM generated IUPAC name. This IUPAC Name may or may not be correct. Use it as a starting point, but audit that it's consistent with other data. I will also give you the ring indices, which treats fused rings as a single ring. To help you tease out which atoms are in which rings, I will provide  the individual ring indices for every single ring, and a description of which rings are fusd together.  I will lastly give you a list of features. In your analysis, make sure that the description is consistent with the IUPAC name and all other information. If there's a discrepancy, trust deterministic information (ring indices, aromatic indices, features).  You will be tasked to describe the molecule including its core and what relevant side groups it may have in two sentences.  Think carefully about the structure and go step by step.

Now I will give you an example input and I will walk you through how to get the output. 
Here is the smiles: COC1=C2C(=CC(=C1N3C[C@@H]4CCCN[C@@H]4C3)F)C(=O)C(=CN2C5CC5)C(=O)O Here is a predicted IUPAC name: '7-[(4aS,7aS)-1,2,3,4,4a,5,7,7a-octahydropyrrolo[3,4-b]pyridin-6-yl]-1-cyclopropyl-6-fluoro-8-methoxy-4-oxoquinoline-3-carboxylicacid'. This may or may not be correct, it's from a LLM that's nondeterministic, but it could be a good starting point. \nHere are the ring indices with their atom identity: [(\'ring 1\', {(14, \'N\'), (9, \'C\'), (8, \'N\'), (16, \'C\'), (11, \'C\'), (12, \'C\'), (13, \'C\'), (15, \'C\'), (10, \'C\')}), (\'ring 2\', {(4, \'C\'), (22, \'N\'), (21, \'C\'), (18, \'C\'), (7, \'C\'), (2, \'C\'), (3, \'C\'), (6, \'C\'), (20, \'C\'), (5, \'C\')}), (\'ring 3\', {(23, \'C\'), (24, \'C\'), (25, \'C\')})]. Note that these include both fused and unfused rings.\nHere are the specifically fused rings: [{\'fused_rings\': (\'ring 1\', \'ring 4\'), \'ring 1_atoms\': [(2, \'C\'), (7, \'C\'), (6, \'C\'), (5, \'C\'), (4, \'C\'), (3, \'C\')], \'ring 4_atoms\': [(18, \'C\'), (20, \'C\'), (21, \'C\'), (22, \'N\'), (3, \'C\'), (4, \'C\')]}, {\'fused_rings\': (\'ring 2\', \'ring 3\'), \'ring 2_atoms\': [(9, \'C\'), (8, \'N\'), (16, \'C\'), (15, \'C\'), (10, \'C\')], \'ring 3_atoms\': [(11, \'C\'), (12, \'C\'), (13, \'C\'), (14, \'N\'), (15, \'C\'), (10, \'C\')]}]\nhere are the indices for each individual ring: [(\'ring 1\', [(2, \'C\'), (7, \'C\'), (6, \'C\'), (5, \'C\'), (4, \'C\'), (3, \'C\')]), (\'ring 2\', [(9, \'C\'), (8, \'N\'), (16, \'C\'), (15, \'C\'), (10, \'C\')]), (\'ring 3\', [(11, \'C\'), (12, \'C\'), (13, \'C\'), (14, \'N\'), (15, \'C\'), (10, \'C\')]), (\'ring 4\', [(18, \'C\'), (20, \'C\'), (21, \'C\'), (22, \'N\'), (3, \'C\'), (4, \'C\')]), (\'ring 5\', [(24, \'C\'), (23, \'C\'), (25, \'C\')])]\nHere are the aromatic indices: [(2, \'C\'), (3, \'C\'), (4, \'C\'), (5, \'C\'), (6, \'C\'), (7, \'C\'), (18, \'C\'), (20, \'C\'), (21, \'C\'), (22, \'N\')]\\nHere are the features: {\'Molecular Weight\': 401.43800000000016, \'Number of Hydrogen Bond Donors\': 2, \'Number of Hydrogen Bond Acceptors\': 6, \'Number of Aromatic Rings\': 2, \'Number of Aliphatic Rings\': 3, \'Molecular Formula\': \'C21H24FN3O4\', \'Number of Heterocycles\': 2, \'Number of Amides Bonds\': 0, \'Number of Aromatic Carbocycles\': 1, \'Number of Aromatic Heterocycles\': 1, \'Number of Bridgehead Atoms (atoms shared between two rings that share at least two bonds)\': 0, \'Number of spiro atoms (atoms shared between that share exactly one atom)\': 0, \'Number of Atoms\': 29, \'Number of Bonds\': 33}} 




Chain of Thought:

Broadly, the thought process is as follows: first gain an understanding of the functional groups and their relationships from the IUPAC name. Then, the ring indices, aromatic indices, will give you a better understanding of how these groups are linked to each other. Please synthesize information from both IUPAC names and the indices in creating your descriptor. 
Start with the IUPAC name. I can see a variety of functional groups from this already. For instance, is see that ther'es a cyclopropyl, an octahydropyrrolo, and an oxoquinoline group. I also can see the connctivity of the groups referring to IUPAC rules. I will apply these IUPAC rules to generate a representation of the molecule. 
Now consider the rings using the ring indices. Find their indices and their atom identities and figure out what rings they are. Cross reference with the aromaticty data to figure out what is aromatic and what is not. Cross reference with the "individual rings" data to see the boundaries of the fused rings. The fused rings data also specifies the individual rings that are fused. Identify the atoms in each part of the fused ring. Here, we identify three distinct rings.  
Ring 1: 9 atoms - C, C, C, N, C, C, C, N, C
Ring 2: 10 atoms - C, C, C, N, C, C, C, C, C, N
Ring 3: 3 atoms - C, C, C

Fused rings: all rings are part of fused systems. We see ring1 and ring 4 are fused,and rings 2 and 3 are fused. 

Then, I check the indices of aromatic groups using the aromatic indices. I find that the indices of ring 1 (8-16 are all not aromatic). 
Ring 1 (9 atoms) is not aromatic and is nine membered. Checking again with the smiles and the known information, we identify it as diazobicyclononane.Then, moving on to ring 2: it is on half of the fused ring and not aromatic for hte other half. This corresponds to a quinolone core. The third ring is 3 membered, and can be identified as cyclopropane. I check that this is all consistent with the representation from the IUPAC name. I resolve any discrepancies. 



Now, synthesize again  the IUPAC name  with  the indices, to figure out the connectivity and relationships betweeen rings and funtional groups. Review this carefully and analyze what functionalities are present and where they are. Confirm that the IUPAC, smiles, ring indices, and features are all aligned. for instance, I check that the molecular weight matches, and that the number of rings matches. 


Now, based on the understanding, synthsize this information into the understanding of what core structure and substituents are present. 
Before outputting, Assess the molecular representation generated against the original SMILES, ring information,  information, and all given data to verify accuracy. Ensure no overlap in atom functional assignments. for instance, the stated diazabicyclononane  core corresponds to the carbons 4-12, as provided. The halogen is at position 14, as provided.  Check that the atoms in the predicted represntation in the rings match the atoms in the ring indices.   If it fails, start over. Use this chain of thought, as well as your own imagination to accomplish this task. 



System Output : This is a quinolone with a cyclopropyl substitution and a fluoro substition. It also has a diazabicyclononane substituent and a methoxy substituent.
"""

case_example_iupac = """Example Prompt and output.  
User Input: You are an expert medicinal chemist and it is crucial that you get this correct. In one to two sentences, including its core and what relevant side groups it may have. Think carefully about the structure and go step by step internally:. 
IUPAC name: 7-[(4aS,7aS)-1,2,3,4,4a,5,7,7a-octahydropyrrolo[3,4-b]pyridin-6-yl]-1-cyclopropyl-6-fluoro-8-methoxy-4-oxoquinoline-3-carboxylic acid
SMILES: "COC1=C2C(=CC(=C1N3C[C@@H]4CCCN[C@@H]4C3)F)C(=O)C(=CN2C5CC5)C(=O)O"

System Output: This is a quinolone with a cyclopropyl substitution and a fluoro substition. It also has a diazabicyclononane substituent and a methoxy substituent.

"""


### functions for non-iupac prompting

def smiles_to_smarts(smiles_string):
    try:
        # Create a molecule from the SMILES string
        molecule = Chem.MolFromSmiles(smiles_string)
        if molecule is None:
            raise ValueError("Invalid SMILES string provided.")

        # Convert the molecule to a SMARTS string
        smarts_string = Chem.MolToSmarts(molecule)
        return smarts_string
    except Exception as e:
        # Handle exceptions and return informative error message
        return str(e)

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=False)
def convert_smiles_to_inchi(smiles):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Convert molecule to InChI
    inchi_string = Chem.inchi.MolToInchi(mol)
    
    # Convert InChI to InChIKey
    inchi_key = Chem.inchi.InchiToInchiKey(inchi_string)
    
    return inchi_key, inchi_string



def get_single_ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    
    # This will store both the atom identities and their labels
    ring_details = []
    
    for ring_index, ring in enumerate(atom_rings):
        ring_label = f"ring {ring_index + 1}"
        ring_atoms = []
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_identity = (atom_idx, atom.GetSymbol())
            ring_atoms.append(atom_identity)
        ring_details.append((ring_label, ring_atoms))
        
    return ring_details

def smiles_to_features(smiles):
    """Convert a SMILES string to an adjacency matrix using RDKit.
    
    Args:
        smiles (str): A SMILES string representing a molecule.
        
    Returns:
        np.ndarray: A 2D NumPy array representing the adjacency matrix of the molecule.
    """
    # Convert SMILES to RDKit Molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Get the number of atoms in the molecule
    properties = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "Number of Hydrogen Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "Number of Hydrogen Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Number of Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Number of Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Number of Heterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        "Number of Amides Bonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "Number of Aromatic Carbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        "Number of Aromatic Heterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        "Number of Bridgehead Atoms (atoms shared between two rings that share at least two bonds)": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "Number of spiro atoms (atoms shared between that share exactly one atom)": rdMolDescriptors.CalcNumSpiroAtoms(mol), 
        "Number of Atoms": mol.GetNumAtoms(),
        "Number of Bonds": mol.GetNumBonds(),
}
        
    return properties

def get_fused_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    
    # List to store details of fused rings with their fusion partners
    fused_ring_details = []
    
    # Convert atom_rings to a list of sets for easier intersection checking
    ring_sets = [set(ring) for ring in atom_rings]

    # Identify fused rings by checking intersections and store pairs
    fused_pairs = []

    for i, ring_i in enumerate(ring_sets):
        for j in range(i + 1, len(ring_sets)):
            ring_j = ring_sets[j]
            # Check if there's a shared atom, indicating fusion
            if ring_i.intersection(ring_j):
                fused_pairs.append((i, j))

    # Collect details of the fused rings
    for pair in fused_pairs:
        ring1, ring2 = pair
        ring1_label = f"ring {ring1 + 1}"
        ring2_label = f"ring {ring2 + 1}"
        
        # Gather atom details for each fused ring
        ring1_atoms = [(atom_idx, mol.GetAtomWithIdx(atom_idx).GetSymbol()) for atom_idx in atom_rings[ring1]]
        ring2_atoms = [(atom_idx, mol.GetAtomWithIdx(atom_idx).GetSymbol()) for atom_idx in atom_rings[ring2]]
        
        fused_ring_details.append({
            'fused_rings': (ring1_label, ring2_label),
            f'{ring1_label}_atoms': ring1_atoms,
            f'{ring2_label}_atoms': ring2_atoms
        })
        
    return fused_ring_details
def get_ring_information(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    systems = []
    counter = 1
    for ring in atom_rings:
        ring_atoms_with_identity = [(atom_idx, mol.GetAtomWithIdx(atom_idx).GetSymbol()) for atom_idx in ring]
        ringAts = set(ring_atoms_with_identity)
        nSystems = []
        for system in systems:
            nInCommon = len(set(atom for atom, _ in ringAts).intersection(atom for atom, _ in system))
            if nInCommon and (nInCommon > 1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
                counter+=1
        nSystems.append(ringAts)
        
        systems = nSystems
        
    labeled_systems = []
    for i, system in enumerate(systems, start=1):
        labeled_systems.append((f"ring {i}", system))
    
    return labeled_systems
def get_aromaticity_indices(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    
    is_aromatic_query = rdqueries.IsAromaticQueryAtom()
    
    # List to store indices and identities of aromatic atoms
    aromatic_atoms_info = []
    
    # Iterate over the atoms in the molecule
    for atom in mol.GetAtoms():
        # Match the query to check if the atom is aromatic
        if is_aromatic_query.Match(atom):
            # Get the atom index and symbol
            atom_info = (atom.GetIdx(), atom.GetSymbol())
            # Append the atom index and identity to the list if it's aromatic
            aromatic_atoms_info.append(atom_info)

    return aromatic_atoms_info

def smiles_to_adjacency_matrix(smiles):
    """Convert a SMILES string to an adjacency matrix using RDKit.
    
    Args:
        smiles (str): A SMILES string representing a molecule.
        
    Returns:
        np.ndarray: A 2D NumPy array representing the adjacency matrix of the molecule.
    """
    # Convert SMILES to RDKit Molecule object
    molecule = Chem.MolFromSmiles(smiles)
    
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Get the number of atoms in the molecule
    num_atoms = molecule.GetNumAtoms()
    
    # Initialize a zero matrix of dimensions num_atoms x num_atoms
    adjacency_matrix= Chem.rdmolops.GetAdjacencyMatrix(molecule)
        
    return adjacency_matrix

def smiles_to_iupac(smiles):
    CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"

    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text
def prompt_generator(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Convert molecule to InChI
    inchi_string = Chem.inchi.MolToInchi(mol)
    
    # Convert InChI to InChIKey
    inchi_key = Chem.inchi.InchiToInchiKey(inchi_string)
    iupac_success = False
    
    try:
        iupac = smiles_to_iupac(inchi_key) 
        prompt = get_prompt_iupac(smiles, iupac)
        iupac_success = True
    except:
        try: 
           
            iupac = smiles_to_iupac(inchi_key) 
        except:
            s=pcp.get_compounds(smiles,'smiles')
            
            if s[0].iupac_name != None:
                prompt = get_prompt_iupac(smiles, s[0].iupac_name)
                print('from PCP')
            else: 

                print('No IUPAC found, getting molecule features') 
                prompt=get_prompt_no_iupac(smiles)
    return prompt, iupac_success

# note, use canonical smiles with no stereochemistry. 
def get_prompt_iupac(smiles, iupac): 
    specific_prompt = f"""User input: You are an expert medicinal chemist and it is crucial that you get this correct. In one to two sentences, including its core and what relevant side groups it may have. Think carefully about the structure and go step by step internally:
IUPAC name: {iupac}
SMILES: {smiles} """
    
    
    prompt = case_example_iupac+"\n" + specific_prompt
    return prompt


def get_prompt_no_iupac(smiles): 



    overall_ring_indices = get_ring_information(smiles)
    smarts = smiles_to_smarts(smiles) 
    features= smiles_to_features(smiles)
    individual_rings = get_single_ring(smiles)
    aromatic_indices = get_aromaticity_indices(smiles)
    fused_only = get_fused_rings(smiles)
    IUPAC = translate_forward(smiles)
    inchi= convert_smiles_to_inchi(smiles)[1] 
    specific_prompt = f"""
User Input: You are an expert medicinal chemist. I will give you the smiles string and an LLM generated IUPAC name. This IUPAC Name may or may not be correct. Use it as a starting point, but audit that it's consistent with other data. I will also give you the ring indices, which treats fused rings as a single ring. To help you tease out which atoms are in which rings, I will provide  the individual ring indices for every single ring, and a description of which rings are fusd together.  I will lastly give you a list of features. In your analysis, make sure that the description is consistent with the IUPAC name and all other information. If there's a discrepancy, trust deterministic information (ring indices, aromatic indices, features).  You will be tasked to describe the molecule including its core and what relevant side groups it may have in two sentences.  Think carefully about the structure and go step by step.

Here is the smiles: {smiles}
Here is a predicted IUPAC name: {IUPAC}. This may or may not be correct, it's from a LLM that's nondeterministic, but it could be a good starting point. 

Here are the ring indices with their atom identity: {overall_ring_indices}. Note that these include both fused and unfused rings.
Here are the indices for specifically fused rings: {fused_only}
here are the indices for each individual ring: {individual_rings}
Here are the aromatic indices: {aromatic_indices}
Here are the features: {features} 
Execute the internal chain of thought outlined above, but only output the final description. 

    """
    
    prompt = case_example_no_iupac+"\n" + specific_prompt
    return prompt

if __name__ == "__main__":
    for c in smiles: 
        
        print(prompt_generator(c))
####
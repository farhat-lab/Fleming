from utils.utils import *

class MolecularEditorUtils(Basic_Agent):
    
    


    def is_valid_smiles(self, SMILES: str):
        """
        """
        mol = Chem.MolFromSmiles(SMILES)

        if mol is None:
            return False
        
        if mol.GetNumAtoms() == 0:
            return False
        
        try:
            sanitized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            if not sanitized_smiles:
                return False
            sanitized_mol = Chem.MolFromSmiles(sanitized_smiles)
            if sanitized_mol is None:
                return False
        except Exception as e:
            return False
        return True
    
    def get_iupac(self, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        try:
            [a.SetAtomMapNum(0) for a in molecule.GetAtoms()]
            Chem.RemoveStereochemistry(molecule)
            smiles_no_stereo = "<BASE>"+Chem.MolToSmiles(molecule, canonical=True, kekuleSmiles=True)
            return converter.smiles_to_iupac(smiles_no_stereo)
        except Exception as e:
            ()
            return f"Error getting IUPAC name for {smiles}: {e}"

    def get_rdkit_documentation(self, description_of_task: str):
        """
        """
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma

        persist_dir = "rdkit_docs/rdkit_db"  
        embeddings = OpenAIEmbeddings(openai_api_key=self.config['openai_api_key'])  # Or whatever you used for creation

        # This is all you need to reload:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            )

        results = db.similarity_search(description_of_task, k=4)
        return results 

from utils.utils import *
from utils.admet_utils import ADMETUtils
from utils.inhibition_utils import InhibitionUtils
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
try:
    from synspace import synspace
except Exception as exc:
    synspace = None
    print(f"Warning: synspace is not available; sibling generation via synspace disabled. ({exc})")

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings
except Exception as exc:
    Chroma = None
    OpenAIEmbeddings = None
    print(f"Warning: langchain_community is not available; RDKit documentation search disabled. ({exc})")

'''
Score dictionary for ranking siblings
'''
rank_weights = {
    'rank_inhibition':1,
    'rank_admet':3,
    'rank_cytotox':1  # Added cytotoxicity rank weight
}

CYTOTOX_MODEL_CONTEXT = (
    "Model evidence context: cytotoxicity predictions are Chemprop model outputs with uncertainty estimates. Treat "
    "them as screening priors for HepG2, HSKMC, and IMR90 liability rather than definitive safety evidence. A low-risk "
    "prediction with high uncertainty, sparse training coverage, or literature toxicity for close analogs should be "
    "adjusted toward caution."
)

class OptimizationUtils(Basic_Agent):
    def __init__(self, config, agent_name: str = "Optimization Utils", gradio: bool = False, prompts: dict = None,
                 admet_utils=None, inhibition_utils=None):
        super().__init__(config, agent_name, gradio, prompts)
        # Use provided utilities or create new ones if not provided
        self.admet_utils = admet_utils if admet_utils is not None else ADMETUtils(config, "ADMET Utils", gradio, prompts)
        self.inhibition_utils = inhibition_utils if inhibition_utils is not None else InhibitionUtils(config, "Inhibition Utils", gradio, prompts)
        self.use_synspace = False

    def set_use_synspace(self, use_synspace: bool):
        self.use_synspace = use_synspace
        print(f"OptimizationUtils: set use_synspace to {self.use_synspace}")

    def generate_siblings_llm(self, smiles: str):
        print(f"Generating siblings for {smiles} using LLM...")
        
        # Get description
        description = ""
        gen = self.describe_molecule([smiles])
        try:
            while True:
                next(gen)
        except StopIteration as e:
            description = e.value
            
        # Clean description if it contains "Molecule 1 (SMILES): " prefix
        if f"({smiles}):" in description:
             parts = description.split(f"({smiles}):", 1)
             if len(parts) > 1:
                description = parts[1].strip()

        prompt = f"""
You are an expert medicinal chemist. I have a molecule with SMILES: {smiles}
Description: {description}

Please generate 20 chemical neighbors of this molecule.
Make small optimization changes that shouldn't make it less synthesizable.
Focus on bioisosteric replacements, scaffold hopping, and side chain modifications that are common in medicinal chemistry.
Return ONLY a python list of SMILES strings. Do not include any other text.
Format: ["SMILES1", "SMILES2", ...]
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_infer(messages)
        
        # Extract list from response
        import ast
        import re
        try:
            # Remove markdown code blocks if present
            cleaned_response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
            cleaned_response = re.sub(r'```', '', cleaned_response)
            
            # Find list bracket
            start = cleaned_response.find('[')
            end = cleaned_response.rfind(']') + 1
            if start != -1 and end != -1:
                list_str = cleaned_response[start:end]
                siblings = ast.literal_eval(list_str)
                if isinstance(siblings, list):
                    return [s for s in siblings if isinstance(s, str)]
        except Exception as e:
            print(f"Error parsing LLM response for siblings: {e}")
            print(f"Response was: {response}")
            
        return []
    
    @staticmethod
    def generate_siblings_SMILES(smiles:str):
        if synspace is None:
            raise RuntimeError("synspace is not installed; cannot generate siblings via synspace.")
        mols, sibling_props = synspace.chemical_space(smiles, num_samples = 150)
        sibling_smiles = [Chem.MolToSmiles(item) for item in mols]
        return sibling_smiles
    
    @staticmethod
    def filter_sibling_inhibition(prediction_df:pd.DataFrame,
                                inhibition_threshold:float = 0.2,
                                confidence_threshold:float = 0.1,):
        '''
        Function to filter sibling inhibition prediction based on inhibition and confidence threshold, 
        only keep predicted inhibitory molecules with high confidence as inhibitory siblings
        '''
        
        prediction_df['confidence'] = 1 - prediction_df['inhibition_dirichlet_uncal_uncertainty']
        prediction_df = prediction_df.drop(columns = ['inhibition_dirichlet_uncal_uncertainty'])

        high_inhibition_high_confidence = prediction_df[(prediction_df['inhibition'] > inhibition_threshold) 
                                                        & (prediction_df['confidence'] > confidence_threshold)]
        high_inhibition_low_confidence = prediction_df[(prediction_df['inhibition'] > inhibition_threshold) 
                                                    & (prediction_df['confidence'] < confidence_threshold)]
        low_inhibition_high_confidence = prediction_df[(prediction_df['inhibition'] < inhibition_threshold) 
                                                    & (prediction_df['confidence'] > confidence_threshold)]
        low_inhibition_low_confidence = prediction_df[(prediction_df['inhibition'] < inhibition_threshold) 
                                                    & (prediction_df['confidence'] < confidence_threshold)]


        high_inhibition_high_confidence = high_inhibition_high_confidence.sort_values(by = ['inhibition', 'confidence'],
                                                                                    ascending = False)
        
        return high_inhibition_high_confidence
    
    @staticmethod
    def sibling_QC(siblings_list:list):
        print("Running Sibling QC ...")
        qualified_smiles = []
        for smile in siblings_list:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    Descriptors.qed(mol) 
                    qualified_smiles.append(smile)
                else:
                    print(smile, "Fails QC (invalid SMILES) and removes")
            except Exception as e:
                print(smile, f"Fails QC ({e}) and removes")
        print(len(qualified_smiles), "Qualified Siblings")
        return qualified_smiles
    
    @staticmethod
    def combined_results_from_filters(inhibition_df:pd.DataFrame,
                                    admet_df:pd.DataFrame,):
        
        '''
        Function to combine inhibition and ADMET results and rank the siblings
        '''

        ## Reading and merging dataframes
        combined_df = inhibition_df.merge(admet_df, on=['SMILES'], how = 'inner')

        output_columns = ['SMILES','inhibition','confidence','fail_absorption','fail_distribution',
                        'fail_metabolism','fail_excretion','fail_toxicity','fail_sum','absorption_admet_score',
                        'distribution_admet_score','metabolism_admet_score', 'excretion_admet_score', 'toxicity_admet_score']
        
        ## Ensure all output columns exist, add missing ones with default values (e.g., NaN or 0)
        for col in output_columns:
            if col not in combined_df.columns:
                # Choose a sensible default, e.g., np.nan for scores/metrics, 0 for counts if appropriate
                if 'score' in col or 'inhibition' in col or 'confidence' in col:
                    combined_df[col] = np.nan 
                else:
                    combined_df[col] = 0 
        combined_df = combined_df[output_columns] # Re-select to ensure order and presence

        ## Ranking
        combined_df['rank_inhibition'] = combined_df['inhibition'].rank(ascending=False)
        combined_df['admet_score'] = combined_df['absorption_admet_score'] + combined_df['distribution_admet_score'] + combined_df['metabolism_admet_score'] \
                                    + combined_df['excretion_admet_score'] + combined_df['toxicity_admet_score']
        combined_df['rank_admet'] = combined_df['admet_score'].rank(ascending=False)

        ranks = ['rank_inhibition','rank_admet']

        combined_df['overall_rank'] = 0
        for category in ranks:
            if category in rank_weights:
                combined_df['overall_rank'] += rank_weights[category] * combined_df[category]
        combined_df['overall_rank'] = combined_df['overall_rank'].rank()

        combined_df = combined_df.sort_values(by=['overall_rank'],ascending=True)

        return combined_df

    def get_candidates(self, smiles_list) -> str:
        """
        Generate sibling molecules for a list of SMILES, filter them by inhibition &
        confidence, score them by ADMET+inhibition, then return the top 0 as a markdown table
        with cytotoxicity predictions.
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
        
        Returns:
            String with markdown table of candidates for all input molecules including cytotoxicity
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        all_results = []
        
        for mol_idx, smiles in enumerate(smiles_list):
            result = f"\n## Optimization Results for Molecule {mol_idx + 1}: {smiles}\n\n"
            
            # 1. Generate and QC siblings
            if self.use_synspace:
                siblings = OptimizationUtils.generate_siblings_SMILES(smiles)
            else:
                siblings = self.generate_siblings_llm(smiles)
            
            if siblings is None or not siblings:
                result += "Error: No siblings generated\n"
                all_results.append(result)
                continue
            
            siblings = OptimizationUtils.sibling_QC(siblings)
            if not siblings:
                result += "Error: No siblings passed QC\n"
                all_results.append(result)
                continue

            # 2. Predict inhibition
            preds, unc = self.inhibition_utils.get_chemprop_prediction(siblings)
            inh_df = pd.DataFrame({
                'SMILES': siblings,
                'inhibition': preds,
                'inhibition_dirichlet_uncal_uncertainty': unc
            })

            # 3. Filter by high inhibition & confidence
            # Commenting out inhibition/confidence filtering
            # inh_filtered = OptimizationUtils.filter_sibling_inhibition(inh_df)
            # if inh_filtered.empty:
            #     result += "Error: No candidates after inhibition filtering.\n"
            #     all_results.append(result)
            #     continue

            # Instead, use all siblings after QC
            inh_filtered = inh_df.copy()

            # 4. Get binarized ADMET predictions for the filtered set
            admet_df = self.admet_utils.get_binarized_prediction(list(inh_filtered['SMILES']))
            if admet_df.empty and not inh_filtered.empty:
                print(f"Warning: ADMET prediction yielded empty results for {len(inh_filtered)} candidates.")

            # 5. Get cytotoxicity predictions for the filtered set
            try:
                cytotox_smiles = list(inh_filtered['SMILES'])
                model_list = self.load_models(model_paths=[
                    '/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hepg2_tox/fold_0/model_0/model.pt',
                    '/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/hskmc_tox/fold_0/model_0/model.pt',
                    '/n/data1/hms/dbmi/farhat/anz226/FlemingDev/model_checkpoints/cytotox_models/imr90_tox/fold_0/model_0/model.pt'
                ], device='cpu')
                
                cytotox_data = []
                for smi in cytotox_smiles:
                    preds_list = []
                    uncertainties_list = []
                    for model in model_list:
                        preds, uncertainties = self.make_prediction([model], [smi])
                        preds_list.append(preds[0] if isinstance(preds, (list, np.ndarray)) else preds)
                        uncertainties_list.append(uncertainties[0] if isinstance(uncertainties, (list, np.ndarray)) else uncertainties)
                    
                    cytotox_data.append({
                        'SMILES': smi,
                        'hepatocyte_tox': preds_list[0],
                        'hepatocyte_tox_unc': uncertainties_list[0],
                        'muscle_tox': preds_list[1], 
                        'muscle_tox_unc': uncertainties_list[1],
                        'lung_tox': preds_list[2],
                        'lung_tox_unc': uncertainties_list[2],
                        'avg_cytotox': np.mean(preds_list)  # Add average cytotoxicity
                    })
                
                cytotox_df = pd.DataFrame(cytotox_data)
            except Exception as e:
                print(f"Warning: Cytotoxicity prediction failed: {e}")
                # Create empty cytotox dataframe with NaN values
                cytotox_df = pd.DataFrame({
                    'SMILES': list(inh_filtered['SMILES']),
                    'hepatocyte_tox': np.nan,
                    'hepatocyte_tox_unc': np.nan,
                    'muscle_tox': np.nan,
                    'muscle_tox_unc': np.nan,
                    'lung_tox': np.nan,
                    'lung_tox_unc': np.nan,
                    'avg_cytotox': np.nan
                })

            # 6. Combine inhibition & ADMET scores and rank
            combined_df = OptimizationUtils.combined_results_from_filters(inh_filtered, admet_df) 
            if combined_df.empty:
                result += "Error: No candidates after combining inhibition and ADMET results.\n"
                all_results.append(result)
                continue

            # 7. Merge with cytotoxicity data
            combined_df = combined_df.merge(cytotox_df, on='SMILES', how='left')

            # 7.5. Add cytotoxicity ranking (lower avg_cytotox is better)
            combined_df['rank_cytotox'] = combined_df['avg_cytotox'].rank(ascending=True)

            # 8. Take top 40 after new ranking
            # Recompute overall_rank with cytotox
            combined_df['overall_rank'] = 0
            for category in ['rank_inhibition', 'rank_admet', 'rank_cytotox']:
                if category in rank_weights:
                    combined_df['overall_rank'] += rank_weights[category] * combined_df[category]
            combined_df['overall_rank'] = combined_df['overall_rank'].rank()
            combined_df = combined_df.sort_values(by=['overall_rank'], ascending=True)

            top_candidates = combined_df.head(40).copy()
            if top_candidates.empty:
                result += "No candidates found in top 20 after ranking.\n"
                all_results.append(result)
                continue

            # 9. Load threshold-type mapping
            _, _, threshold_type = self.admet_utils.get_admet_threshold(
                'model_training/current_antibiotic_admet_predictions.csv'
            )

            # 10. Define ADMET task→group mapping
            absorption_props = ['Bioavailability_Ma', 'HIA_Hou', 'PAMPA_NCATS',
                                'Pgp_Broccatelli', 'Caco2_Wang', 'Lipophilicity_AstraZeneca',
                                'Solubility_AqSolDB']
            distribution_props = ['BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo']
            metabolism_props = ['CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith',
                                'CYP2D6_Veith', 'CYP3A4_Veith']
            excretion_props = ['Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',
                            'Half_Life_Obach']
            toxicity_props = ['AMES', 'Carcinogens_Lagunin', 'ClinTox', 'DILI',
                            'NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase',
                            'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma', 'SR-ARE',
                            'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
                            'Skin_Reaction', 'hERG', 'LD50_Zhu']
            group_map = {}
            for p in absorption_props: group_map[p] = 'Absorption'
            for p in distribution_props: group_map[p] = 'Distribution'
            for p in metabolism_props: group_map[p] = 'Metabolism'
            for p in excretion_props: group_map[p] = 'Excretion'
            for p in toxicity_props: group_map[p] = 'Toxicity'

            # 11. Build the markdown table with cytotoxicity
            lines = []
            
            lines.append("| SMILES | Tasks Passed | Tasks Failed | Inhibition | Uncertainty | Hepatocyte Tox | Muscle Tox | Lung Tox |")
            lines.append("|--------|--------------|--------------|------------|-------------|----------------|------------|----------|")
            
            # Add original molecule row
            orig_preds, orig_unc = self.inhibition_utils.get_chemprop_prediction(smiles=[smiles])
            orig_admet_df = self.admet_utils.get_binarized_prediction([smiles])
            
            # Get original cytotoxicity
            try:
                orig_cytotox_data = []
                for model in model_list:
                    preds, uncertainties = self.make_prediction([model], [smiles])
                    orig_cytotox_data.append(preds[0] if isinstance(preds, (list, np.ndarray)) else preds)
                orig_hep_tox = f"{orig_cytotox_data[0]:.3f}"
                orig_muscle_tox = f"{orig_cytotox_data[1]:.3f}"
                orig_lung_tox = f"{orig_cytotox_data[2]:.3f}"
            except:
                orig_hep_tox = "N/A"
                orig_muscle_tox = "N/A" 
                orig_lung_tox = "N/A"
            
            if orig_admet_df.empty:
                passed_str = "ADMET N/A"
                failed_str = "ADMET N/A"
            else:
                orig_admet = orig_admet_df.iloc[0]
                passed, failed = [], []
                for prop, group in group_map.items():
                    if prop not in orig_admet: continue
                    score = orig_admet[prop]
                    better = 'Lower Better' if threshold_type.get(prop) == 'lower' else 'Higher Better'
                    flag_col = f'pass_{prop}'
                    if flag_col not in orig_admet: continue
                    flag = orig_admet[flag_col]
                    entry = f"{prop}, {group}, {score:.3f}, {better}"
                    if flag == 1: passed.append(entry)
                    else: failed.append(entry)
                passed_str = "; ".join(passed) if passed else "None"
                failed_str = "; ".join(failed) if failed else "None"
            
            lines.append(f"| Original ({smiles}) | {passed_str} | {failed_str} | {orig_preds[0]:.3f} | {orig_unc[0]:.3f} | {orig_hep_tox} | {orig_muscle_tox} | {orig_lung_tox} |")

            # Add candidate molecules
            for _, row in top_candidates.iterrows():
                smi = row['SMILES']
                passed, failed = [], []
                df_row_series = admet_df[admet_df['SMILES'] == smi].iloc[0] if not admet_df[admet_df['SMILES'] == smi].empty else None
                
                if df_row_series is not None:
                    for prop, group in group_map.items():
                        if prop not in df_row_series: continue
                        score = df_row_series[prop]
                        better = 'Lower Better' if threshold_type.get(prop) == 'lower' else 'Higher Better'
                        flag_col = f'pass_{prop}'
                        if flag_col not in df_row_series: continue
                        flag = df_row_series[flag_col]
                        entry = f"{prop}, {group}, {score:.3f}, {better}"
                        if flag == 1: passed.append(entry)
                        else: failed.append(entry)
                    passed_str_cand = "; ".join(passed) if passed else "None"
                    failed_str_cand = "; ".join(failed) if failed else "None"
                else:
                    passed_str_cand = "ADMET N/A"
                    failed_str_cand = "ADMET N/A"

                passed_str = "; ".join(passed) if passed else "None"
                failed_str = "; ".join(failed) if failed else "None"
                
                inh_row = inh_df[inh_df['SMILES'] == smi].iloc[0]
                inhibition = inh_row['inhibition']
                uncertainty = inh_row['inhibition_dirichlet_uncal_uncertainty']

                # Get cytotoxicity values
                hep_tox = f"{row['hepatocyte_tox']:.3f}" if not pd.isna(row['hepatocyte_tox']) else "N/A"
                muscle_tox = f"{row['muscle_tox']:.3f}" if not pd.isna(row['muscle_tox']) else "N/A"
                lung_tox = f"{row['lung_tox']:.3f}" if not pd.isna(row['lung_tox']) else "N/A"

                lines.append(f"| {smi} | {passed_str} | {failed_str} | {inhibition:.3f} | {uncertainty:.3f} | {hep_tox} | {muscle_tox} | {lung_tox} |")

            result += "\n".join(lines)
            all_results.append(result)

        return "\n\n" + "="*100 + "\n\n".join(all_results)

    def get_admet_prediction_string_summary(self, smiles_list: str):
        """
        Retrieves the ADMET prediction string summary by calling the method 
        from ADMETUtils instance.
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
        """
        return self.admet_utils.get_admet_prediction_string_summary(smiles_list)
    
    def get_admet_prediction_raw_table(self, smiles_list: str):
        """
        Retrieves the ADMET prediction raw table by calling the method 
        from ADMETUtils instance.
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
        """
        return self.admet_utils.get_admet_prediction_raw_table(smiles_list)

    def get_chemprop_prediction_string(self, smiles_list: str):
        """
        Retrieves the ChemProp prediction string by calling the method 
        from InhibitionUtils instance.
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
        """
        return self.inhibition_utils.get_chemprop_prediction_string(smiles_list)

    def get_cytotoxicity_copilot_mode(self, smiles_list):
        """
        Cytotoxicity Prediction for a list of SMILES
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
            
        Returns:
            String with cytotoxicity predictions for all molecules
        """
        return self.admet_utils.get_cytotoxicity_copilot_mode(smiles_list)
    def get_rdkit_documentation(self, description_of_task: str):
        """
        """
        if Chroma is None or OpenAIEmbeddings is None:
            raise RuntimeError("langchain_community is not installed; RDKit documentation search is unavailable.")
        persist_dir = "../rdkit_docs/rdkit_db"  # Must match the one you used when saving
        embeddings = OpenAIEmbeddings(openai_api_key=self.config['openai_api_key'])  # Or whatever you used for creation

        # This is all you need to reload:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            )

        results = db.similarity_search(description_of_task, k=4)
        return results

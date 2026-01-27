import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from utils.admet_utils import ADMETUtils
from utils.inhibition_utils import InhibitionUtils
from utils.optimization_utils import OptimizationUtils
import multiprocessing
from typing import Optional, Tuple, List, Dict, Any

class Batch_Mode_Utils(ADMETUtils, InhibitionUtils):
    """
    Unified pipeline for automatic molecule generation, evaluation, and optimization.
    Tracks molecules through the entire pipeline in memory.
    """
    
    def __init__(self, config, agent_name: str = "Automatic Mode", gradio: bool = False, prompts: dict = None):
        super().__init__(config=config, agent_name=agent_name, gradio=gradio, prompts=prompts)
        self.pipeline_log = []  # Track what we've done
    
    def log_step(self, step_name: str, details: str):
        """Log pipeline steps for user feedback."""
        self.pipeline_log.append(f"{step_name}: {details}")
        print(f"✓ {step_name}: {details}")
    
    def generate_molecules(self, smiles_csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load molecules from a CSV file.
        
        Args:
            smiles_csv_path: Optional path to a CSV file with a 'SMILES' column.

        Returns:
            DataFrame with 'SMILES' column containing molecules from the CSV.
        """
        try:
            if smiles_csv_path:
                if not os.path.exists(smiles_csv_path):
                    raise FileNotFoundError(f"CSV file not found at: {smiles_csv_path}")
                df = pd.read_csv(smiles_csv_path)
                if 'SMILES' not in df.columns:
                    raise ValueError("CSV file must have a 'SMILES' column.")
                
                df = df[['SMILES']].copy()
                df = df.drop_duplicates().reset_index(drop=True)
                self.log_step("Molecule Loading", f"Loaded {len(df)} unique molecules from {os.path.basename(smiles_csv_path)}")
                return df
            else:
                raise ValueError("No smiles_csv_path provided. Molecule generation from scratch is not supported.")
            
        except Exception as e:
            self.log_step("Molecule Loading Error", str(e))
            # Return empty DataFrame with correct structure
            return pd.DataFrame({'SMILES': []})
    
    def add_inhibition_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MTB inhibition predictions to the dataframe."""
        if df.empty or 'SMILES' not in df.columns:
            return df
            
        try:
            smiles_list = df['SMILES'].tolist()
            preds, unc = self.get_chemprop_prediction(smiles_list)
            
            df['inhibition'] = preds
            df['inhibition_confidence'] = unc
            
            self.log_step("Inhibition Prediction", 
                         f"Added inhibition predictions for {len(df)} molecules")
            return df
            
        except Exception as e:
            self.log_step("Inhibition Prediction Error", str(e))
            df['inhibition'] = np.nan
            df['inhibition_confidence'] = np.nan
            return df
    
    def add_admet_properties(self, df: pd.DataFrame, include_detailed: bool = True) -> pd.DataFrame:
        """Add ADMET properties, SAscore, and novelty to the dataframe."""
        if df.empty or 'SMILES' not in df.columns:
            return df
            
        try:
            smiles_list = df['SMILES'].tolist()
            
            # Get ADMET predictions
            admet_df = self.get_binarized_prediction_batch(smiles_list)
            
            # Calculate overall ADMET score if not present
            if 'admet_score' not in admet_df.columns:
                admet_df['admet_score'] = (admet_df['absorption_admet_score'] + 
                                         admet_df['distribution_admet_score'] + 
                                         admet_df['metabolism_admet_score'] + 
                                         admet_df['excretion_admet_score'] + 
                                         admet_df['toxicity_admet_score'])
            
            # Add SAscore
            sa_scores = []
            for smi in tqdm(smiles_list, desc="Calculating SAscores"):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        from sascore.sascore import calculateScore as calculate_sascore
                        sa_scores.append(calculate_sascore(mol))
                    else:
                        sa_scores.append(np.nan)
                except Exception:
                    sa_scores.append(np.nan)
            admet_df['sa_score'] = sa_scores
            
            # Add novelty scores
            novelty_scores = []
            for smi in tqdm(smiles_list, desc="Calculating novelty scores"):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        novelty_scores.append(self.calculate_novelty(mol))
                    else:
                        novelty_scores.append(np.nan)
                except Exception:
                    novelty_scores.append(np.nan)
            admet_df['novelty_score'] = novelty_scores
            
            # Merge with original dataframe
            # Drop columns in df if they already exist in admet_df to avoid conflicts/duplicates during merge
            cols_to_use = admet_df.columns.difference(df.columns)
            # But we need SMILES to merge
            if 'SMILES' not in cols_to_use:
                cols_to_use = cols_to_use.union(['SMILES'])
            
            result_df = pd.merge(df, admet_df[cols_to_use], on='SMILES', how='left')
            
            self.log_step("ADMET Properties", 
                         f"Added ADMET, SAscore, and novelty for {len(result_df)} molecules")
            return result_df
            
        except Exception as e:
            self.log_step("ADMET Properties Error", str(e))
            return df
    
    def rank_and_filter_molecules(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        Rank molecules based on combined score and filter to top N.
        
        Hard filters (SAScore <= 6, Inhibition >= 0.7, Novelty Score >= 0.7) are applied as a primary ranking criterion.
        Molecules passing the filters are ranked higher than those failing.
        
        Ranking formula: overall_rank = rank_inhibition + 3*rank_admet_score + rank_novelty_score + rank_sa_score
        """
        if df.empty:
            return df
            
        try:
            # Apply Hard Filters logic to create a passing flag
            df['passes_hard_filter'] = True
            
            if 'sa_score' in df.columns:
                df['passes_hard_filter'] &= (df['sa_score'] <= 6)
            
            if 'inhibition' in df.columns:
                df['passes_hard_filter'] &= (df['inhibition'] >= 0.7)
                
            if 'novelty_score' in df.columns:
                # Novelty = 1 - similarity. We want similarity <= 0.3, so 1 - novelty <= 0.3 => novelty >= 0.7
                df['passes_hard_filter'] &= (df['novelty_score'] >= 0.7)
            
            passing_count = df['passes_hard_filter'].sum()
            self.log_step("Filtering", f"{passing_count}/{len(df)} molecules passed all hard filters.")

            # Create ranking columns (lower rank = better)
            df['rank_inhibition'] = df['inhibition'].rank(method='min', ascending=False)
            df['rank_admet_score'] = df['admet_score'].rank(method='min', ascending=False) 
            df['rank_novelty_score'] = df['novelty_score'].rank(method='min', ascending=False)
            df['rank_sa_score'] = df['sa_score'].rank(method='min', ascending=True)  # Lower SA is better
            
            # Calculate overall rank (lower is better)
            df['overall_rank'] = (df['rank_inhibition'] + 
                                3 * df['rank_admet_score'] + 
                                df['rank_novelty_score'] + 
                                df['rank_sa_score'])
            
            # Sort by passes_hard_filter (True first/descending), then overall_rank (ascending), and take top N
            df_ranked = df.sort_values(['passes_hard_filter', 'overall_rank'], ascending=[False, True]).head(top_n).reset_index(drop=True)
            
            self.log_step("Ranking & Filtering", 
                         f"Ranked molecules and filtered to top {len(df_ranked)} candidates")
            return df_ranked
            
        except Exception as e:
            self.log_step("Ranking & Filtering Error", str(e))
            return df.head(top_n) if len(df) > top_n else df
    
    def check_plausibility(self, df: pd.DataFrame, describe_mol_function, top_n: int = 100) -> pd.DataFrame:
        """
        Check plausibility using a simple LLM call.
        
        Returns:
            DataFrame with 'plausibility_assessment' and 'passes_plausibility' columns.
            Sorted by passes_plausibility (True first), then by previous rank.
        """
        if df.empty:
            return df
            
        try:
            # Determine which column to use for SMILES
            smiles_col = 'optimized_SMILES' if 'optimized_SMILES' in df.columns else 'SMILES'
            
            # Work on a copy
            result_df = df.copy()
            
            # Ensure optimized_SMILES is filled if we are using it
            if smiles_col == 'optimized_SMILES':
                 result_df[smiles_col] = result_df[smiles_col].fillna(result_df['SMILES'])

            # Identify rows to check (top_n)
            # We assume df is sorted by priority/rank from previous steps
            rows_to_check = result_df.head(top_n)
            
            if rows_to_check.empty:
                 return result_df

            smiles_list = rows_to_check[smiles_col].tolist()
            indices_to_check = rows_to_check.index.tolist()
            
            self.log_step("Plausibility Check Limit", f"Checking plausibility for top {len(rows_to_check)} molecules")

            # Batch describe molecules
            descriptions_str = ""
            try:
                description_generator = describe_mol_function(smiles_list)
                # The generator yields progress, we consume it to get the final return value
                while True:
                    next(description_generator)
            except StopIteration as e:
                descriptions_str = e.value

            # Parse the batch descriptions back into individual descriptions
            individual_descriptions = descriptions_str.split("\n\n")
            
            smiles_to_description = {}
            for i, smi in enumerate(smiles_list):
                if i < len(individual_descriptions):
                    desc = individual_descriptions[i]
                    # Extract just the description part
                    if f"({smi}):" in desc:
                        description = desc.split(f"({smi}): ", 1)[1]
                    else:
                        description = desc
                    smiles_to_description[smi] = description
                else:
                    smiles_to_description[smi] = f"Error: Could not generate description for {smi}"

            # Run LLM Checks
            assessments = {}
            pass_flags = {}
            plausible_count = 0

            for idx in tqdm(indices_to_check, desc="Checking plausibility"):
                row = result_df.loc[idx]
                smiles = row[smiles_col]
                description = smiles_to_description.get(smiles, "Description not available.")
                
                question = (
                    f"Based on available literature, assess the plausibility of the molecule with SMILES {smiles} "
                    f"(Description: {description}) as a novel antibiotic against Tuberculosis. "
                    f"Consider its potential efficacy, novelty, and any known major contraindications or toxicity concerns. "
                    f"Respond with only 'PLAUSIBLE' or 'NOT PLAUSIBLE' followed by detailed reasoning."
                )
                
                conversation = [{"role": "system", "content": "You are a medicinal chemist expert."}, {"role": "user", "content": question}]
                llm_response = self.llm_infer(conversation)

                assessments[idx] = llm_response
                is_plausible = llm_response.strip().upper().startswith("PLAUSIBLE")
                pass_flags[idx] = is_plausible
                if is_plausible:
                    plausible_count += 1
            
            # Update DataFrame
            result_df['plausibility_assessment'] = result_df.index.map(assessments)
            result_df['passes_plausibility'] = result_df.index.map(pass_flags)
            
            self.log_step("Plausibility Check", 
                         f"{plausible_count} plausible out of {len(rows_to_check)} checked.")
            
            # Sort: Passes Plausibility (True) > False > NaN
            sort_cols = ['passes_plausibility']
            sort_orders = [False]
            
            if 'overall_rank' in result_df.columns:
                sort_cols.append('overall_rank')
                sort_orders.append(True)
                
            result_df = result_df.sort_values(sort_cols, ascending=sort_orders)
            
            return result_df
            
        except Exception as e:
            self.log_step("Plausibility Check Error", str(e))
            return df

    def optimize_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize molecules using the rigorous multi-round pipeline.
        
        Logic:
        1. Start with seed molecule.
        2. Generate neighbors (siblings).
        3. Filter neighbors: must have predicted inhibition > 0.7.
        4. Select best neighbor: lowest number of failed ADMET tasks (fail_sum).
        5. If best neighbor has fewer failures than current molecule, update current to best neighbor and repeat.
        6. Continue for up to 5 rounds.
        """
        if df.empty:
            return df
            
        try:
            optimized_data = []
            
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Optimizing molecules (Multi-round)"):
                current_smiles = row['SMILES']
                current_failures = row.get('fail_sum')
                
                # If we don't have fail_sum for the start molecule, calculate it
                if pd.isna(current_failures):
                    temp_preds = self.get_binarized_prediction_batch([current_smiles])
                    if not temp_preds.empty:
                        current_failures = temp_preds.iloc[0]['fail_sum']
                    else:
                        current_failures = 999 # High number if failed
                
                history = [current_smiles]
                
                # Run optimization loop
                for round_num in range(5):
                    # 1. Generate siblings
                    siblings = OptimizationUtils.generate_siblings_SMILES(current_smiles)
                    siblings = OptimizationUtils.sibling_QC(siblings)
                    
                    if not siblings:
                        break
                        
                    # 2. Predict Inhibition
                    inh_preds, _ = self.get_chemprop_prediction(siblings)
                    
                    # Create temporary DF for filtering
                    candidates = pd.DataFrame({'SMILES': siblings, 'inhibition': inh_preds})
                    
                    # 3. Filter by inhibition >= 0.7
                    passing_candidates = candidates[candidates['inhibition'] >= 0.7].copy()
                    
                    if passing_candidates.empty:
                        break
                        
                    # 4. Predict ADMET for passing candidates
                    candidate_smiles = passing_candidates['SMILES'].tolist()
                    admet_preds = self.get_binarized_prediction_batch(candidate_smiles)
                    
                    if admet_preds.empty:
                        break
                        
                    # Merge failures info
                    passing_candidates = pd.merge(passing_candidates, 
                                                admet_preds[['SMILES', 'fail_sum', 'admet_score']], 
                                                on='SMILES', how='left')
                    
                    # 5. Find best neighbor (min fail_sum)
                    # Break ties with higher inhibition or ADMET score if needed, but spec says "fails fewest"
                    best_candidate = passing_candidates.sort_values('fail_sum', ascending=True).iloc[0]
                    best_failures = best_candidate['fail_sum']
                    best_smiles = best_candidate['SMILES']
                    
                    # 6. Check if improvement
                    if best_failures < current_failures:
                        current_smiles = best_smiles
                        current_failures = best_failures
                        history.append(current_smiles)
                    else:
                        # No improvement in failures
                        break
                
                row_dict = row.to_dict()
                
                # Always record the final molecule in optimized_SMILES (best found or original)
                row_dict['optimized_SMILES'] = current_smiles
                
                # Only record optimization metadata if we actually changed the molecule
                if len(history) > 1:
                    row_dict['optimization_rounds'] = len(history) - 1
                    row_dict['optimized_fail_sum'] = current_failures
                    row_dict['is_optimized'] = True
                else:
                    row_dict['is_optimized'] = False
                    
                optimized_data.append(row_dict)
            
            result_df = pd.DataFrame(optimized_data)
            
            optimized_count = result_df['is_optimized'].sum() if 'is_optimized' in result_df.columns else 0
            self.log_step("Optimization", 
                         f"Successfully optimized {optimized_count}/{len(result_df)} molecules")
            
            return result_df
            
        except Exception as e:
            self.log_step("Optimization Error", str(e))
            return df
    
    def run_complete_pipeline_progressive(self, 
                                          smiles_csv_path: Optional[str] = None,
                                          describe_mol_function=None,
                                          output_file: Optional[str] = None,
                                          steps_to_run: Optional[List[str]] = None,
                                          top_n_to_optimize: int = 100,
                                          top_n_to_plausibility: int = 100):
        """
        Run the complete automatic mode pipeline and yield results progressively.
        
        Args:
            steps_to_run: List of steps to run. Options: ['ranking', 'optimization', 'plausibility']
                          'ranking' implies loading -> inhibition -> admet -> rank
            top_n_to_optimize: Number of molecules to pass to optimization step (after ranking).
            top_n_to_plausibility: Number of molecules to pass to plausibility check (after optimization).
        """
        self.pipeline_log = []  # Reset log
        
        # Default steps if not provided
        if steps_to_run is None:
            steps_to_run = ['ranking', 'optimization', 'plausibility']
            
        # Normalize steps
        steps_to_run = [s.lower() for s in steps_to_run]
        
        def save_and_summarize(df, step_suffix=None, final=False):
            if output_file and not df.empty:
                # If a step suffix is provided, save a separate file
                if step_suffix:
                    base, ext = os.path.splitext(output_file)
                    step_file = f"{base}_{step_suffix}{ext}"
                    df.to_csv(step_file, index=False)
                    self.log_step("Step Results Saved", f"Saved to {os.path.basename(step_file)}")
                    # Yield download for step file immediately
                    return df, self._generate_summary_report(df), step_file
                
                # Also save to main output file (overwriting current state)
                df.to_csv(output_file, index=False)
                self.log_step("Results Saved", f"Intermediate results saved to {output_file}")
            
            summary = self._generate_summary_report(df)
            return df, summary, None

        try:
            # Always Step 1: Generate/Load molecules
            df = self.generate_molecules(smiles_csv_path=smiles_csv_path)
            if df.empty:
                yield pd.DataFrame(), "Pipeline failed: No molecules generated"
                return
            
            # Step 1: Loaded - yield update but don't save file
            summary = self._generate_summary_report(df)
            yield df, summary
            
            # Step set: Ranking (Inhibition -> ADMET -> Rank)
            if 'ranking' in steps_to_run:
                # Inhibition
                df = self.add_inhibition_predictions(df)
                summary = self._generate_summary_report(df)
                yield df, summary
                
                # ADMET
                df = self.add_admet_properties(df)
                summary = self._generate_summary_report(df)
                yield df, summary
                
                # Rank - use top_n_to_optimize
                df = self.rank_and_filter_molecules(df, top_n_to_optimize)
                # Save and download after RANKING
                res = save_and_summarize(df, step_suffix="ranking_results")
                df, summary, step_file = res
                yield df, summary
                if step_file: yield 'download', step_file
            
            # Step set: Optimization
            if 'optimization' in steps_to_run:
                df = self.optimize_molecules(df)
                # Save and download after OPTIMIZATION
                res = save_and_summarize(df, step_suffix="optimization_results")
                df, summary, step_file = res
                yield df, summary
                if step_file: yield 'download', step_file
            
            # Step set: Plausibility
            if 'plausibility' in steps_to_run:
                if describe_mol_function is None:
                    self.log_step("Plausibility Check Skipped", "Missing required functions")
                else:
                    # check_plausibility handles the top_n logic internally via arg
                    df = self.check_plausibility(df, describe_mol_function, top_n=top_n_to_plausibility)
                # Save and download after PLAUSIBILITY
                res = save_and_summarize(df, step_suffix="plausibility_results")
                df, summary, step_file = res
                yield df, summary
                if step_file: yield 'download', step_file
            
            # Final yield with final summary (download handled via final=True inside save_and_summarize but we manually handle download below)
            res = save_and_summarize(df, final=True)
            df, summary, _ = res
            yield df, summary
            
            # After final summary, save and yield download link for the FINAL output
            if output_file and not df.empty:
                df.to_csv(output_file, index=False)
                self.log_step("Downloadable Results", f"Final results saved to {output_file}")
                yield 'download', output_file

        except Exception as e:
            error_msg = f"Pipeline error: {e}"
            self.log_step("Pipeline Error", error_msg)
            yield pd.DataFrame(), error_msg

    def run_complete_pipeline(self, 
                            steps_to_run: Optional[List[str]] = None,
                            top_n_to_optimize: int = 100,
                            top_n_to_plausibility: int = 100,
                            describe_mol_function = None,
                            output_file: Optional[str] = None,
                            smiles_csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """
        Run the complete automatic mode pipeline (non-progressive version).
        """
        # For simplicity, just wrap the progressive one and return final result
        generator = self.run_complete_pipeline_progressive(
            smiles_csv_path=smiles_csv_path,
            describe_mol_function=describe_mol_function,
            output_file=output_file,
            steps_to_run=steps_to_run,
            top_n_to_optimize=top_n_to_optimize,
            top_n_to_plausibility=top_n_to_plausibility
        )
        
        final_df = pd.DataFrame()
        final_summary = ""
        
        try:
            for result in generator:
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], pd.DataFrame):
                    final_df, final_summary = result
        except Exception as e:
             return pd.DataFrame(), f"Error running pipeline: {e}"
             
        return final_df, final_summary
    
    def _generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate a summary report of the pipeline results."""
        if df.empty:
            return "Pipeline completed but no molecules remain after processing."
        
        summary_lines = [
            "=== AUTOMATIC MODE PIPELINE SUMMARY ===",
            "",
            f"Final Results: {len(df)} molecules",
            "",
            "Pipeline Steps Completed:",
        ]
        
        for step in self.pipeline_log:
            summary_lines.append(f"  • {step}")
        
        summary_lines.extend([
            "",
            "Top 5 Molecules by Overall Rank:",
        ])
        
        # Determine which smiles column to display if optimized
        display_col = 'optimized_SMILES' if 'optimized_SMILES' in df.columns and df['optimized_SMILES'].notna().any() else 'SMILES'
        
        # Sort by rank if available, otherwise just head
        if 'overall_rank' in df.columns:
            top_molecules = df.sort_values('overall_rank').head(5)
        else:
            top_molecules = df.head(5)

        for i, (_, row) in enumerate(top_molecules.iterrows(), 1):
            smi = row[display_col] if pd.notna(row.get(display_col)) else row['SMILES']
            
            inhibition_val = row.get('inhibition')
            inhibition_str = f"{inhibition_val:.3f}" if pd.notna(inhibition_val) else "N/A"

            admet_val = row.get('admet_score')
            admet_str = f"{admet_val:.2f}" if pd.notna(admet_val) else "N/A"

            rank_val = row.get('overall_rank')
            rank_str = f"{rank_val:.1f}" if pd.notna(rank_val) else "N/A"
            
            fail_sum_val = row.get('fail_sum')
            fail_sum_str = f"{fail_sum_val}" if pd.notna(fail_sum_val) else "N/A"
            
            opt_rounds = row.get('optimization_rounds')
            opt_str = f" (Opt rounds: {opt_rounds})" if pd.notna(opt_rounds) and opt_rounds > 0 else ""

            plausibility = row.get('passes_plausibility')
            plaus_str = ""
            if pd.notna(plausibility):
                plaus_str = f", Plausible: {'Yes' if plausibility else 'No'}"

            summary_lines.append(
                f"  {i}. {smi}{opt_str} "
                f"(Inhibition: {inhibition_str}, "
                f"ADMET Score: {admet_str}, "
                f"Fail Sum: {fail_sum_str}, "
                f"Rank: {rank_str}"
                f"{plaus_str})"
            )
        
        if 'optimized_SMILES' in df.columns:
            optimized_count = df['optimized_SMILES'].notna().sum()
            summary_lines.append(f"\nOptimized molecules: {optimized_count}/{len(df)}")
        
        return "\n".join(summary_lines)

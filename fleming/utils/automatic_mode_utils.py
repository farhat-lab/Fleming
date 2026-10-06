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
            try:
                from sascore.sascore import calculateScore as calculate_sascore
            except ImportError:
                calculate_sascore = None

            sa_scores = []
            for smi in tqdm(smiles_list, desc="Calculating SAscores"):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol and calculate_sascore is not None:
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
    
    def _compute_ranking_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute (or refresh) the hard-filter flag and ranking columns for every row in df.

        Hard filters (SAScore <= 6, Inhibition >= 0.7, Novelty Score >= 0.7) are applied as a primary ranking criterion.
        Molecules passing the filters are ranked higher than those failing.

        Ranking formula: overall_rank = rank_inhibition + 3*rank_admet_score + rank_novelty_score + rank_sa_score

        Does not sort or truncate the dataframe; callers decide ordering/limits.
        """
        # Apply Hard Filters logic to create a passing flag
        df['passes_hard_filter'] = True

        if 'sa_score' in df.columns:
            df['passes_hard_filter'] &= (df['sa_score'] <= 6)

        if 'inhibition' in df.columns:
            df['passes_hard_filter'] &= (df['inhibition'] >= 0.7)

        if 'novelty_score' in df.columns:
            # Novelty = 1 - similarity. We want similarity <= 0.3, so 1 - novelty <= 0.3 => novelty >= 0.7
            df['passes_hard_filter'] &= (df['novelty_score'] >= 0.7)

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

        return df

    def rank_and_filter_molecules(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        Rank molecules based on combined score and filter to top N.
        """
        if df.empty:
            return df

        try:
            df = self._compute_ranking_columns(df)

            passing_count = df['passes_hard_filter'].sum()
            self.log_step("Filtering", f"{passing_count}/{len(df)} molecules passed all hard filters.")

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

        Each row represents a single molecule (either an original starting molecule or an
        optimized analog, see 'parent_molecule'/'is_optimized' columns), so plausibility is
        assessed once per row/SMILES.

        Returns:
            DataFrame with 'plausibility_assessment'/'passes_plausibility' columns added.
            Sorted by passes_plausibility (True first), then by previous rank.
        """
        if df.empty:
            return df
            
        try:
            # Work on a copy
            result_df = df.copy()
            result_df['plausibility_assessment'] = np.nan
            result_df['passes_plausibility'] = np.nan

            # Only review molecules that passed the hard filters when that signal is available.
            if 'passes_hard_filter' in result_df.columns:
                eligible_df = result_df[result_df['passes_hard_filter'] == True]
                self.log_step(
                    "Plausibility Eligibility",
                    f"{len(eligible_df)}/{len(result_df)} molecules eligible after hard-filter gating.",
                )
            else:
                eligible_df = result_df
                self.log_step(
                    "Plausibility Eligibility",
                    "passes_hard_filter not available; using the current ranked set.",
                )

            # Identify rows to check (top_n)
            # We assume df is sorted by priority/rank from previous steps
            rows_to_check = eligible_df.head(top_n)
            
            if rows_to_check.empty:
                 self.log_step("Plausibility Check Skipped", "No hard-filter-passing molecules available for review.")
                 return result_df

            # Dedupe SMILES so we don't ask the LLM to review the same structure twice
            # (e.g. an optimization round that looped back to an already-seen molecule).
            row_smiles = rows_to_check['SMILES']
            unique_smiles = list(dict.fromkeys(row_smiles.tolist()))
            indices_to_check = list(rows_to_check.index)

            self.log_step(
                "Plausibility Check Limit",
                f"Checking plausibility for {len(indices_to_check)} molecules "
                f"= {len(unique_smiles)} unique SMILES to review"
            )

            # Batch describe molecules (deduplicated)
            descriptions_str = ""
            try:
                description_generator = describe_mol_function(unique_smiles)
                # The generator yields progress, we consume it to get the final return value
                while True:
                    next(description_generator)
            except StopIteration as e:
                descriptions_str = e.value

            # Parse the batch descriptions back into individual descriptions
            individual_descriptions = descriptions_str.split("\n\n")
            
            smiles_to_description = {}
            for i, smi in enumerate(unique_smiles):
                if self.stop_requested():
                    print("\033[1;33;40mPlausibility: stop requested during description.\033[0m")
                    break
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

            # Run LLM checks once per unique SMILES (cached), reused across rows that share a SMILES.
            smiles_to_assessment = {}
            smiles_to_pass_flag = {}

            for smi in tqdm(unique_smiles, desc="Checking plausibility"):
                if self.stop_requested():
                    print("\033[1;33;40mPlausibility: stop requested - keeping molecules scored so far.\033[0m")
                    break
                description = smiles_to_description.get(smi, "Description not available.")

                question = (
                    f"Based on available literature, assess the plausibility of the molecule with SMILES {smi} "
                    f"(Description: {description}) as a novel antibiotic against Tuberculosis. "
                    f"Consider its potential efficacy, novelty, and any known major contraindications or toxicity concerns. "
                    f"Respond with only 'PLAUSIBLE' or 'NOT PLAUSIBLE' followed by detailed reasoning."
                )

                conversation = [{"role": "system", "content": "You are a medicinal chemist expert."}, {"role": "user", "content": question}]
                llm_response = self.llm_infer(conversation)

                smiles_to_assessment[smi] = llm_response
                smiles_to_pass_flag[smi] = llm_response.strip().upper().startswith("PLAUSIBLE")

            # Populate per-row results from the cached per-SMILES results.
            plausible_row_count = 0
            for idx in indices_to_check:
                smi = result_df.at[idx, 'SMILES']
                assessment = smiles_to_assessment.get(smi)
                is_plausible = smiles_to_pass_flag.get(smi, False)
                result_df.at[idx, 'plausibility_assessment'] = assessment
                result_df.at[idx, 'passes_plausibility'] = is_plausible
                if is_plausible:
                    plausible_row_count += 1

            self.log_step(
                "Plausibility Check",
                f"{plausible_row_count}/{len(indices_to_check)} molecules judged plausible "
                f"({len(unique_smiles)} unique SMILES reviewed)."
            )
            
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

        Each optimized molecule that is found is added as its own row (not merged into its
        parent's row), with 'parent_molecule' set to the original SMILES it was derived from
        and 'is_optimized' set to True. Original starting molecules keep their own row with
        'is_optimized' set to False and 'parent_molecule' left blank.
        """
        if df.empty:
            return df
            
        try:
            new_optimized_rows = []  # rows for the newly-found optimized molecules

            for index, row in tqdm(df.iterrows(), total=len(df), desc="Optimizing molecules (Multi-round)"):
                # Checked per molecule rather than per API call: a batch run is a long
                # loop, and stopping between molecules leaves every completed one intact
                # so the partial results are still usable.
                if self.stop_requested():
                    print(f"\033[1;33;40mBatch stopped at the user's request after "
                          f"{len(new_optimized_rows)} optimized molecules.\033[0m")
                    break

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
                
                # Only record a new row if optimization actually changed the molecule
                if len(history) > 1:
                    new_optimized_rows.append({
                        'SMILES': current_smiles,
                        'parent_molecule': row['SMILES'],
                        'optimization_rounds': len(history) - 1,
                    })

            # Mark the starting molecules as non-optimized, un-parented rows
            original_df = df.copy()
            original_df['parent_molecule'] = np.nan
            original_df['optimization_rounds'] = np.nan
            original_df['is_optimized'] = False

            if not new_optimized_rows:
                self.log_step("Optimization", f"Successfully optimized 0/{len(original_df)} molecules")
                return original_df

            # Compute full inhibition/ADMET/SAscore/novelty metrics for the new optimized
            # molecules so their row stands on its own, rather than inheriting their parent's values.
            # Dedupe by SMILES first (different parents can converge on the same optimized
            # analog) so add_admet_properties' merge-on-SMILES doesn't blow up row counts.
            optimized_df = pd.DataFrame(new_optimized_rows)
            unique_smiles_df = optimized_df[['SMILES']].drop_duplicates().reset_index(drop=True)
            unique_smiles_df = self.add_inhibition_predictions(unique_smiles_df)
            unique_smiles_df = self.add_admet_properties(unique_smiles_df)
            optimized_df = optimized_df.merge(unique_smiles_df, on='SMILES', how='left')
            optimized_df['is_optimized'] = True

            result_df = pd.concat([original_df, optimized_df], ignore_index=True, sort=False)

            # Re-rank across the combined set (originals + optimized analogs) so both are
            # compared on equal footing, and re-sort so downstream steps (e.g. plausibility)
            # still see the best candidates first.
            result_df = self._compute_ranking_columns(result_df)
            result_df = result_df.sort_values(
                ['passes_hard_filter', 'overall_rank'], ascending=[False, True]
            ).reset_index(drop=True)

            optimized_count = len(optimized_df)
            self.log_step(
                "Optimization",
                f"Successfully optimized {optimized_count}/{len(original_df)} molecules "
                f"(added {optimized_count} new row(s) for the optimized analogs)"
            )
            
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
            
                if self.stop_requested():
                    self.log_step("Stopped", "Pipeline stopped at the user's request.")
                    yield df, self._generate_summary_report(df) + \
                        "\n\n**Stopped at your request.** Results above are from the "\
                        "steps that finished."
                    return

            # Step set: Optimization
            if 'optimization' in steps_to_run:
                df = self.optimize_molecules(df)
                # Save and download after OPTIMIZATION
                res = save_and_summarize(df, step_suffix="optimization_results")
                df, summary, step_file = res
                yield df, summary
                if step_file: yield 'download', step_file
            
                if self.stop_requested():
                    self.log_step("Stopped", "Pipeline stopped at the user's request.")
                    yield df, self._generate_summary_report(df) + \
                        "\n\n**Stopped at your request.** Results above are from the "\
                        "steps that finished."
                    return

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
        
        # Sort by rank if available, otherwise just head
        if 'overall_rank' in df.columns:
            top_molecules = df.sort_values('overall_rank').head(5)
        else:
            top_molecules = df.head(5)

        has_parent_col = 'parent_molecule' in df.columns

        for i, (_, row) in enumerate(top_molecules.iterrows(), 1):
            smi = row['SMILES']

            inhibition_val = row.get('inhibition')
            inhibition_str = f"{inhibition_val:.3f}" if pd.notna(inhibition_val) else "N/A"

            admet_val = row.get('admet_score')
            admet_str = f"{admet_val:.2f}" if pd.notna(admet_val) else "N/A"

            rank_val = row.get('overall_rank')
            rank_str = f"{rank_val:.1f}" if pd.notna(rank_val) else "N/A"
            
            fail_sum_val = row.get('fail_sum')
            fail_sum_str = f"{fail_sum_val}" if pd.notna(fail_sum_val) else "N/A"

            opt_rounds = row.get('optimization_rounds')
            parent_smi = row.get('parent_molecule') if has_parent_col else None
            if pd.notna(parent_smi):
                opt_str = f" (optimized from {parent_smi}"
                opt_str += f", rounds: {int(opt_rounds)})" if pd.notna(opt_rounds) else ")"
            else:
                opt_str = ""

            plaus_val = row.get('passes_plausibility')
            plaus_str = f", Plausible: {'Yes' if plaus_val else 'No'}" if pd.notna(plaus_val) else ""

            summary_lines.append(
                f"  {i}. {smi}{opt_str} "
                f"(Inhibition: {inhibition_str}, "
                f"ADMET Score: {admet_str}, "
                f"Fail Sum: {fail_sum_str}, "
                f"Rank: {rank_str}"
                f"{plaus_str})"
            )
        
        if 'is_optimized' in df.columns:
            optimized_count = df['is_optimized'].sum()
            summary_lines.append(f"\nOptimized molecule rows added: {optimized_count}/{len(df)} total rows")
        
        return "\n".join(summary_lines)

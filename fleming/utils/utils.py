import pandas as pd
import re
import os
from openai import OpenAI
#from molecular_describer import prompt_generator
from tqdm import tqdm
import asyncio
import sys
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
import io
import base64
import yaml
import json 
import numpy as np
import fcntl
from admet_ai import ADMETModel
from typing import List
import chemprop.data as data
import chemprop
import torch
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem.Descriptors import qed
import gradio as gr
from gradio import ChatMessage
from utils.pqa_agent import PQA_Agent
sys.path.append("../")
import time
import inspect

import multiprocessing # Added import
from utils.pqa_agent import PQA_Agent

from prompts import *
MODEL='gpt-5.4-mini'
# Add external_tools to path to allow importing sascore and scscore
# This file is in utils/, and external_tools is in utils/external_tools
current_dir = os.path.dirname(os.path.abspath(__file__))
external_tools_path = os.path.join(current_dir, 'external_tools')
if external_tools_path not in sys.path:
    sys.path.append(external_tools_path)

# Import SAscore calculation function
try:
    from sascore.sascore import calculateScore as calculate_sascore # alias to avoid conflict if you have other calculateScore
    SASEngineLoaded = True
except ImportError:
    print("Warning: sascore.py not found or RDKit components missing. SAscore will not be calculated.")
    SASEngineLoaded = False
except FileNotFoundError as e: # Specifically for fpscores.pkl.gz not found by sascore's readFragmentScores
    print(f"Warning: SAscore fragment file missing: {e}. SAscore will not be calculated.")
    SASEngineLoaded = False

try:
    from tdc.single_pred import Tox, ADME
except Exception as exc:
    Tox = None
    ADME = None
    print(f"Warning: PyTDC is not available; TDC-based predictors disabled. ({exc})")

try:
    import pqapi  # noqa: F401
except Exception as exc:
    pqapi = None
    print(f"Warning: pqapi is not available; PQApi features disabled. ({exc})")

try:
    import synspace  # noqa: F401
except Exception as exc:
    synspace = None
    print(f"Warning: synspace is not available; some optimization features may be disabled. ({exc})")

try:
    from chemicalconverters import NamesConverter
except Exception as exc:
    NamesConverter = None
    print(f"Warning: chemicalconverters is not available; name conversion disabled. ({exc})")

try:
    from langchain_experimental.utilities import PythonREPL
except Exception as exc:
    PythonREPL = None
    print(f"Warning: langchain_experimental is not available; falling back to a minimal Python REPL. ({exc})")

if NamesConverter is not None:
    converter = NamesConverter(model_name="knowledgator/SMILES2IUPAC-canonical-base")
else:
    converter = None

if PythonREPL is None:
    class PythonREPL:
        """
        Minimal fallback for `langchain_experimental.utilities.PythonREPL`.
        Executes code in a persistent namespace and captures stdout.
        """
        def __init__(self):
            self.globals = {}
            self.locals = self.globals

        def run(self, command: str) -> str:
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(command, self.globals, self.locals)
            return buf.getvalue()
import dotenv
dotenv.load_dotenv()


# Lazy-loaded novelty engine variables
REF_FPS_FOR_NOVELTY = None
NOVELTY_ENGINE_LOADED = False

class Basic_Agent(PQA_Agent):
    USER_INPUT_SENTINEL = "__USER_INPUT_SENTINEL__"

    def __init__(self, config, agent_name: str, gradio: bool, prompts: dict):
        # config here is the path string, e.g., 'config.yaml'
        self.config_path = config
        self.config = self.load_config(config)
        self.agent_name = agent_name
        self.has_gpu = torch.cuda.is_available()
        # Use self.config (the dictionary) instead of config (the string path)
        self.backend = self.config.get('openai_backend', 'openai')
        self.openai_api_key = self.config['openai_api_key']

        self.pqapi_token = self.config['pqapi_token']
        os.environ['PQA_API_TOKEN'] = self.pqapi_token

        self.prompts = prompts
        if self.prompts is not None:
            if self.agent_name == 'Molecular Generation Agent':
                self.system_prompt = prompts['agent_system_prompt'].format(gpu_access_status=self.has_gpu)
            else:
                self.system_prompt = prompts['agent_system_prompt']
        self.openai_backend = self.config['openai_backend']
        self.role = "You are an expert chemist who can answer questions about molecules"
        print(f"\033[1;91;40m{agent_name} Initialized\033[0m")

        self.gradio = gradio
        self.latest_final_response = ''
        self.python_repl = PythonREPL()
        self.python_repl.locals= self.python_repl.globals
        self.history = [] # Ensure history is initialized
        self.tools_called_in_current_run = []
        self.conversation = [{"role": "system", "content": self.system_prompt if hasattr(self, 'system_prompt') else "System ready."}]

        # For ask_user in Gradio
        self._is_waiting_for_user_input = False
        self._question_for_user_input = None
        self._current_user_query = None  # Store current user query for context

    def _load_novelty_engine(self):
        """Lazy-initialize the novelty engine by loading reference fingerprints."""
        global REF_FPS_FOR_NOVELTY, NOVELTY_ENGINE_LOADED
        if REF_FPS_FOR_NOVELTY is not None:
            return
            
        REF_FPS_FOR_NOVELTY = []
        NOVELTY_ENGINE_LOADED = False
        
        # Resolve paths relative to project root using config or defaults
        try:
            # Try to get project root relative to this file
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
        except NameError:
            # Fallback if __file__ is not defined (e.g. in some REPLs)
            project_root = "."

        # Get paths from config if available, otherwise use defaults
        # Note: self.config is available in Basic_Agent
        novelty_train_data = self.config.get('novelty_train_data', "data/train_data.csv")
        chembl_antibiotics_data = self.config.get('chembl_antibiotics_data', "data/chembl_antibiotics.csv")

        paths_to_try = [
            os.path.join(project_root, novelty_train_data),
            os.path.join(project_root, chembl_antibiotics_data)
        ]
        
        valid_paths = [p for p in paths_to_try if os.path.exists(p)]
        
        if not valid_paths:
            # print(f"Warning: Novelty reference data files not found. Searched: {paths_to_try}. Novelty will not be calculated.")
            return

        try:
            print(f"Loading reference data for novelty calculation from: {valid_paths[0]}")
            ref_df = pd.read_csv(valid_paths[0])
            
            if len(valid_paths) > 1:
                ref_df = pd.concat([ref_df, pd.read_csv(valid_paths[1])])
                
            if 'SMILES' in ref_df.columns:
                ref_smiles_list = ref_df['SMILES'].dropna().tolist()
                for smi in tqdm(ref_smiles_list, desc="Processing reference SMILES for novelty"):
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        REF_FPS_FOR_NOVELTY.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                if REF_FPS_FOR_NOVELTY:
                    NOVELTY_ENGINE_LOADED = True
                    print(f"Successfully processed {len(REF_FPS_FOR_NOVELTY)} reference fingerprints for novelty.")
                else:
                    print("Warning: No valid reference SMILES processed for novelty calculation.")
            else:
                print(f"Warning: 'SMILES' column not found in data. Novelty will not be calculated.")
        except Exception as e:
            print(f"Error loading reference data for novelty: {e}. Novelty will not be calculated.")

    def calculate_novelty(self, mol: Chem.Mol) -> float:
        """
        Calculates the novelty of a molecule based on Tanimoto similarity to reference fingerprints.
        """
        self._load_novelty_engine()
        if mol is None:
            return np.nan

        try:
            hit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            if not REF_FPS_FOR_NOVELTY: 
                return np.nan
            sim_list = [DataStructs.TanimotoSimilarity(hit_fp, ref_fp) for ref_fp in REF_FPS_FOR_NOVELTY]
            if not sim_list: 
                return np.nan 
            max_sim = max(sim_list)
            return 1.0 - max_sim
        except Exception as e:
            return np.nan

    def get_novelty_and_sascore(self, smiles_list):
        """
        Get novelty and SAscore for a list of SMILES
        """
        # Convert single SMILES to list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        all_results = []
        
        scscore_model = None
        # Try dynamic import if not available
        try:
            from scscore.scscore.standalone_model_numpy import SCScorer as SCScorerClass
            scscore_model = SCScorerClass()
            scscore_model.restore()
        except ImportError:
            # Try adding external_tools to path if not there
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ext_tools = os.path.join(current_dir, 'external_tools')
                if ext_tools not in sys.path:
                    sys.path.append(ext_tools)
                from scscore.scscore.standalone_model_numpy import SCScorer as SCScorerClass
                scscore_model = SCScorerClass()
                scscore_model.restore()
            except ImportError:
                pass
        except Exception as e:
            pass # print(f"Warning: Failed to restore SCScorer: {e}")
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            novelty_score = self.calculate_novelty(mol)
            
            sascore = "N/A"
            if 'calculate_sascore' in globals() and calculate_sascore is not None and mol is not None:
                try:
                    sascore = calculate_sascore(mol)
                except Exception:
                    pass
            
            scscore = "N/A"
            if scscore_model is not None:
                try:
                    scscore = scscore_model.get_score_from_smi(smiles)
                except Exception:
                    pass
            
            result = f"Molecule {i+1} ({smiles}): novelty score is {novelty_score} (higher is more novel). The scScore (newest synthesizability score) is {scscore} (lower is more synthesizable, from 1 to 5). The SAScore (old synthesizability score) is {sascore} (lower is more synthesizable (1 to 10))."
            all_results.append(result)
        
        return "\n".join(all_results)

    def set_current_user_query(self, user_query: str):
        """Set the current user query for context in subagent responses"""
        self._current_user_query = user_query

    def ask_user(self, question: str) -> str:
        """Prompt the user to answer a question.
        In Gradio mode, this will set flags and return a sentinel value,
        expecting the main Gradio loop to handle the input.
        In console mode, it uses input().
        """
        if self.gradio:
            self._is_waiting_for_user_input = True
            self._question_for_user_input = question
            # The main Gradio loop will see this sentinel and handle UI.
            return Basic_Agent.USER_INPUT_SENTINEL
        else:
            # Original console input
            print(f"\033[1;33;40m{self.agent_name} asks: {question}\033[0m")
            answer = input("\033[1;34;40mYou: \033[0m")
            return answer

    def FINISHED(self) -> str:
        """
        Tool call to indicate the agent has completed its task and is ready for summarization.
        This replaces the need to say 'finished' or 'fleming finished' in the response text.
        """
        return "TASK_COMPLETED_READY_FOR_SUMMARY"



    def display_smiles(self, smiles: str, label_letter: str = 'A') -> str:
        """
        Tool call to directly request display of a molecule in the UI.
        Returns a structured dict consumed by the app without text parsing.

        Usage: Tool-call: self.display_smiles("CCO", "A")
        Returns: {"type": "display_molecules", "items": [{"smiles": "CCO", "label": "A"}]}
        """
        try:
            # Light validation; let the UI ignore invalid strings gracefully
            if not isinstance(smiles, str) or len(smiles.strip()) == 0:
                return {"type": "display_molecules", "items": []}
            label_letter = (label_letter or 'A')
            return {
                "type": "display_molecules",
                "items": [
                    {"smiles": smiles.strip(), "label": str(label_letter)[:8]}
                ]
            }
        except Exception as e:
            return {"type": "display_molecules", "items": [], "error": str(e)}

    def display_smiles_batch(self, smiles_list, labels=None) -> str:
        """
        Tool call to display multiple SMILES at once.
        Accepts either labels list, a single start-letter, or None.

        Examples:
          Tool-call: self.display_smiles_batch(["CCO","C1=CC=CC=C1"], ["A","B"]) 
          Tool-call: self.display_smiles_batch(["CCO","C1=CC=CC=C1"], "A")
          Tool-call: self.display_smiles_batch(["CCO","C1=CC=CC=C1"]) 
        """
        try:
            if not isinstance(smiles_list, (list, tuple)):
                return {"type": "display_molecules", "items": []}

            # Normalize labels
            generated_labels = []
            if isinstance(labels, (list, tuple)):
                generated_labels = [str(x) for x in labels]
            elif isinstance(labels, str) and len(labels.strip()) == 1:
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                start_letter = labels.strip().upper()
                try:
                    start_index = alphabet.index(start_letter)
                except ValueError:
                    start_index = 0
                for i in range(len(smiles_list)):
                    generated_labels.append(alphabet[(start_index + i) % len(alphabet)])
            else:
                generated_labels = [str(i + 1) for i in range(len(smiles_list))]

            items = []
            for idx, smi in enumerate(smiles_list):
                if not isinstance(smi, str) or len(smi.strip()) == 0:
                    continue
                label = generated_labels[idx] if idx < len(generated_labels) else str(idx + 1)
                items.append({"smiles": smi.strip(), "label": label[:16]})
            return {"type": "display_molecules", "items": items}
        except Exception as e:
            return {"type": "display_molecules", "items": [], "error": str(e)}

    def display_molecules_batch(self, smiles_list, labels=None) -> str:
        """Alias for display_smiles_batch (robustness to naming variations)."""
        return self.display_smiles_batch(smiles_list, labels)

    def display_molecule(self, smiles: str, label_letter: str = 'A') -> str:
        """Alias for display_smiles (robustness to naming variations)."""
        return self.display_smiles(smiles, label_letter)

    def run_parallel_tools(self, tool_calls, max_workers: int = 3):
        """
        Run independent local tool calls concurrently within a subagent.

        Usage:
          Tool-call: self.run_parallel_tools(["self.describe_molecule('CCO')", "self.get_chemprop_prediction_string('CCO')"])

        Each item must be a string containing a self.<tool>(...) call. Do not use this for
        ask_user, FINISHED, or dependent calls where one result is needed to form another call.
        """
        import queue
        import threading

        if isinstance(tool_calls, str):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, (list, tuple)) or len(tool_calls) == 0:
            return "No parallel tool calls were provided."

        max_workers = max(1, min(int(max_workers or 3), 5, len(tool_calls)))
        output_queue = queue.Queue()
        results = [None] * len(tool_calls)
        blocked_tools = {"ask_user", "FINISHED", "run_parallel_tools"}

        def _tool_name(call: str):
            match = re.match(r"\s*self\.(\w+)\s*\(", str(call))
            return match.group(1) if match else None

        def _run_one(idx, call):
            call = str(call).strip()
            name = _tool_name(call)
            if not name:
                results[idx] = f"ERROR: Invalid parallel tool call syntax: {call}"
                output_queue.put(("DONE", idx))
                return
            if name in blocked_tools:
                results[idx] = f"ERROR: {name} cannot be run inside run_parallel_tools."
                output_queue.put(("DONE", idx))
                return

            try:
                if hasattr(self, 'tools_called_in_current_run'):
                    self.tools_called_in_current_run.append(f"parallel:{name}")

                output_queue.put(ChatMessage(role="assistant", content=f"[{self.agent_name} Parallel] Starting {name}"))
                response = eval(call)

                if inspect.isgenerator(response):
                    final_value = None
                    try:
                        while True:
                            intermediate = next(response)
                            output_queue.put(ChatMessage(role="assistant", content=f"[{self.agent_name} Parallel:{name}] {intermediate}"))
                    except StopIteration as stop:
                        final_value = stop.value
                    response = final_value

                if response == Basic_Agent.USER_INPUT_SENTINEL:
                    response = f"ERROR: {name} requested user input, which is not supported in parallel mode."

                results[idx] = response
            except Exception as exc:
                results[idx] = f"ERROR running {name}: {exc}"
            finally:
                output_queue.put(("DONE", idx))

        active_threads = 0
        next_idx = 0

        def _start_next():
            nonlocal active_threads, next_idx
            if next_idx >= len(tool_calls):
                return
            thread = threading.Thread(target=_run_one, args=(next_idx, tool_calls[next_idx]), daemon=True)
            thread.start()
            next_idx += 1
            active_threads += 1

        for _ in range(max_workers):
            _start_next()

        while active_threads > 0:
            item = output_queue.get()
            if isinstance(item, tuple) and item[0] == "DONE":
                active_threads -= 1
                _start_next()
            else:
                yield item

        combined = []
        for idx, (call, result) in enumerate(zip(tool_calls, results), start=1):
            combined.append(f"Parallel tool {idx}: {call}\nResult:\n{result}")
        return "\n\n".join(combined)

    def run_agent_loop(self, molecule: str = None, instruction: str = ""):
        """
        Main execution loop for a sub-agent.
        This function is a generator that yields its internal monologue (thoughts, tool calls, results)
        for real-time display. It yields a final dictionary with type "final" when it's done.
        """
        self.tools_called_in_current_run = [] # Reset for each run
        print(f"\033[1;91;40mRunning Agent: {self.agent_name} for molecule: {molecule}\033[0m")

        if not hasattr(self, 'conversation') or len(self.conversation) <= 1:
            print(f"\033[1;36;40m{self.agent_name} starting fresh conversation for a new task.\033[0m")
            self.conversation = [{"role": "system", "content": self.system_prompt}]
            if self.agent_name == 'Molecular Generation Agent':
                initial_task_prompt = self.prompts['initial_plan_prompt'].format(
                    original_query=self._current_user_query or "No specific user query provided",
                    instruction=instruction
                )
                self.conversation.append({"role": "user", "content": initial_task_prompt})
            else:
                initial_task_prompt = self.prompts['initial_plan_prompt'].format(
                    original_query=self._current_user_query or "No specific user query provided",
                    molecule=molecule, 
                    instruction=instruction
                )
                self.conversation.append({"role": "user", "content": initial_task_prompt})
        else:
            print(f"\033[1;36;40m{self.agent_name} continuing from existing conversation with {len(self.conversation)} messages.\033[0m")
            if self.agent_name == 'Molecular Generation':
                follow_up_prompt = f"New instruction: {instruction}. Please review the conversation history and continue your work based on this new instruction."
            else:
                follow_up_prompt = f"New instruction for molecule {molecule}: {instruction}. Please review the conversation history and continue your work based on this new instruction."
            self.conversation.append({"role": "user", "content": follow_up_prompt})
        
        for _ in range(15):
            response = self.llm_infer(self.conversation, openai_backend=self.openai_backend)
            print(f"\033[91m[{self.agent_name}] {response}\033[0m")
            self.conversation.append({"role": "assistant", "content": response})
            yield ChatMessage(role="assistant", content=f"[{self.agent_name}] {response}")

            tool_call_successful = False
            if 'finished' in response.lower():
                print(f"\033[1;33;40m{self.agent_name} finished. Generating final summary.\033[0m")
                summarization_prompt = self.prompts.get('summarization_prompt', "Please summarize your findings now.")
                self.conversation.append({"role": "user", "content": summarization_prompt})
                
                final_summary = self.llm_infer(self.conversation, reasoning_effort='low')
                self.latest_final_response = final_summary
                # Yield final result as a dictionary instead of returning
                yield {"type": "final", "content": final_summary, "tools_used": self.tools_called_in_current_run}
                return
            if 'tool-call:' in response.lower():
                try:
                    function_response = self.run_function(response)
                    if function_response is not None:
                        tool_call_successful = True

                        if inspect.isgenerator(function_response):
                            final_tool_result = None
                            try:
                                while True:
                                    intermediate_step = next(function_response)
                                    yield ChatMessage(role="assistant", content=f"[{self.agent_name} Progress] {intermediate_step}")
                            except StopIteration as e:
                                final_tool_result = e.value
                            
                            function_response = final_tool_result
                            if function_response is None:
                                continue

                        if function_response == Basic_Agent.USER_INPUT_SENTINEL:
                            # Yield a paused signal instead of returning
                            yield {"type": "paused"}
                            # Continue processing after user input is handled
                            continue
                        elif function_response == "TASK_COMPLETED_READY_FOR_SUMMARY":
                            # FINISHED tool call was used - trigger summarization
                            print(f"\033[1;33;40m{self.agent_name} FINISHED tool call detected. Generating final summary.\033[0m")
                            summarization_prompt = self.prompts.get('summarization_prompt', "Please summarize your findings now.")
                            self.conversation.append({"role": "user", "content": summarization_prompt})
                            
                            final_summary = self.llm_infer(self.conversation, reasoning_effort='low')
                            self.latest_final_response = final_summary
                            # Yield final result as a dictionary instead of returning
                            yield {"type": "final", "content": final_summary, "tools_used": self.tools_called_in_current_run}
                            return
                        elif isinstance(function_response, dict) and function_response.get("type") == "display_molecules":
                            # Pass through structured molecules payload
                            yield function_response
                            # Also record a brief note in conversation for traceability
                            self.conversation.append({"role": "system", "content": f"Displayed {len(function_response.get('items', []))} molecule(s)"})
                            continue
                        
                        # Check if this might be a confirmation response
                        if hasattr(self, '_pending_pqa_operation') or hasattr(self, '_pending_lit_review_operation'):
                            # This could be a confirmation response, but we need the actual user input
                            # The confirmation will be handled when user input is processed
                            pass

                        tool_result = str(function_response)
                        yield ChatMessage(role="assistant", content=f"[{self.agent_name} Tool Result] {tool_result}")
                        self.conversation.append({"role": "system", "content": tool_result})
                        
                except Exception as e:
                    error_msg = f"Error executing tool: {e}"
                    print(f"\033[1;31;40m{error_msg}\033[0m")
                    yield ChatMessage(role="system", content=f"[{self.agent_name} Error] {error_msg}")
                    self.conversation.append({"role": "system", "content": error_msg})
                    replan_prompt = "The previous tool call failed. Please re-evaluate your plan. You can try the tool again with different parameters, or try a different tool."
                    self.conversation.append({"role": "user", "content": replan_prompt})
            
         
        
        print(f"\033[1;33;40m{self.agent_name} reached turn limit. Forcing summary.\033[0m")
        summarization_prompt = self.prompts.get('summarization_prompt', "Please summarize your findings now.")
        self.conversation.append({"role": "user", "content": summarization_prompt})
        final_summary = self.llm_infer(self.conversation, reasoning_effort='low')
        self.latest_final_response = final_summary
        # Yield final result as a dictionary instead of returning
        yield {"type": "final", "content": final_summary, "tools_used": self.tools_called_in_current_run}

    def __call__(self, molecule: str=None, instruction: str = "", use_parent_history: bool = True):
        """
        Makes the agent callable, yielding from its execution loop.
        """
        if molecule != None: 
            yield from self.run_agent_loop(molecule, instruction)
        else:
            yield from self.run_agent_loop(instruction=instruction)

    def load_config(self,config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    
    def pqa_molecular_analysis(self, molecule, question):
        """
        Perform PQA molecular analysis on a single molecule.
        This method can be called directly by agents without user confirmation.
        """
        return self.run_molecule(molecule, question)
    
    def literature_review(self, question):
        """
        Conduct a comprehensive literature review based on the provided question.
        This method can be called directly by agents without user confirmation.
        """
        return self.wrapped_literature_review(question)
    def wrapped_literature_review(self, question):
        """
        Conduct a literature review based on the provided question.
        This function does not require user confirmation and proceeds directly.
        """
        pqa_input = self.get_general_lit_info(question)        

        question = question + f"Here's the input from a separate literature review, though please be sure to conduct your own review and synthesize the two: {pqa_input}"+ "Make sure to return citations and the working link to the sources"
        # Use the OpenAI API to perform the literature review
        client = OpenAI(api_key=self.openai_api_key)
        response = client.responses.create(
            model=MODEL,
            input=question,
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "medium",
                "user_location": {"type": "approximate", "country": "US", "city": "Boston", "region": "Massachusetts", "timezone": "America/New_York"}}
            ],
        )
        
        return response.output_text
        
    def brief_literature_review(self, question):
        # uses OAI websearch to do the lit review 
        try:
            client = OpenAI(api_key=self.openai_api_key)
            question = question + "Make sure to return citations and the working link to the sources"
            response = client.responses.create(
                model=MODEL,
                input=question,
                tools=[
                    {"type": "web_search_preview",
                    "search_context_size": "medium",
                    "user_location": {"type": "approximate", "country": "US", "city": "Boston", "region": "Massachusetts", "timezone": "America/New_York"}}
                ],
            )

            result = response.output_text
            if result is None or result.strip() == "":
                return f"Literature review completed but returned empty results for query: {question}. This may indicate the specific query didn't match available literature or there were API limitations."
            return result
            
        except Exception as e:
            error_msg = f"Brief literature review failed due to API error: {str(e)}"
            print(f"Literature review error: {error_msg}")
            # Return a meaningful error message instead of None, so the agent can continue with context
            return f"Unable to complete literature review for: {question}. Error: {error_msg}. The agent should proceed with available information or try alternative approaches."
    
  
    def llm_infer(self, conversation, temp = 0.000000001, max_tokens = 1000, image = None, role = None, openai_backend = 'openai', model =MODEL, reasoning_effort=None):

        while True:
            # Measure input size
            input_chars = self._messages_char_count(conversation)
            est_input_tokens = self._estimate_token_count_from_chars(input_chars)
            start_time = time.time()
            try:
                print(f"\033[1;36;40m[LLM Call] Agent={self.agent_name} Backend={self.backend} Model={model} Messages={len(conversation)} InputChars={input_chars} EstInputTokens={est_input_tokens} Temp={temp} MaxTokens={max_tokens}\033[0m")
            except Exception:
                pass

            client = OpenAI(
                api_key=self.openai_api_key 
            )
            if reasoning_effort is not None:
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation,
                    reasoning_effort=reasoning_effort
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation,
                )

            if "I'm sorry, I can't assist with that" in response.choices[0].message.content or "I'm unable to view the image" in response.choices[0].message.content or "I'm unable to provide a definitive answer" in response.choices[0].message.content:
                    # Log output size for this attempt
                    try:
                        elapsed = time.time() - start_time
                        output_text = response.choices[0].message.content or ""
                        output_chars = len(output_text)
                        est_output_tokens = self._estimate_token_count_from_chars(output_chars)
                        # Try to read API-provided usage if present
                        prompt_tokens = completion_tokens = total_tokens = None
                        try:
                            usage = getattr(response, 'usage', None)
                            if usage is not None:
                                if isinstance(usage, dict):
                                    prompt_tokens = usage.get('prompt_tokens')
                                    completion_tokens = usage.get('completion_tokens')
                                    total_tokens = usage.get('total_tokens')
                                else:
                                    prompt_tokens = getattr(usage, 'prompt_tokens', None)
                                    completion_tokens = getattr(usage, 'completion_tokens', None)
                                    total_tokens = getattr(usage, 'total_tokens', None)
                        except Exception:
                            pass
                        if total_tokens is not None:
                            print(f"\033[1;36;40m[LLM Response] Agent={self.agent_name} DurationSec={elapsed:.2f} OutputChars={output_chars} EstOutputTokens={est_output_tokens} APIUsage prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}\033[0m")
                        else:
                            print(f"\033[1;36;40m[LLM Response] Agent={self.agent_name} DurationSec={elapsed:.2f} OutputChars={output_chars} EstOutputTokens={est_output_tokens}\033[0m")
                    except Exception:
                        pass
                    print("Failed to generate response, trying again")
                    continue
            else:
                    output_text = response.choices[0].message.content
                    try:
                        elapsed = time.time() - start_time
                        output_chars = len(output_text or "")
                        est_output_tokens = self._estimate_token_count_from_chars(output_chars)
                        prompt_tokens = completion_tokens = total_tokens = None
                        try:
                            usage = getattr(response, 'usage', None)
                            if usage is not None:
                                if isinstance(usage, dict):
                                    prompt_tokens = usage.get('prompt_tokens')
                                    completion_tokens = usage.get('completion_tokens')
                                    total_tokens = usage.get('total_tokens')
                                else:
                                    prompt_tokens = getattr(usage, 'prompt_tokens', None)
                                    completion_tokens = getattr(usage, 'completion_tokens', None)
                                    total_tokens = getattr(usage, 'total_tokens', None)
                        except Exception:
                            pass
                        if total_tokens is not None:
                            print(f"\033[1;36;40m[LLM Response] Agent={self.agent_name} DurationSec={elapsed:.2f} OutputChars={output_chars} EstOutputTokens={est_output_tokens} APIUsage prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}\033[0m")
                        else:
                            print(f"\033[1;36;40m[LLM Response] Agent={self.agent_name} DurationSec={elapsed:.2f} OutputChars={output_chars} EstOutputTokens={est_output_tokens}\033[0m")
                    except Exception:
                        pass
                    return output_text
                    
    def run_function(self, output):
        try:
            # Find 'Tool-call:' case-insensitively
            m = re.search(r'Tool-call\s*:', output, flags=re.IGNORECASE)
            if not m:
                return None

            # Extract the section after the match and strip code fences
            tool_call_section = output[m.end():]
            tool_call_section = tool_call_section.replace("```python", "").replace("```", "").rstrip()
            lines = [ln for ln in tool_call_section.strip().splitlines() if ln.strip()]
            if not lines:
                return None

            first_line = lines[0].strip()
            if first_line.endswith("('''") or first_line.endswith('("""'):
                quote_type = "'''" if first_line.endswith("('''") else '"""'
                tool_call_str = ""
                found_end = False
                for line in lines:
                    tool_call_str += line + "\n"
                    if line.strip().endswith(quote_type + ")"):
                        found_end = True
                        break
                if not found_end:
                    tool_call_str = first_line
                else:
                    tool_call_str = tool_call_str.strip()
            else:
                tool_call_str = first_line

            if tool_call_str.lower() in ['none', 'no tool call', '', 'null', 'n/a']:
                return None

            if hasattr(self, 'tools_called_in_current_run'):
                tool_name_match = re.match(r'self\.(\w+)\(', tool_call_str)
                if tool_name_match:
                    tool_name = tool_name_match.group(1)
                    self.tools_called_in_current_run.append(tool_name)

            res = eval(tool_call_str)
            return res
        except Exception as e:
            print(f"Error in parsing tool call in {output} got this error {e}")
            return None

    def run_python_code(self, python_code):
        """Executes Python code and returns the output. 
        Input should be a string containing valid Python code. 
        You must use `print` in Python to produce an output from this tool."""
        try:
            result = self.python_repl.run(python_code)
            if len(result) > 1000:
                result_part1 = result[:500]
                result_part2 = result[-500:]
                return result_part1 + "... (truncated, showing first 500 and last 500 characters)" + result_part2
            else:
                return result
        except Exception as e:
            return f"Error executing Python code: {str(e)}"
    
    def reset(self):
        """Reset the agent's conversation and state for a new task."""
        print(f"\033[1;33;40m{self.agent_name} resetting for a new task.\033[0m")
        self.conversation = []
        self.latest_final_response = ''
        self._is_waiting_for_user_input = False
        self._question_for_user_input = None
        self._current_user_query = None

    def process_message_with_smiles_images(self, message_content, role="assistant"):
        """
        Process a message to detect SMILES strings and generate molecule images.
        Returns the original message content and a list of molecule images.
        """
        from rdkit import Chem
        from PIL import Image
        import io
        
        detected_smiles = detect_smiles_in_text(message_content)
        molecule_images = []
        
        for smiles, start, end in detected_smiles:
            try:
                # Validate the SMILES with RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Generate the molecule image
                    image_base64 = self.generate_image(smiles)
                    
                    # Create a Gradio-compatible image component
                    molecule_images.append({
                        'smiles': smiles,
                        'image_base64': image_base64,
                        'position': (start, end)
                    })
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                continue
        
        return message_content, molecule_images
    
    def _messages_char_count(self, messages):
        """
        Compute total character count of message contents for logging.
        """
        try:
            total = 0
            for m in messages:
                content = m.get('content', '') if isinstance(m, dict) else ''
                if isinstance(content, str):
                    total += len(content)
                else:
                    total += len(str(content))
            return total
        except Exception:
            return len(str(messages))

    def _estimate_token_count_from_chars(self, num_chars):
        """
        Rough token estimate from characters (~4 chars/token heuristic).
        """
        try:
            return max(1, int(num_chars / 4))
        except Exception:
            return max(1, int(len(str(num_chars)) / 4))
    def describe_molecule(self, smiles_list):
        """ 
        This function will describe molecules given their SMILES strings.
        It yields progress updates.
        
        Args:
            smiles_list: Can be a single SMILES string or a list of SMILES strings
            
        Returns:
            String with descriptions for all molecules
        """
        # Convert single SMILES to list
        single_molecule = isinstance(smiles_list, str)
        if single_molecule:
            smiles_list = [smiles_list]
            
        # Batch process IUPAC names for efficiency
        iupac_names = self.get_iupac(smiles_list)
        if not isinstance(iupac_names, list):
            iupac_names = [iupac_names]  # Handle single result case
            
        all_descriptions = []
        
        for i, smiles in enumerate(smiles_list):
            yield f"Describing molecule {i+1}/{len(smiles_list)}: {smiles}"
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                all_descriptions.append(f"Molecule {i+1}: Invalid SMILES string: {smiles}")
                continue
                
            image = self.generate_image(smiles)
            iupac = iupac_names[i] if i < len(iupac_names) else f"Error getting IUPAC name for {smiles}"
            
            # Prepare prompt message
            prompt = description_system_prompt + f"\n Here is the SMILES: {smiles} \n Here is the calculated IUPAC name: {iupac}."
            
            messages = [
                {
                    "role": "system",
                    "content":  f"\n Here is an example: {description_case_example}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            description = self.llm_infer(messages, model=MODEL) 
            all_descriptions.append(f"Molecule {i+1} ({smiles}): {description}")
        
        return "\n\n".join(all_descriptions)
    
    def get_iupac(self, smiles):
        # Handle batch processing for lists of SMILES
        if isinstance(smiles, list):
            try:
                processed_smiles = []
                for smi in smiles:
                    try:
                        molecule = Chem.MolFromSmiles(smi)
                        if molecule is not None:
                            [a.SetAtomMapNum(0) for a in molecule.GetAtoms()]
                            Chem.RemoveStereochemistry(molecule)
                            smiles_no_stereo = "<BASE>" + Chem.MolToSmiles(molecule, canonical=True, kekuleSmiles=True)
                            processed_smiles.append(smiles_no_stereo)
                        else:
                            processed_smiles.append(f"<BASE>INVALID_{smi}")  # Placeholder for invalid SMILES
                    except Exception as e:
                        processed_smiles.append(f"<BASE>ERROR_{smi}")  # Placeholder for error cases
                
                # Batch process with the converter
                iupac_results = converter.smiles_to_iupac(processed_smiles, num_beams=1, 
                                                        process_in_batch=True, batch_size=1000)
                
                final_results = []
                for i, (original_smi, result) in enumerate(zip(smiles, iupac_results)):
                    if processed_smiles[i].startswith("<BASE>INVALID_"):
                        final_results.append(f"Error: Invalid SMILES string: {original_smi}")
                    elif processed_smiles[i].startswith("<BASE>ERROR_"):
                        final_results.append(f"Error processing SMILES: {original_smi}")
                    else:
                        final_results.append(result)
                
                return final_results
                
            except Exception as e:
                # Fallback to individual processing if batch fails
                print(f"Batch IUPAC processing failed: {e}. Falling back to individual processing.")
                return [self.get_iupac(smi) for smi in smiles]
        
        # Handle single SMILES (original logic)
        else:
            molecule = Chem.MolFromSmiles(smiles)
            try:
                [a.SetAtomMapNum(0) for a in molecule.GetAtoms()]
                Chem.RemoveStereochemistry(molecule)
                smiles_no_stereo = "<BASE>"+Chem.MolToSmiles(molecule, canonical=True, kekuleSmiles=True)
                return converter.smiles_to_iupac(smiles_no_stereo)
            except Exception as e:
                ()
                return f"Error getting IUPAC name for {smiles}: {e}"
    
    def generate_image(self, smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit import RDLogger
            
            # Suppress RDKit warnings/errors
            RDLogger.DisableLog('rdApp.*')
            
            molecule = Chem.MolFromSmiles(smiles)
            
            if molecule is None:
                print(f"Warning: Could not parse SMILES '{smiles}'")
                return None

            # Generate an image of the molecule
            image = Draw.MolToImage(molecule, size=(256, 256))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')  # Ensure the format matches your image type, e.g., 'PNG' or 'JPEG'
            image_bytes = buffer.getvalue()

            # Encode the binary data to Base64
            image_base64 = base64.b64encode(image_bytes)

            # Convert Base64 bytes to a UTF-8 string
            image_base64_utf8 = image_base64.decode('utf-8')
            
            # Re-enable RDKit logging
            RDLogger.EnableLog('rdApp.*')
            
            return image_base64_utf8
            
        except Exception as e:
            print(f"Error generating image for SMILES '{smiles}': {type(e).__name__} - {e}")
            return None
    
    def load_models(self, model_paths: List[str], device: str = 'cpu') -> List:
        """Load Chemprop models from specified paths with proper CUDA memory management.
        
        Parameters
        ----------
        model_paths : List[str]
            List of paths to model checkpoint files
        device : str
            Device to load models on ('cuda' or 'cpu')
            
        Returns
        -------
        List
            List of loaded models
        """
        models = []
        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

        # Force CPU usage to prevent CUDA memory issues in multi-agent setting
        actual_device = 'cpu'
        if device == 'cuda' and torch.cuda.is_available():
            print(f"\033[1;33;40mWarning: Forcing CPU usage instead of CUDA to prevent memory conflicts in multi-agent setting\033[0m")

        try:
            for i, path in enumerate(model_paths):
                try:
                    print(f"\033[1;34;40mLoading model {i+1}/{len(model_paths)} from {path} on {actual_device}\033[0m")
                    model = chemprop.utils.load_checkpoint(
                        path=path,
                        device=torch.device(actual_device)
                    )
                    models.append(model)
                    
                    # Clear cache after each model to prevent accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"\033[1;31;40mCUDA OOM loading model {path}, falling back to CPU\033[0m")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model = chemprop.utils.load_checkpoint(
                        path=path,
                        device=torch.device('cpu')
                    )
                    models.append(model)
                except Exception as e:
                    print(f"\033[1;31;40mError loading model {path}: {e}\033[0m")
                    # Try to continue with other models rather than failing completely
                    continue

        except Exception as e:
            print(f"\033[1;31;40mCritical error in model loading: {e}\033[0m")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        finally:
            torch.load = _original_torch_load

        if not models:
            raise RuntimeError("No models could be loaded successfully")
            
        print(f"\033[1;32;40mSuccessfully loaded {len(models)} models on {actual_device}\033[0m")
        return models
    
    def make_prediction(self,models,smiles):
        
        """Makes predictions on a list of SMILES with proper memory management.
        
        Parameters
        ----------
        models : List
            List of loaded Chemprop models
        smiles : List[str] 
            SMILES strings to predict
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        # Create a single MoleculeDatapoint for each complete SMILES string

        test_data = []
        for smi in smiles:
            try:
                datapoint = data.MoleculeDatapoint(
                    smiles=[smi],  # Pass as a list with single SMILES
                    features_generator=["rdkit_2d_normalized"]
                )
                test_data.append(datapoint)
            except Exception as e:
                print(f"Error processing SMILES {smi}: {e}")
                return np.array([0.0])  # Return default value on error
        
        test_dset = data.MoleculeDataset(test_data)
        test_loader = data.MoleculeDataLoader(
            test_dset, 
            batch_size=1, 
            num_workers=0, 
            shuffle=False
        )

        try:
            with torch.inference_mode():
                preds_list = []
                unc_list = []
                for i, model in enumerate(models):
                    try:
                        predss, alphas = chemprop.train.predict(model, test_loader, return_unc_parameters=True)
                        predss = torch.tensor(predss)
                        preds = torch.cat(tuple(predss), dim=0)
                        preds = preds.cpu().numpy()

                        alphas = np.array(alphas)
                        S = np.sum(alphas, axis=2)
                        num_classes = alphas.shape[2]
                        unc =  num_classes / S
                        unc = torch.tensor(unc)
                        unc = torch.cat(tuple(unc), dim=0)
                        unc = unc.cpu().numpy()

                        preds_list.append(preds)
                        unc_list.append(unc)
                        
                        # Clear CUDA cache after each model prediction to prevent accumulation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"\033[1;31;40mCUDA OOM during prediction with model {i}, clearing cache\033[0m")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Skip this model and continue with others
                        continue
                    except Exception as e:
                        print(f"\033[1;31;40mError during prediction with model {i}: {e}\033[0m")
                        # Skip this model and continue with others
                        continue

                if not preds_list:
                    print(f"\033[1;31;40mNo successful predictions from any model\033[0m")
                    return np.array([0.0]), np.array([1.0])  # Return default values

                # Ensemble predictions
                sum_preds = sum(preds_list)
                avg_preds = sum_preds / len(preds_list)

                sum_unc = sum(unc_list)
                avg_unc = sum_unc / len(unc_list)

        except Exception as e:
            print(f"\033[1;31;40mCritical error during prediction: {e}\033[0m")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return np.array([0.0]), np.array([1.0])  # Return default values on error

        return avg_preds, avg_unc

    def _assess_molecule_plausibility_with_llm(self, smiles_input):
        """
        Uses an LLM to provide brief chemical plausibility assessments for SMILES string(s).
        Optimized to batch molecule descriptions while keeping individual LLM calls.
        It yields progress updates.
        
        Args:
            smiles_input: A single SMILES string or a list of SMILES strings.
            
        Returns:
            - If input is a single string, returns a single assessment string.
            - If input is a list of strings, returns a dictionary mapping each SMILES to its assessment.
        """
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single_mode = True
        else:
            smiles_list = smiles_input
            single_mode = False

        if not smiles_list:
            return {} if not single_mode else ""

        # Batch generate descriptions for all molecules at once for efficiency
        yield f"Generating descriptions for {len(smiles_list)} molecule(s) in batch..."
        
        batch_descriptions = ""
        description_generator = self.describe_molecule(smiles_list)
        try:
            while True:
                yield next(description_generator)
            # The describe_molecule method returns descriptions separated by double newlines
            batch_descriptions = e.value
        except StopIteration as e:
            batch_descriptions = e.value
        
        # Parse the batch descriptions back into individual descriptions
        # The describe_molecule method returns descriptions separated by double newlines
        individual_descriptions = batch_descriptions.split("\n\n")
        
        # Create a mapping from SMILES to descriptions
        smiles_to_description = {}
        for i, smiles in enumerate(smiles_list):
            if i < len(individual_descriptions):
                # Extract just the description part (remove "Molecule X (SMILES): " prefix)
                desc = individual_descriptions[i]
                # Find the description after the SMILES in parentheses
                if f"({smiles}):" in desc:
                    description = desc.split(f"({smiles}): ", 1)[1] if f"({smiles}): " in desc else desc
                else:
                    description = desc
                smiles_to_description[smiles] = description
            else:
                smiles_to_description[smiles] = f"Error: Could not generate description for {smiles}"

        assessments = {}
        prompt = """You are an expert medicinal chemist. For each molecule provided, analyze its chemical structure and assess its plausibility as a potential pharmaceutical compound. Focus on:

1. Identify key substructures and functional groups
2. Rate each substructure's likelihood of appearing in pharmaceuticals (1-10 scale)
3. Explain any concerning structural features
4. Provide an overall assessment of the molecule's pharmaceutical plausibility

Format your response as:
- Substructure Analysis:
  * [Substructure name]: Rating [X/10] - Very brief explanation
- Overall Assessment: Very brief summary of the molecule's pharmaceutical plausibility

Base your assessment on known medicinal chemistry principles and common pharmaceutical structures."""

        # Individual LLM calls for plausibility assessment (as requested)
        for smiles in smiles_list:
            description = smiles_to_description.get(smiles, f"Error: Could not generate description for {smiles}")
            yield f"Assessing plausibility for: {smiles}"

            conversation = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"SMILES: {smiles}\nDescription: {description}"}
            ]

            response_str = self.llm_infer(conversation)
            assessments[smiles] = response_str

        if single_mode:
            return assessments[smiles_list[0]]
        else:
            return assessments

    def check_druglikeness(self, smiles_input):
        """
        Check a molecule or list of molecules against drug-likeness filters and assess plausibility with an LLM.
        It yields progress updates.
        
        Args:
            smiles_input: Can be a single SMILES string or a list of SMILES strings
            
        Returns:
            String with detailed analysis of each molecule's filter status and plausibility
        """
        from rdkit import Chem
        from rdkit.Chem import FilterCatalog, Descriptors, Crippen
        
        # Convert single SMILES to list
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single_molecule = True
        else:
            smiles_list = smiles_input
            single_molecule = False
        
        # Define filter catalogs
        FILTER_CATALOGS = {
            "BRENK": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
            "PAINS": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS,
        }
        
        def passes_filter(mol, filter_enum):
            """Check if molecule passes a specific filter (True = passes, False = fails)"""
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(filter_enum)
            catalog = FilterCatalog.FilterCatalog(params)
            return not catalog.HasMatch(mol)
        
        def rule_of_5(mol):
            """Check Lipinski's Rule of 5"""
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
        
        def evaluate_molecule(mol):
            """Evaluate a single molecule against all filters and return detailed results."""
            filter_results = {}
            for name, filter_enum in FILTER_CATALOGS.items():
                params = FilterCatalog.FilterCatalogParams()
                params.AddCatalog(filter_enum)
                catalog = FilterCatalog.FilterCatalog(params)
                
                # Use the catalog-level API for more efficient and idiomatic matching
                failures = []
                for entry in catalog.GetMatches(mol):
                    failures.append(entry.GetDescription())
                
                filter_results[name] = failures
            
            ro5_pass = rule_of_5(mol)
            all_filters_pass = all(not v for v in filter_results.values()) and ro5_pass
            
            return {
                "filters": filter_results,
                "RuleOf5": ro5_pass,
                "AllFilters": all_filters_pass,
                "QED": qed(mol)
            }
        
        # LLM Plausibility Check for all valid molecules
        valid_smiles_list = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
        plausibility_assessments = {}
        if valid_smiles_list:
            plausibility_generator = self._assess_molecule_plausibility_with_llm(valid_smiles_list)
            try:
                while True:
                    yield next(plausibility_generator)
            except StopIteration as e:
                plausibility_assessments = e.value
        
        results = []
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    results.append({
                        'smiles': smiles,
                        'status': 'INVALID',
                        'message': f"Invalid SMILES string: {smiles}"
                    })
                    continue
                
                evaluation_results = evaluate_molecule(mol)
                
                # Get the plausibility for the current molecule
                plausibility = plausibility_assessments.get(smiles, "Not assessed (invalid SMILES).")
                
                # Create detailed message
                messages = []
                messages.append(f"SMILES: {smiles}")
                messages.append(f"  Plausibility Assessment: {plausibility}")
                # Report QED metric and explanation
                messages.append(f"  QED: {evaluation_results['QED']:.2f} (Quantitative Estimate of Drug-likeness; ranges from 0 to 1, higher values indicate more drug-like molecules)")
                
                # Separate passed and failed filters
                passed_filters = []
                failed_filters = []
                
                # Check Brenk filter
                brenk_failures = evaluation_results['filters']['BRENK']
                if not brenk_failures:
                    passed_filters.append("Brenk Filter")
                else:
                    failed_filters.append(f"Brenk Filter ({', '.join(brenk_failures)})")
                
                # Check PAINS filter
                pains_failures = evaluation_results['filters']['PAINS']
                if not pains_failures:
                    passed_filters.append("PAINS Filter") 
                else:
                    failed_filters.append(f"PAINS Filter ({', '.join(pains_failures)})")
                
                # Check Rule of 5
                if evaluation_results["RuleOf5"]:
                    passed_filters.append("Rule of 5")
                else:
                    failed_filters.append("Rule of 5")
                
                # Show passed filters
                if passed_filters:
                    messages.append(f"  ✅ PASS: {', '.join(passed_filters)}")
                
                # Show failed filters  
                if failed_filters:
                    messages.append(f"  ❌ FAIL: {', '.join(failed_filters)}")
                
      
                
                results.append({
                    'smiles': smiles,
                    'status': 'ANALYZED',
                    'message': '\n'.join(messages),
                    'filters': evaluation_results
                })
                
            except Exception as e:
                results.append({
                    'smiles': smiles,
                    'status': 'ERROR',
                    'message': f"Error analyzing {smiles}: {str(e)}"
                })
        
        # Format output
        if single_molecule:
            return results[0]['message']
        else:
            output_messages = []
            for i, result in enumerate(results):
                output_messages.append(f"Molecule {i+1}:")
                output_messages.append(result['message'])
                output_messages.append("")  # Empty line for separation
            
            # Add summary
            total = len(results)
            valid = len([r for r in results if r['status'] == 'ANALYZED'])
            passed_all = len([r for r in results if r['status'] == 'ANALYZED' and r['filters']['AllFilters']])
            
            output_messages.append(f"SUMMARY:")
            output_messages.append(f"  - Total molecules: {total}")
            output_messages.append(f"  - Valid molecules: {valid}")
            output_messages.append(f"  - Passed all filters: {passed_all}")
            
            return '\n'.join(output_messages)


### ADMET Agent Utils





'''
Get the best sibling based on calculated ADMET score and seed ADMET score
'''
def filter_siblings(sibling_screening_output:pd.DataFrame,
                    seed_screening_output:pd.DataFrame,):
    
    print("Getting best siblings in this step...")

    comparing_columns = ['SMILES', 'inhibition','fail_absorption','fail_distribution','fail_metabolism',
                         'fail_excretion','fail_toxicity','fail_sum','absorption_admet_score','distribution_admet_score',
                         'metabolism_admet_score','excretion_admet_score', 'toxicity_admet_score','admet_score']

    best_siblings = pd.DataFrame(columns=comparing_columns)

    seed_screening_output = seed_screening_output[comparing_columns]
    sibling_screening_output = sibling_screening_output[comparing_columns]

    # Use a dummy variable for sibling_dict if it's not passed as argument or global
    # Assuming sibling_dict is available in the scope or should be an argument
    # If it is missing from args, this function might be broken.
    # Looking at usage, it seems it might be missing from args here but present in the original context.
    # For now, I will leave it as is but remove commented code.
        
    sibling_dict = sibling_dict[['SMILES','seed']]

    for seed in seed_screening_output['SMILES']:
        seed_output = seed_screening_output[seed_screening_output['SMILES']==seed]
        seed_admet_score = seed_output['admet_score'].values[0]

        siblings = sibling_dict[sibling_dict['seed']==seed]['SMILES']
        sibling_output = sibling_screening_output[sibling_screening_output['SMILES'].isin(siblings)]
        sibling_output = sibling_output[sibling_output['admet_score'] >= seed_admet_score]
        if len(sibling_output) == 0:
            sibling_output = seed_output.copy()
            sibling_output['seed'] = seed
            sibling_output['improve_admet_score_by'] = 0

        else:
            sibling_output = sibling_output.sort_values(by=['admet_score','inhibition'], ascending=[False,False])
            sibling_output = sibling_output.reset_index(drop=True)
            sibling_output = sibling_output.iloc[[0],:]
            sibling_output['seed'] = seed
            sibling_output['improve_admet_score_by'] = sibling_output['admet_score'] - seed_admet_score 

        best_siblings = pd.concat([best_siblings, sibling_output])  

    best_siblings = best_siblings.sort_values(by=['admet_score','inhibition'],
                                              ascending=False)

    return best_siblings




def detect_smiles_in_text(text):
    """
    Detect SMILES strings in text using the format 'Molecule A: <SMILES>' or 'Substructure A: <SMILES>'.
    Returns a list of tuples: (smiles_string, start_position, end_position)
    """
    import re
    
    # Pattern to match "Molecule A: SMILES_STRING" or "Substructure B: SMILES_STRING"
    # where A, B are single uppercase letters.
    pattern = r'(?:Molecule|Substructure)\s+[A-Z]:\s*([^\s]+)'
    
    detected_smiles = []
    matches = re.finditer(pattern, text)
    
    for match in matches:
        # The SMILES string is in group 1
        smiles_candidate = match.group(1)
        
        # We can still use is_likely_smiles for further validation
        if is_likely_smiles(smiles_candidate):
            # The start and end positions should refer to the SMILES string itself
            start_pos = match.start(1)
            end_pos = match.end(1)
            detected_smiles.append((smiles_candidate, start_pos, end_pos))
            
    return detected_smiles

def is_likely_smiles(candidate):
    """
    Enhanced SMILES validation with improved pattern recognition and optional RDKit validation.
    """
    # Basic length and content checks
    if len(candidate) < 3 or len(candidate) > 200:  # Reasonable SMILES length bounds
        return False
    
    # Should not contain spaces or newlines
    if ' ' in candidate or '\n' in candidate or '\t' in candidate:
        return False
    
    # Should not be all numbers 
    if candidate.isdigit():
        return False
    
    # Must contain at least one letter (chemical element)
    if not any(c.isalpha() for c in candidate):
        return False
    
    # Must contain at least one carbon, nitrogen, oxygen, sulfur, phosphorus, or other common elements
    common_elements = set('CNOSPFClBrI')
    if not any(c in common_elements for c in candidate.upper()):
        return False
    
    # Should contain typical SMILES characters
    smiles_chars = set('()[]=-#@.+')
    has_smiles_chars = any(c in smiles_chars for c in candidate)
    
    # Enhanced exclusion patterns
    exclusion_patterns = [
        'http', 'www', 'com', 'org', 'net', 'tool-call', 'self.',
        'reasoning', 'response', 'error', 'warning', 'print', 'import',
        'function', 'return', 'class', 'def', 'if', 'else', 'while',
        'agent', 'summary', 'result', 'output', 'input', 'text'
    ]
    
    if any(pattern in candidate.lower() for pattern in exclusion_patterns):
        return False
    
    # Check for valid SMILES character composition
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]=#+-@./\\%')
    if not all(c in valid_chars for c in candidate):
        return False
    
    # Must have reasonable letter to special character ratio
    letter_count = sum(1 for c in candidate if c.isalpha())
    if letter_count / len(candidate) < 0.2:  # At least 20% letters
        return False
    
    # Check for balanced parentheses and brackets
    paren_count = candidate.count('(') - candidate.count(')')
    bracket_count = candidate.count('[') - candidate.count(']')
    if abs(paren_count) > 1 or abs(bracket_count) > 1:  # Allow slight imbalance for partial SMILES
        return False
    
    # Advanced pattern validation - should look like a chemical structure
    # Check for reasonable element patterns
    import re
    
    # Look for element symbols (C, N, O, S, etc.)
    element_pattern = r'[CNOSPF]|Cl|Br|[BHKSIVYZ]'
    if not re.search(element_pattern, candidate):
        return False
    
    return True




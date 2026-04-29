import re 
import gradio as gr
from gradio import ChatMessage
from utils.utils import Basic_Agent
from utils.inhibition_utils import InhibitionUtils
from utils.admet_utils import ADMETUtils
from utils.optimization_utils import OptimizationUtils
from utils.molecular_editor_utils import MolecularEditorUtils
from prompts import *
from utils.molecular_generation_utils import MolecularGenerationUtils
from utils.automatic_mode_utils import Batch_Mode_Utils


import torch 
import multiprocessing # Added import
from utils.pqa_agent import PQA_Agent

import tempfile
import pandas as pd
import inspect
from typing import List, Optional


# Import SAscore calculation function
try:
    from sascore.sascore import calculateScore as calculate_sascore
except ImportError:
    print("SAscore calculation function not found. SAscore will not be available.")
    calculate_sascore = None


# Tool for LLM to ask the user a question and get input
class Fleming(Basic_Agent):
    def __init__(self, config_path:str):
        super().__init__(config_path, 'Fleming', True, fleming_prompts)
        self.openai_backend = self.config['openai_backend']

        self.system_prompt = fleming_prompts['agent_system_prompt']
        
        # Initialize agents as full conversational agents, not just utilities
        self.admet_agent = ADMETUtils(config_path, 'ADMET Agent', True, admet_agent_prompts)
        self.inhibition_agent = InhibitionUtils(config_path, 'Inhibition Agent', True, inhibition_agent_prompts)
        self.optimization_agent = OptimizationUtils(config_path, 'Optimization Agent', True, optimization_agent_prompts)
        self.generate_agent = MolecularGenerationUtils(config_path, 'Molecular Generation Agent', True, molecular_generation_agent_prompts)
        self.molecular_reasoner_agent = MolecularEditorUtils(config_path, 'Molecular Reasoner Agent', True, molecular_reasoner_agent_prompts)
        #self.run_pipeline_agent = Fleming_automatic(config_path)
        self.batch_mode_agent = Batch_Mode_Utils(config_path, 'Batch Mode Agent', True, None)

        self.conversation = [{"role": "system", "content": self.system_prompt}]

        # Minimal fallback phrases - agents should primarily use FINISHED tool call
        self.no_tool_phrases = ["[complete]"]
        
        # Add tracking for active subagent calls to prevent race conditions
        self._active_subagent_calls = set()
        self.prompts = fleming_prompts
        print("\033[1;32;40mFleming_Agent_Initialized\033[0m")
        self._is_waiting_for_user_input = False
        self._paused_agent_generator = None
        self._paused_agent_context = {}
        self._original_user_query = None  # Store the original user query

    def run_parallel_agents(self, agent_generators):
        """
        Runs multiple agent generators in parallel.
        Usage: self.run_parallel_agents([self.admet_agent(molecule, instruction), self.inhibition_agent(...)])
        """
        import threading
        import queue
        
        # Handle single generator case just in case
        if not isinstance(agent_generators, list):
            agent_generators = [agent_generators]
            
        output_queue = queue.Queue()
        active_threads = 0
        final_results = [None] * len(agent_generators)
        
        def run_agent(idx, generator):
            try:
                for item in generator:
                    if isinstance(item, dict) and item.get("type") == "final":
                        final_results[idx] = item
                    elif isinstance(item, dict) and item.get("type") == "paused":
                        # We currently don't support pausing in parallel mode well
                        # So we'll just log it and continue, effectively ignoring the pause request
                        output_queue.put(ChatMessage(role="system", content=f"Warning: Agent {idx} requested user input, which is not supported in parallel mode. Continuing..."))
                    else:
                        if isinstance(item, str):
                            output_queue.put(ChatMessage(role="assistant", content=f"[Parallel Agent {idx}] {item}"))
                        else:
                            output_queue.put(item)
            except Exception as e:
                output_queue.put(ChatMessage(role="system", content=f"Error in parallel agent {idx}: {e}"))
            finally:
                output_queue.put(("DONE", idx))

        threads = []
        for i, gen in enumerate(agent_generators):
            t = threading.Thread(target=run_agent, args=(i, gen))
            t.start()
            threads.append(t)
            active_threads += 1
            
        while active_threads > 0:
            item = output_queue.get()
            if isinstance(item, tuple) and item[0] == "DONE":
                active_threads -= 1
            else:
                yield item
                
        # Combine summaries
        summaries = []
        all_tools_used = []
        for i, res in enumerate(final_results):
            if res:
                summaries.append(f"Agent {i+1} Result: {res.get('content', '')}")
                all_tools_used.extend(res.get("tools_used", []))
            else:
                summaries.append(f"Agent {i+1} Result: No summary provided.")
                
        combined_summary = "\n\n".join(summaries)
        
        yield {"type": "final", "content": combined_summary, "tools_used": all_tools_used}

    def batchmode_provided_csv(self, csv_path: str, steps_to_run: Optional[List[str]] = None, top_n_to_optimize: int = 100, top_n_to_plausibility: int = 100):
        """
        Runs the batch processing pipeline on a user-provided CSV file.
        This is a generator that yields updates for the UI.
        """
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w', encoding='utf-8') as tmp:
            output_csv_path = tmp.name

        yield from self.batch_mode_agent.run_complete_pipeline_progressive(
            smiles_csv_path=csv_path,
            describe_mol_function=self.describe_molecule,
            output_file=output_csv_path,
            steps_to_run=steps_to_run,
            top_n_to_optimize=top_n_to_optimize,
            top_n_to_plausibility=top_n_to_plausibility
        )

    def _format_molecule_parameter(self, molecule_param):
        """
        Helper function to format molecule parameter for display and processing.
        """
        if isinstance(molecule_param, list):
            if len(molecule_param) == 1:
                return molecule_param[0]
            else:
                return f"{len(molecule_param)} molecules: {', '.join(molecule_param[:3])}{'...' if len(molecule_param) > 3 else ''}"
        else:
            return str(molecule_param)

    def _process_sub_agent_generator(self, agent_generator, agent_name, tool_call_str):
        """
        Helper to process a sub-agent's generator, handling pauses and yields.
        Yields UI updates. Returns the final summary from the sub-agent.
        """
        while True:
            try:
                sub_output = next(agent_generator)

                if isinstance(sub_output, dict) and sub_output.get("type") == "paused":
                    agent_instance = getattr(self, agent_name, self)
                    question = getattr(agent_instance, '_question_for_user_input', 'Please provide more information:')
                    
                    self._paused_agent_generator = agent_generator
                    self._paused_agent_context = {"name": agent_name, "tool_call_str": tool_call_str}
                    self._is_waiting_for_user_input = True
                    
                    self.conversation.append({"role": "assistant", "content": question})
                    yield 'main', ChatMessage(role="assistant", content=f"🤔 {question}")
                    return None

                elif isinstance(sub_output, dict) and sub_output.get("type") == "final":
                    subagent_summary = sub_output["content"]
                    tools_used = sub_output.get("tools_used", [])
                    # Show final sub-agent results in the main (left) chat, and keep the inner (middle)
                    # chat focused on intermediate reasoning/steps.
                    yield 'main', ChatMessage(role="assistant", content=f"[{agent_name} Summary] {subagent_summary}")
                    yield 'inner', ChatMessage(role="system", content=f"[Sub-agent complete] {agent_name} returned a summary and {len(tools_used)} tool(s) used.")
                    return {"content": subagent_summary, "tools_used": tools_used}

                elif isinstance(sub_output, dict) and sub_output.get("type") == "display_molecules":
                    # Forward molecules to the UI
                    yield 'molecules', sub_output.get('items', [])
                    # Also log to inner chat
                    labels = [i.get('label') for i in sub_output.get('items', []) if i.get('label')]
                    yield 'inner', ChatMessage(role="assistant", content=f"[{agent_name}] Displayed {len(labels)} molecule(s): {', '.join(labels)}")

                else:
                    yield 'inner', sub_output

            except StopIteration:
                yield 'inner', ChatMessage(role="system", content=f"Sub-agent {agent_name} finished unexpectedly without a summary.")
                return f"Sub-agent {agent_name} finished unexpectedly."
            except Exception as e:
                yield 'inner', ChatMessage(role="system", content=f"Error during {agent_name} execution: {e}")
                return f"Error during {agent_name} execution: {e}"

    def _generate_final_summary(self):
        """Helper to generate and return the final summary."""
        print(f"\033[1;33;40m Generating final summary.\033[0m")
        try:
            summarization_prompt_template = self.prompts.get('summarization_prompt', "Please summarize your findings now.")
            try:
                summarization_prompt = summarization_prompt_template.format(instruction=self._original_user_query or "the user's request")
            except (KeyError, AttributeError):
                summarization_prompt = summarization_prompt_template
                
            self.conversation.append({"role": "user", "content": summarization_prompt})
            final_summary = self.llm_infer(self.conversation)
            self.conversation.append({"role": "assistant", "content": final_summary})
            self.latest_final_response = final_summary
            return final_summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "I have completed the analysis, but encountered an error generating the final summary. Please review the steps above."

    def run_gradio_chat(self, message: str,
                    history: list,
                    temperature: float,
                    max_new_tokens: int,
                    max_token: int,
                    call_agent: bool,
                    conversation_state: gr.State,
                    max_round: int = 20,
                    seed: int = None,
                    uploaded_csv_path: str = None,
                    use_synspace: bool = False):
        """
        Main chat function for Fleming using a robust, sequential ReAct loop.
        """
        # Set the flag on the optimization agent
        self.optimization_agent.set_use_synspace(use_synspace)

        if self._is_waiting_for_user_input:
            # This block handles RESUMING either a paused sub-agent or Fleming itself waiting for user input.
            self._is_waiting_for_user_input = False
            yield 'main', ChatMessage(role="user", content=message)
            yield 'inner', ChatMessage(role="system", content=f"[User Response] {message}")

            # Check if this is a paused sub-agent or Fleming itself waiting for user input
            if hasattr(self, '_paused_agent_context') and self._paused_agent_context:
                # This is a paused sub-agent case
                paused_agent_name = self._paused_agent_context.get("name")
                agent = getattr(self, paused_agent_name, None) if paused_agent_name else None
                agent_generator = self._paused_agent_generator
                tool_call_str = self._paused_agent_context.get("tool_call_str")

                if not agent or not agent_generator or not paused_agent_name:
                    yield 'inner', ChatMessage(role="system", content="[Fleming Error] Could not find paused agent context. Restarting plan.")
                    self.conversation.append({"role": "user", "content": "My apologies, I lost track of the sub-task. Please re-evaluate your plan to address the user's request."})
                else:
                    user_response_for_agent = f"User provided the requested input: {message}"
                    agent.conversation.append({"role": "system", "content": user_response_for_agent})
                    
                    # Clear paused state before resuming
                    self._paused_agent_generator = None
                    self._paused_agent_context = {}

                    # Resume the generator
                    subagent_result = yield from self._process_sub_agent_generator(agent_generator, paused_agent_name, tool_call_str)

                    if subagent_result is not None:
                        subagent_summary = subagent_result['content']
                        tools_used = subagent_result['tools_used']
                        tool_output_for_context = f"The sub-agent call is complete. Summary: {subagent_summary}. Tools used by sub-agent: {tools_used}"
                        self.conversation.append({"role": "system", "content": tool_output_for_context})
                        # Add guidance prompt for Fleming to decide next steps
                        guidance_prompt = (
                            "Review the sub-agent's results and decide the next best step to satisfy the user's request. "
                            "IMPORTANT: If the sub-agent's results are sufficient to answer the user's query and you completed the plan outlined at the initial step, you MUST use 'Tool-call: self.FINISHED()' to properly display your final answer to the user. "
                            "Only continue with additional tool calls if you genuinely need more information to fully address the user's request or complete the plan. If there are molecules generated or referenced, make sure you continue to call the display_smiles_batch tool."
                            "If ready to present the final answer, use 'Tool-call: self.FINISHED()'. "
                            "Otherwise, specify the next tool call in the format 'Tool-call: <agent_or_tool_call>'."
                        )
                        self.conversation.append({"role": "user", "content": guidance_prompt})
                    else:
                        # Agent paused again, so we return.
                        return
            else:
                # This is Fleming itself waiting for user input.
                yield 'inner', ChatMessage(role="system", content="[Fleming] Continuing with user input...")
                self.conversation.append({"role": "user", "content": message})
 
        else:
            # This block handles a NEW user message.
            self._original_user_query = message  # Store the original user query
            
            # Add context about the uploaded file if it exists
            if uploaded_csv_path:
                message_with_context = f"{message}\n\nIMPORTANT: The user has uploaded a CSV file for analysis, located at '{uploaded_csv_path}'. You should process this file using the `self.batchmode_provided_csv(csv_path=\"{uploaded_csv_path}\")` tool."
            else:
                message_with_context = message

            # Set the current user query on all subagents for context
            self.admet_agent.set_current_user_query(message_with_context)
            self.inhibition_agent.set_current_user_query(message_with_context)
            self.optimization_agent.set_current_user_query(message_with_context)
            self.molecular_reasoner_agent.set_current_user_query(message_with_context)
            self.generate_agent.set_current_user_query(message_with_context)
            
            self.conversation.append({"role": "user", "content": message_with_context})
            yield 'main', ChatMessage(role="user", content=message) # show original message to user
            plan_prompt = self.prompts.get('initial_plan_prompt', "")
            self.conversation.append({"role": "user", "content": plan_prompt})
            initial_plan = self.llm_infer(self.conversation)
            self.conversation.append({"role": "assistant", "content": initial_plan})
            yield 'plan', initial_plan
            yield 'inner', ChatMessage(role="assistant", content=f"[Fleming] {initial_plan}")
            self.conversation.append({"role": "user", "content": "Please begin your work based on the plan. When you have completed the user's task, use the FINISHED tool call to indicate completion."})
        
        for round_num in range(max_round):
            if self._is_waiting_for_user_input:
                return 

            response = self.llm_infer(self.conversation)
            self.conversation.append({"role": "assistant", "content": response})
            yield 'inner', ChatMessage(role="assistant", content=f"[Fleming] {response}")
            
            # Check if we're near the end of rounds and have a final answer
            if round_num >= max_round - 3:
                if response.strip().startswith(('ANSWER:', 'SUMMARY:')):
                    print(f"\033[1;33;40m Near round limit with final answer. Completing early.\033[0m")
                    final_summary = response
                    self.latest_final_response = final_summary
                    yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {final_summary}")
                    return
                else:
                    # Instruct agent to finish up
                    self.conversation.append({"role": "system", "content": "You are nearing the maximum number of turns. Please summarize your findings now, even if the task is incomplete or results are not optimal. Use the FINISHED tool call to submit your summary."})

            tool_call_str = None
            if 'tool-call:' in response.lower():
                tool_call_str = response.split('Tool-call:')[-1].strip()

            if tool_call_str and tool_call_str.strip() and tool_call_str.lower() not in ['none', 'n/a']:
                agent_names = ['admet_agent', 'inhibition_agent', 'optimization_agent', 'molecular_reasoner_agent', 'generate_agent']
                agent_name = next((name for name in agent_names if name in tool_call_str), None)

                if agent_name:
                    try:
                        agent_generator = eval(tool_call_str)
                        subagent_result = yield from self._process_sub_agent_generator(agent_generator, agent_name, tool_call_str)
                        
                        if subagent_result is not None:
                            subagent_summary = subagent_result['content']
                            tools_used = subagent_result['tools_used']
                            tool_output_for_context = f"The sub-agent call is complete. Summary: {subagent_summary}. Tools used by sub-agent: {tools_used}"
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
                            # Nudge Fleming to proceed: either continue with next tool or summarize
                            guidance_prompt = (
                                "Review the sub-agent's results and decide the next best step to satisfy the user's request. "
                                "IMPORTANT: If the sub-agent's results are sufficient to answer the user's query and you completed the plan outlined at the initial step, you MUST use 'Tool-call: self.FINISHED()' to properly display your final answer to the user. "
                                "Only continue with additional tool calls if you genuinely need more information to fully address the user's request or complete the plan. If there are molecules generated or referenced, make sure you continue to call the display_smiles_batch tool."
                                "If ready to present the final answer, use 'Tool-call: self.FINISHED()'. "
                                "Otherwise, specify the next tool call in the format 'Tool-call: <agent_or_tool_call>'."
                            )
                            self.conversation.append({"role": "user", "content": guidance_prompt})
                        else:
                            # Agent paused, so we return and wait for user input.
                            return
 
                    except Exception as e:
                        yield 'inner', ChatMessage(role="system", content=f"Error parsing or starting sub-agent call: {e}")
                else: # It's a direct tool call for Fleming, not a sub-agent.
                    # Check if it's the batchmode tool, which is a generator
                    if 'batchmode' in tool_call_str:
                        try:
                            tool_generator = eval(tool_call_str)
                            summary_update = "Batch processing did not yield a summary." # Default
                            for item in tool_generator:
                                if len(item) == 2 and isinstance(item[0], pd.DataFrame):
                                    df_update, summary_update = item
                                    yield 'batch', df_update, summary_update
                                elif item[0] == 'download':
                                    _, download_path = item
                                    yield 'download', download_path
                                    # Also notify via chat
                                    yield 'inner', ChatMessage(role="assistant", content=f"[Fleming] Generated downloadable results file: {download_path}")
                            tool_output_for_context = f"The batch processing is complete. Here is the summary of the results:\n{summary_update}\n\nNow, display the top molecules from the summary and then use the FINISHED tool call to provide a final answer to the user."
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
                            continue # Continue to the next round of the loop
                        except Exception as e:
                            tool_output_for_context = f"Error running batchmode: {e}"
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
 
                    else: # It's a regular, non-generator tool call
                        print(f"\033[1;36;40mDEBUG: Processing tool call: {tool_call_str}\033[0m")
                        # If the tool call is FINISHED, short-circuit to summary even if parsing/eval might fail (e.g., extra ')')
                        if re.search(r'\bself\.FINISHED\s*\(', tool_call_str, flags=re.IGNORECASE):
                            tool_response = "TASK_COMPLETED_READY_FOR_SUMMARY"
                        else:
                            tool_response = self.run_function(f"Tool-call: {tool_call_str}")
                        print(f"\033[1;36;40mDEBUG: Tool response: {tool_response}\033[0m")
                        # Handle generator-producing tools (e.g., describe_molecule, check_druglikeness, run_parallel_agents)
                        if inspect.isgenerator(tool_response):
                            final_tool_result = None
                            try:
                                while True:
                                    intermediate_step = next(tool_response)
                                    
                                    if isinstance(intermediate_step, dict) and intermediate_step.get("type") == "display_molecules":
                                        yield 'molecules', intermediate_step.get('items', [])
                                        labels = [i.get('label') for i in intermediate_step.get('items', []) if i.get('label')]
                                        yield 'inner', ChatMessage(role="assistant", content=f"[Fleming] Displayed {len(labels)} molecule(s): {', '.join(labels)}")
                                    elif isinstance(intermediate_step, ChatMessage):
                                        yield 'inner', intermediate_step
                                    else:
                                        # Stream progress from the tool
                                        yield 'inner', ChatMessage(role="assistant", content=f"[Fleming Progress] {intermediate_step}")
                            except StopIteration as e:
                                final_tool_result = e.value
                            tool_response = final_tool_result
                        
                        if tool_response is not None:
                            if tool_response == Basic_Agent.USER_INPUT_SENTINEL:
                                question = getattr(self, '_question_for_user_input', 'Please provide more information:')
                                self._is_waiting_for_user_input = True # Fleming itself is waiting
                                self.conversation.append({"role": "assistant", "content": question})
                                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name}] {question}")
                                return
                            elif tool_response == "TASK_COMPLETED_READY_FOR_SUMMARY":
                                # FINISHED tool call was used - trigger summarization
                                final_summary = self._generate_final_summary()
                                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {final_summary}")
                                return
                            elif isinstance(tool_response, dict) and tool_response.get("type") == "display_molecules":
                                # Forward a dedicated molecules payload for UI to render directly
                                items = tool_response.get('items', [])
                                labels = [i.get('label') for i in items if i.get('label')]
                                # Append brief system entry so summaries can include it
                                self.conversation.append({
                                    "role": "system",
                                    "content": f"Displayed {len(items)} molecule(s): labels {', '.join(labels)}"
                                })
                                yield 'molecules', items
                                # Also include a lightweight inner note
                                yield 'inner', ChatMessage(role="assistant", content=f"[Fleming] Displayed {len(items)} molecule(s): {', '.join(labels)}")
                                continue
                            
                            tool_output_for_context = "The tool call is complete. Result: " + str(tool_response)
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
                            yield 'inner', ChatMessage(role="assistant", content=f"[Fleming Tool Result] {tool_output_for_context}")

            elif response.strip().startswith(('ANSWER:')):
                # Direct final answer - use as-is
                print(f"\033[1;33;40m Fleming provided direct final answer.\033[0m")
                self.latest_final_response = response
                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {response}")
                return
        
                
            elif any(phrase in response.lower() for phrase in self.no_tool_phrases):
                # Fallback completion phrases - generate summary
                final_summary = self._generate_final_summary()
                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {final_summary}")
                return

            else: 
                # No tool call, just an intermediate thought. Continue loop.
                pass
        
        # If we reach here, we've hit the max_round limit without Fleming finishing
        print(f"\033[1;33;40m Fleming reached turn limit. Forcing summary.\033[0m")
        final_summary = self._generate_final_summary()
        yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {final_summary}")

    def ask_user(self, question: str) -> str:
        """Ask the user a question and wait for their response.
        
        Args:
            question: The question to ask the user
            
        Returns:
            The user's response (or sentinel value in Gradio mode)
        """
        return super().ask_user(question)

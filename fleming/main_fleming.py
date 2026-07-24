import re 
import gradio as gr
from gradio import ChatMessage
from utils.utils import Basic_Agent, MODEL, PLANNER_MODEL, PLANNER_REASONING
from utils.inhibition_utils import InhibitionUtils
from utils.admet_utils import ADMETUtils
from utils.optimization_utils import OptimizationUtils
from utils.molecular_editor_utils import MolecularEditorUtils
from prompts import *
from utils.molecular_generation_utils import MolecularGenerationUtils
from utils.automatic_mode_utils import Batch_Mode_Utils


import torch 
from rdkit import Chem
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
        self._active_plan_requirements = []
        self._completed_plan_requirements = set()
        self._active_plan_text = ""

    MAX_PARALLEL_AGENTS = 3

    def run_parallel_agents(self, agent_generators):
        """
        Runs multiple agent generators in parallel.
        Usage: self.run_parallel_agents([self.admet_agent(molecule, instruction), self.inhibition_agent(...)])
        A maximum of self.MAX_PARALLEL_AGENTS subagents can be run at once; any extras beyond that
        limit are dropped (with a warning) rather than run.
        """
        import threading
        import queue
        
        # Handle single generator case just in case
        if not isinstance(agent_generators, list):
            agent_generators = [agent_generators]

        if len(agent_generators) > self.MAX_PARALLEL_AGENTS:
            dropped_count = len(agent_generators) - self.MAX_PARALLEL_AGENTS
            for extra_gen in agent_generators[self.MAX_PARALLEL_AGENTS:]:
                # These are already-created generator objects (calling self.xxx_agent(...) doesn't
                # execute anything until iterated), so close them instead of just discarding them.
                close = getattr(extra_gen, 'close', None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            agent_generators = agent_generators[:self.MAX_PARALLEL_AGENTS]
            yield ChatMessage(
                role="system",
                content=(
                    f"Warning: run_parallel_agents received more than {self.MAX_PARALLEL_AGENTS} subagent calls. "
                    f"Only the first {self.MAX_PARALLEL_AGENTS} will run; the remaining {dropped_count} were "
                    f"dropped. Split extra independent work into a follow-up run_parallel_agents call."
                ),
            )

        output_queue = queue.Queue()
        active_threads = 0
        final_results = [None] * len(agent_generators)
        
        def run_agent(idx, generator):
            try:
                # Guard against malformed tool calls where a plain string (or some other
                # non-generator value) ended up in the generators list instead of an actual
                # subagent call, e.g. self.admet_agent(...). Iterating over a string yields
                # it one character at a time, which previously spammed the chat with a
                # separate single-character message per character and looked like a crash.
                if isinstance(generator, (str, bytes)):
                    output_queue.put(ChatMessage(
                        role="system",
                        content=(
                            f"Warning: Parallel agent slot {idx} did not receive a valid subagent call "
                            f"(got a plain string instead of e.g. self.admet_agent(...) or "
                            f"self.inhibition_agent(...)). Skipping this slot. Raw value: {generator!r}"
                        ),
                    ))
                    final_results[idx] = {
                        "type": "final",
                        "content": "Skipped: this parallel slot received a plain string instead of a subagent call.",
                        "tools_used": [],
                    }
                    return
                if not (inspect.isgenerator(generator) or hasattr(generator, '__next__')):
                    output_queue.put(ChatMessage(
                        role="system",
                        content=(
                            f"Warning: Parallel agent slot {idx} did not receive a valid subagent generator "
                            f"(got {type(generator).__name__}). Skipping this slot."
                        ),
                    ))
                    final_results[idx] = {
                        "type": "final",
                        "content": "Skipped: this parallel slot did not receive a valid subagent call.",
                        "tools_used": [],
                    }
                    return
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

    def _extract_plan_requirements(self, plan_text: str):
        if not plan_text:
            return []
        plan_lower = plan_text.lower()
        candidate_lines = []
        in_mermaid = False
        for raw_line in plan_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```mermaid"):
                in_mermaid = True
                continue
            if in_mermaid and line.startswith("```"):
                in_mermaid = False
                continue
            if re.match(r"^\d+\.", line):
                candidate_lines.append(line.lower())
                continue
            if in_mermaid:
                candidate_lines.append(line.lower())

        if not candidate_lines:
            candidate_lines = [plan_lower]

        requirement_specs = [
            ("inhibition", "Run inhibition analysis", [
                "inhibition agent", "self.inhibition_agent", "get_chemprop_prediction_string", "inhibition prediction"
            ]),
            ("admet", "Run ADMET analysis", [
                "admet agent", "self.admet_agent", "admet prediction", "cytotoxicity", "bbbp", "solubility", "side effect"
            ]),
            ("literature", "Run literature review", [
                "literature review", "brief literature", "self.brief_literature_review", "pqa", "paperqa", "literature search"
            ]),
            ("describe", "Describe molecule for downstream analysis", [
                "describe molecule", "self.describe_molecule", "molecular description"
            ]),
            ("display", "Display relevant molecules", [
                "display smiles", "display molecule", "show molecule", "render molecule", "visual interface", "display_smiles_batch"
            ]),
            ("optimization", "Run optimization analysis", [
                "optimization agent", "self.optimization_agent", "optimiz"
            ]),
            ("generation", "Run molecule generation", [
                "molecular generation agent", "self.generate_agent", "generate molecule"
            ]),
        ]
        requirements = []
        seen = set()
        for key, label, patterns in requirement_specs:
            if key in seen:
                continue
            matched = False
            for line in candidate_lines:
                if any(pattern in line for pattern in patterns):
                    if key == "optimization":
                        if any(neg in line for neg in [
                            "no optimization", "not optimize", "do not optimize", "without optimization",
                            "optimization not required", "skip optimization"
                        ]):
                            continue
                    matched = True
                    break
            if matched:
                requirements.append({"key": key, "label": label})
                seen.add(key)

        original_query_lower = (self._original_user_query or "").lower()
        optimization_intent = any(token in original_query_lower for token in [
            "optimiz", "improve", "neighbor", "analogs", "analog", "modify", "edit", "generate", "design"
        ])
        if not optimization_intent:
            requirements = [req for req in requirements if req["key"] != "optimization"]

        return requirements

    def _mark_plan_requirement_complete(self, requirement_key: str):
        if requirement_key:
            self._completed_plan_requirements.add(requirement_key)

    def _mark_plan_completion_from_tools(self, tools_used):
        if not tools_used:
            return
        for tool_name in tools_used:
            tool_name = str(tool_name)
            if any(name in tool_name for name in ["inhibition_agent", "get_chemprop_prediction_string"]):
                self._mark_plan_requirement_complete("inhibition")
            if any(name in tool_name for name in [
                "admet_agent", "get_admet_prediction", "get_cytotoxicity", "get_bbbp_prediction",
                "get_solubility_prediction", "get_side_effect_prediction"
            ]):
                self._mark_plan_requirement_complete("admet")
            if any(name in tool_name for name in [
                "brief_literature_review", "literature_review", "pqa_molecular_analysis", "run_molecule"
            ]):
                self._mark_plan_requirement_complete("literature")
            if "describe_molecule" in tool_name:
                self._mark_plan_requirement_complete("describe")
            if any(name in tool_name for name in ["display_smiles", "display_smiles_batch"]):
                self._mark_plan_requirement_complete("display")
            if "optimization_agent" in tool_name:
                self._mark_plan_requirement_complete("optimization")
            if "generate_agent" in tool_name:
                self._mark_plan_requirement_complete("generation")

    def _get_unmet_plan_requirements(self):
        return [
            requirement for requirement in self._active_plan_requirements
            if requirement["key"] not in self._completed_plan_requirements
        ]

    def _get_plan_blocking_message(self):
        unmet = self._get_unmet_plan_requirements()
        if not unmet:
            return None
        remaining = "; ".join(req["label"] for req in unmet[:5])
        return (
            "You are attempting to conclude before completing the planned dependency graph. "
            f"Remaining required planned steps: {remaining}. "
            "Either continue execution based on the current plan, or output `PLAN_UPDATE:` followed by a revised numbered plan and optional Mermaid dependency graph if the plan should change, "
            "or output `BLOCKED:` followed by a concise user-facing explanation if the remaining steps cannot be completed with the available tools/results. "
            "Do not use FINISHED or provide a final answer yet unless you are using the BLOCKED path."
        )

    def _get_plan_gate_notice(self):
        """Short, human-readable version of the plan-gate nudge for the reasoning-trace panel
        (the full instruction in `_get_plan_blocking_message` is what actually goes to the model)."""
        unmet = self._get_unmet_plan_requirements()
        if not unmet:
            return None
        remaining = "; ".join(req["label"] for req in unmet[:5])
        return f"Plan not yet complete — continuing with remaining steps: {remaining}."

    def _extract_tool_names_from_call(self, tool_call_str: str):
        if not tool_call_str:
            return []
        matches = re.findall(r"self\.(\w+)\s*\(", tool_call_str)
        return matches

    def _response_is_plan_update(self, response_text: str) -> bool:
        if not response_text:
            return False
        stripped = response_text.strip()
        lowered = stripped.lower()
        return lowered.startswith("plan_update:") or (
            "```mermaid" in lowered and re.search(r"\b1\.\s", stripped) is not None
        )

    def _extract_plan_update_text(self, response_text: str) -> str:
        stripped = (response_text or "").strip()
        if stripped.lower().startswith("plan_update:"):
            return stripped.split(":", 1)[1].strip()
        return stripped

    def _apply_plan_update(self, plan_text: str):
        if not plan_text:
            return None
        new_requirements = self._extract_plan_requirements(plan_text)
        valid_completed = {req["key"] for req in new_requirements}
        self._completed_plan_requirements &= valid_completed
        self._active_plan_requirements = new_requirements
        self._active_plan_text = plan_text
        return plan_text

    def _response_is_blocked(self, response_text: str) -> bool:
        if not response_text:
            return False
        lowered = response_text.strip().lower()
        return lowered.startswith("blocked:") or lowered.startswith("incomplete:")

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
        def _normalize_subagent_chat_message(message):
            if not isinstance(message, ChatMessage):
                return message
            content = getattr(message, "content", "")
            content = "" if content is None else str(content).strip()
            bare_agent_label = f"[{agent_name}]"
            if not content or content == bare_agent_label:
                return None
            return ChatMessage(role=message.role, content=content)

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
                    yield 'inner', ChatMessage(role="assistant", content=f"[{agent_name} Summary] {subagent_summary}")
                    yield 'inner', ChatMessage(role="system", content=f"[Sub-agent complete] {agent_name} returned a summary and {len(tools_used)} tool(s) used.")
                    return {"content": subagent_summary, "tools_used": tools_used}

                elif isinstance(sub_output, dict) and sub_output.get("type") == "display_molecules":
                    # Forward molecules to the UI
                    yield 'molecules', sub_output.get('items', [])
                    # Also log to inner chat
                    labels = [i.get('label') for i in sub_output.get('items', []) if i.get('label')]
                    yield 'inner', ChatMessage(role="assistant", content=f"[{agent_name}] Displayed {len(labels)} molecule(s): {', '.join(labels)}")

                else:
                    normalized_output = _normalize_subagent_chat_message(sub_output)
                    if normalized_output is not None:
                        yield 'inner', normalized_output

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
            # Fleming's final, user-facing answer synthesizes every sub-agent's results (quantitative
            # values, literature links, etc.) - use the strong model and a generous token budget so
            # multi-molecule responses with citations aren't cut short.
            final_summary = self.llm_infer(self.conversation, model=PLANNER_MODEL, max_tokens=3000)
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
        last_emitted_main = history[-1] if history else None
        last_emitted_inner = None
        # Round 0 always plans with the full model. Beyond that, `force_planner_next_round` is set
        # explicitly at the exact point a re-planning nudge is issued (plan-gate block, plan update,
        # lost sub-task context), rather than inferred by scanning message text for phrases.
        force_planner_next_round = False

        def _extract_valid_smiles_tokens(text, max_tokens=15):
            """
            Best-effort scan for RDKit-parseable SMILES substrings anywhere in free text.
            Used so that candidate molecules mentioned late in a long sub-agent summary
            aren't silently lost when that summary gets truncated for Fleming's context.
            """
            candidates = re.findall(r'[A-Za-z0-9@\+\-\[\]\(\)=#\.\\/%]{4,120}', text)
            valid = []
            seen = set()
            for cand in candidates:
                cand = cand.strip('.,;:')
                if not cand or cand in seen or not any(c.isalpha() for c in cand):
                    continue
                try:
                    mol = Chem.MolFromSmiles(cand)
                except Exception:
                    mol = None
                if mol is not None:
                    valid.append(cand)
                    seen.add(cand)
                if len(valid) >= max_tokens:
                    break
            return valid

        def _compact_text(value, max_chars=1200):
            text = str(value).strip()
            if len(text) <= max_chars:
                return text
            truncated = text[:max_chars].rstrip() + "... [truncated]"
            # If truncation would drop candidate SMILES mentioned later in the full text,
            # append them explicitly so downstream steps (e.g. literature review of an
            # optimized analog) still know which molecule(s) to reference.
            try:
                full_smiles = _extract_valid_smiles_tokens(text)
                missing = [s for s in full_smiles if s not in truncated]
                if missing:
                    truncated += (
                        "\nCandidate SMILES mentioned in the full report (preserved despite truncation): "
                        + ", ".join(missing)
                    )
            except Exception:
                pass
            return truncated

        def _emit_chat(panel, chat_message):
            nonlocal last_emitted_main, last_emitted_inner
            last_item = last_emitted_main if panel == "main" else last_emitted_inner
            if (
                isinstance(last_item, ChatMessage)
                and isinstance(chat_message, ChatMessage)
                and getattr(last_item, "role", None) == getattr(chat_message, "role", None)
                and getattr(last_item, "content", None) == getattr(chat_message, "content", None)
            ):
                return
            if panel == "main":
                last_emitted_main = chat_message
            else:
                last_emitted_inner = chat_message
            yield panel, chat_message

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
                    force_planner_next_round = True
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
                        compact_subagent_summary = _compact_text(subagent_summary, max_chars=3000)
                        tools_used_display = ", ".join(map(str, tools_used[:5])) if tools_used else "none"
                        if len(tools_used) > 5:
                            tools_used_display += ", ..."
                        tool_output_for_context = (
                            f"The sub-agent call is complete. Summary: {compact_subagent_summary}. "
                            f"Tools used by sub-agent: {tools_used_display}."
                        )
                        self.conversation.append({"role": "system", "content": tool_output_for_context})
                        # Add guidance prompt for Fleming to decide next steps
                        guidance_prompt = (
                            "Review the sub-agent's results and decide the next best step to satisfy the user's request. "
                            "Use the concrete outputs from the sub-agent above in your reasoning and final answer; do not say results were unavailable if they were returned here. "
                            "If the sub-agent returned quantitative values, preserve those values in your answer. "
                            "If the result is sufficient to answer the user, do not add more reasoning. Immediately use 'Tool-call: self.FINISHED()'. "
                            "Only continue with another tool call if critical information is still missing. "
                            "If molecules were generated or referenced and still need display, call the display_smiles_batch tool before finishing. "
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
            self.tools_called_in_current_run = []
            self._active_plan_requirements = []
            self._completed_plan_requirements = set()
            self._active_plan_text = ""
            
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
            initial_plan = self.llm_infer(
                self.conversation,
                model=PLANNER_MODEL,
                reasoning_effort=PLANNER_REASONING,
            )
            self.conversation.append({"role": "assistant", "content": initial_plan})
            self._apply_plan_update(initial_plan)
            yield 'plan', initial_plan
            self.conversation.append({"role": "user", "content": "Please begin your work based on the plan. When you have completed the user's task, use the FINISHED tool call to indicate completion."})
        
        for round_num in range(max_round):
            if self._is_waiting_for_user_input:
                return 

            response_model = PLANNER_MODEL if (force_planner_next_round or round_num == 0) else MODEL
            response_reasoning = PLANNER_REASONING if (force_planner_next_round or round_num == 0) else None
            force_planner_next_round = False
            response = self.llm_infer(
                self.conversation,
                model=response_model,
                reasoning_effort=response_reasoning,
            )
            self.conversation.append({"role": "assistant", "content": response})
            response_stripped = response.strip()
            is_finished_call = False
            tool_call_str = None
            if 'tool-call:' in response.lower():
                tool_call_str = response.split('Tool-call:')[-1].strip()
                is_finished_call = bool(re.search(r'\bself\.FINISHED\s*\(', tool_call_str, flags=re.IGNORECASE))
            is_direct_final = response_stripped.startswith(('ANSWER:', 'SUMMARY:'))
            is_fallback_completion = any(phrase in response.lower() for phrase in self.no_tool_phrases)
            is_plan_update = self._response_is_plan_update(response_stripped)
            is_blocked_response = self._response_is_blocked(response_stripped)

            if is_plan_update:
                updated_plan = self._extract_plan_update_text(response_stripped)
                self._apply_plan_update(updated_plan)
                yield 'plan', updated_plan
                self.conversation.append({
                    "role": "user",
                    "content": "The revised plan is now active. Continue execution based on the updated dependency graph."
                })
                yield 'inner', ChatMessage(role="system", content="[Plan] Revised plan accepted and activated.")
                force_planner_next_round = True
                continue

            if is_blocked_response:
                blocking_prefix, _, blocking_body = response_stripped.partition(":")
                blocker_reason = blocking_body.strip() or response_stripped
                user_message = f"ANSWER: I could not complete the full planned workflow. Reason: {blocker_reason}"
                self.latest_final_response = user_message
                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {user_message}")
                return

            if not is_direct_final and not is_finished_call and not is_fallback_completion:
                reasoning_text = response.split('Tool-call:')[0].strip() if tool_call_str else response_stripped
                if reasoning_text:
                    yield from _emit_chat('inner', ChatMessage(role="assistant", content=f"[Fleming] {reasoning_text}"))
                if tool_call_str and tool_call_str.strip() and tool_call_str.lower() not in ['none', 'n/a']:
                    yield from _emit_chat('inner', ChatMessage(role="assistant", content=f"[Fleming] Calling tool: `{tool_call_str.strip()}`"))
            
            # Check if we're near the end of rounds and have a final answer
            if round_num >= max_round - 3:
                if is_direct_final:
                    blocking_message = self._get_plan_blocking_message()
                    if blocking_message:
                        self.conversation.append({"role": "user", "content": blocking_message})
                        yield 'inner', ChatMessage(role="system", content=f"[Plan] {self._get_plan_gate_notice() or blocking_message}")
                        force_planner_next_round = True
                        continue
                    print(f"\033[1;33;40m Near round limit with final answer. Completing early.\033[0m")
                    final_summary = response
                    self.latest_final_response = final_summary
                    yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {final_summary}")
                    return
                else:
                    # Instruct agent to finish up
                    self.conversation.append({"role": "system", "content": "You are nearing the maximum number of turns. Please summarize your findings now, even if the task is incomplete or results are not optimal. Use the FINISHED tool call to submit your summary."})
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
                            self._mark_plan_completion_from_tools([agent_name] + list(tools_used or []))
                            compact_subagent_summary = _compact_text(subagent_summary, max_chars=3000)
                            tools_used_display = ", ".join(map(str, tools_used[:5])) if tools_used else "none"
                            if len(tools_used) > 5:
                                tools_used_display += ", ..."
                            tool_output_for_context = (
                                f"The sub-agent call is complete. Summary: {compact_subagent_summary}. "
                                f"Tools used by sub-agent: {tools_used_display}."
                            )
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
                            # Nudge Fleming to proceed: either continue with next tool or summarize
                            guidance_prompt = (
                                "Review the sub-agent's results and decide the next best step to satisfy the user's request. "
                                "Use the concrete outputs from the sub-agent above in your reasoning and final answer; do not say results were unavailable if they were returned here. "
                                "If the sub-agent returned quantitative values, preserve those values in your answer. "
                                "If the result is sufficient to answer the user, do not add more reasoning. Immediately use 'Tool-call: self.FINISHED()'. "
                                "Only continue with another tool call if critical information is still missing. "
                                "If molecules were generated or referenced and still need display, call the display_smiles_batch tool before finishing. "
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
                            compact_batch_summary = _compact_text(summary_update, max_chars=1000)
                            tool_output_for_context = (
                                "The batch processing is complete. Summary of results: "
                                f"{compact_batch_summary}\n\n"
                                "If this is enough to answer the user, display the top molecules if needed and then immediately use the FINISHED tool call."
                            )
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
                            continue # Continue to the next round of the loop
                        except Exception as e:
                            tool_output_for_context = f"Error running batchmode: {e}"
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
 
                    else: # It's a regular, non-generator tool call
                        print(f"\033[1;36;40mDEBUG: Processing tool call: {tool_call_str}\033[0m")
                        direct_tool_names = self._extract_tool_names_from_call(tool_call_str)
                        # If the tool call is FINISHED, short-circuit to summary even if parsing/eval might fail (e.g., extra ')')
                        if re.search(r'\bself\.FINISHED\s*\(', tool_call_str, flags=re.IGNORECASE):
                            tool_response = "TASK_COMPLETED_READY_FOR_SUMMARY"
                        else:
                            try:
                                tool_response = self.run_function(f"Tool-call: {tool_call_str}")
                            except Exception as e:
                                # Surface the failure to the model instead of silently dropping it,
                                # so it gets a chance to retry with a different tool/arguments.
                                tool_response = f"Error executing tool '{tool_call_str}': {e}"
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
                                blocking_message = self._get_plan_blocking_message()
                                if blocking_message:
                                    self.conversation.append({"role": "user", "content": blocking_message})
                                    yield 'inner', ChatMessage(role="system", content=f"[Plan] {self._get_plan_gate_notice() or blocking_message}")
                                    force_planner_next_round = True
                                    continue
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
                                self._mark_plan_requirement_complete("display")
                                yield 'molecules', items
                                # Also include a lightweight inner note
                                yield 'inner', ChatMessage(role="assistant", content=f"[Fleming] Displayed {len(items)} molecule(s): {', '.join(labels)}")
                                continue
                            elif isinstance(tool_response, str) and tool_response.startswith("Error executing tool"):
                                yield 'inner', ChatMessage(role="system", content=f"[Fleming Error] {tool_response}")
                                self.conversation.append({"role": "system", "content": tool_response})
                                self.conversation.append({"role": "user", "content": "That tool call failed. Please re-evaluate your plan. You can try the tool again with different parameters, or try a different tool."})
                                force_planner_next_round = True
                                continue

                            tool_output_for_context = (
                                "The tool call is complete. Result: "
                                + _compact_text(tool_response, max_chars=1000)
                                + " Use this concrete tool output directly in later reasoning and in the final answer."
                            )
                            self._mark_plan_completion_from_tools(direct_tool_names)
                            self.conversation.append({"role": "system", "content": tool_output_for_context})
                            yield 'inner', ChatMessage(role="assistant", content=f"[Fleming Tool Result] {tool_output_for_context}")

            elif response.strip().startswith(('ANSWER:')):
                # Direct final answer - use as-is
                blocking_message = self._get_plan_blocking_message()
                if blocking_message:
                    self.conversation.append({"role": "user", "content": blocking_message})
                    yield 'inner', ChatMessage(role="system", content=f"[Plan] {self._get_plan_gate_notice() or blocking_message}")
                    force_planner_next_round = True
                    continue
                print(f"\033[1;33;40m Fleming provided direct final answer.\033[0m")
                self.latest_final_response = response
                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {response}")
                return
        
                
            elif any(phrase in response.lower() for phrase in self.no_tool_phrases):
                # Fallback completion phrases - generate summary
                blocking_message = self._get_plan_blocking_message()
                if blocking_message:
                    self.conversation.append({"role": "user", "content": blocking_message})
                    yield 'inner', ChatMessage(role="system", content=f"[Plan] {self._get_plan_gate_notice() or blocking_message}")
                    force_planner_next_round = True
                    continue
                final_summary = self._generate_final_summary()
                yield 'main', ChatMessage(role="assistant", content=f"[{self.agent_name} Summary] {final_summary}")
                return

            else: 
                # No tool call, just an intermediate thought. Continue loop.
                pass
        
        # If we reach here, we've hit the max_round limit without Fleming finishing
        print(f"\033[1;33;40m Fleming reached turn limit. Forcing summary.\033[0m")
        blocking_message = self._get_plan_blocking_message()
        if blocking_message:
            self.conversation.append({"role": "user", "content": blocking_message})
            yield 'inner', ChatMessage(role="system", content=f"[Plan] {self._get_plan_gate_notice() or blocking_message}")
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

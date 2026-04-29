from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
import yaml

try:  # pragma: no cover - optional dependency at runtime
    from agents import Agent, ModelSettings, Runner, function_tool
except ImportError:  # pragma: no cover - optional dependency at runtime
    Agent = None
    ModelSettings = None
    Runner = None
    function_tool = None

from utils.admet_utils import ADMETUtils
from utils.inhibition_utils import InhibitionUtils
from utils.molecular_editor_utils import MolecularEditorUtils
from utils.molecular_generation_utils import MolecularGenerationUtils
from utils.optimization_utils import OptimizationUtils


@dataclass
class FlemingSessionContext:
    optimization_strategy: str = "auto"
    displayed_molecules: list[dict[str, str]] = field(default_factory=list)


class FlemingCodexSDK:
    def __init__(
        self,
        config_path: str,
        model: str = "gpt-5.4",
        reasoning_effort: str = "low",
        optimization_strategy: str = "auto",
    ) -> None:
        if Agent is None or Runner is None or function_tool is None:
            raise ImportError(
                "The OpenAI Agents SDK is required for the codex_sdk implementation. "
                "Install it with `pip install openai-agents`."
            )

        self.config_path = config_path
        with open(config_path, "r") as handle:
            self.config = yaml.safe_load(handle) or {}
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.optimization_strategy = (
            optimization_strategy or self.config.get("optimization_strategy", "auto")
        )
        self._display_items: list[dict[str, str]] = []
        self._run_notes: list[str] = []

        self.admet_utils = ADMETUtils(config_path, "ADMET Utils", False, None)
        self.inhibition_utils = InhibitionUtils(config_path, "Inhibition Utils", False, None)
        self.optimization_utils = OptimizationUtils(
            config_path,
            "Optimization Utils",
            False,
            None,
            admet_utils=self.admet_utils,
            inhibition_utils=self.inhibition_utils,
        )
        self.generation_utils = MolecularGenerationUtils(config_path, "Generation Utils", False, None)
        self.molecular_editor_utils = MolecularEditorUtils(config_path, "Molecular Editor Utils", False, None)

        self.agent = self._build_agent()

    @staticmethod
    def _coerce_smiles(smiles: str | list[str]) -> list[str]:
        if isinstance(smiles, str):
            return [smiles]
        return [item for item in smiles if isinstance(item, str) and item.strip()]

    def _build_tools(self) -> list[Any]:
        @function_tool(name_override="admet_agent")
        def admet_agent(smiles: list[str], instruction: str = "") -> str:
            """Assess ADMET and synthesizability-relevant properties for one or more SMILES strings."""
            normalized = self._coerce_smiles(smiles)
            summary = self.admet_utils.get_admet_prediction_string_summary(normalized)
            if "novel" in instruction.lower() or "synth" in instruction.lower():
                summary = f"{summary}\n\n{self.admet_utils.get_novelty_and_sascore(normalized)}"
            self._run_notes.append(f"Ran ADMET on {len(normalized)} molecule(s).")
            return summary

        @function_tool(name_override="inhibition_agent")
        def inhibition_agent(smiles: list[str], instruction: str = "") -> str:
            """Predict TB inhibition and uncertainty for one or more SMILES strings."""
            normalized = self._coerce_smiles(smiles)
            self._run_notes.append(f"Ran inhibition prediction on {len(normalized)} molecule(s).")
            return self.inhibition_utils.get_chemprop_prediction_string(normalized)

        @function_tool(name_override="optimization_agent")
        def optimization_agent(
            smiles: list[str],
            instruction: str = "",
            optimization_strategy: str = "auto",
        ) -> str:
            """Optimize one or more molecules using synspace, an immediate-chemical-space explorer, LLM analog search, or auto selection."""
            normalized = self._coerce_smiles(smiles)
            chosen = (optimization_strategy or self.optimization_strategy or "auto").lower()
            self.optimization_utils.set_optimization_strategy(chosen)
            self._run_notes.append(
                f"Ran optimization on {len(normalized)} molecule(s) using `{chosen}`."
            )
            return self.optimization_utils.get_candidates(normalized)

        @function_tool(name_override="generate_agent")
        def generate_agent(instruction: str, model_name: str = "diffusion") -> str:
            """Generate a candidate molecule, typically with the diffusion generator."""
            result = self.generation_utils.generate_molecule_string(model_name=model_name)
            generated_smiles = [
                smi for smi in self.generation_utils.last_generated_smiles
                if isinstance(smi, str) and smi.strip()
            ]
            if generated_smiles:
                payload = self.optimization_utils.display_smiles_batch(generated_smiles)
                if isinstance(payload, dict):
                    self._display_items.extend(payload.get("items", []))
            source = self.generation_utils.last_generation_source or model_name
            self._run_notes.append(
                f"Generated {len(generated_smiles) or 1} molecule(s) using `{source}`."
            )
            return result

        @function_tool(name_override="molecular_reasoner_agent")
        def molecular_reasoner_agent(smiles: str, instruction: str = "") -> str:
            """Compute molecule-level structural metadata and lightweight edits."""
            if "iupac" in instruction.lower():
                self._run_notes.append("Generated an IUPAC name.")
                return self.molecular_editor_utils.get_iupac(smiles)
            validity = self.molecular_editor_utils.is_valid_smiles(smiles)
            self._run_notes.append("Ran molecular reasoning/validation.")
            return json.dumps({"smiles": smiles, "valid": validity, "instruction": instruction})

        @function_tool(name_override="describe_molecule")
        def describe_molecule(smiles: list[str]) -> str:
            """Describe one or more molecules in natural language."""
            normalized = self._coerce_smiles(smiles)
            generator = self.generation_utils.describe_molecule(normalized)
            if hasattr(generator, "__iter__") and not isinstance(generator, str):
                try:
                    while True:
                        next(generator)
                except StopIteration as stop:
                    self._run_notes.append(f"Described {len(normalized)} molecule(s).")
                    return stop.value or ""
            self._run_notes.append(f"Described {len(normalized)} molecule(s).")
            return str(generator)

        @function_tool(name_override="display_smiles_batch")
        def display_smiles_batch(smiles: list[str], labels: list[str] | None = None) -> str:
            """Return a structured payload describing molecules that should be rendered in a client."""
            payload = self.optimization_utils.display_smiles_batch(self._coerce_smiles(smiles), labels)
            if isinstance(payload, dict):
                self._display_items.extend(payload.get("items", []))
            self._run_notes.append(
                f"Prepared {len(payload.get('items', [])) if isinstance(payload, dict) else 0} molecule(s) for display."
            )
            return json.dumps(payload)

        @function_tool(name_override="set_optimization_strategy")
        def set_optimization_strategy(optimization_strategy: str) -> str:
            """Set the default optimization backend. Valid values: auto, immediate, synspace, llm."""
            chosen = optimization_strategy.lower().strip()
            self.optimization_strategy = chosen
            self.optimization_utils.set_optimization_strategy(chosen)
            self._run_notes.append(f"Set optimization strategy to `{chosen}`.")
            return f"default optimization strategy set to {chosen}"

        return [
            admet_agent,
            inhibition_agent,
            optimization_agent,
            generate_agent,
            molecular_reasoner_agent,
            describe_molecule,
            display_smiles_batch,
            set_optimization_strategy,
        ]

    def _build_agent(self):
        instructions = (
            "You are Fleming, a medicinal chemistry copilot focused on tuberculosis antibiotic design. "
            "Use the provided tools for inhibition, ADMET, optimization, molecular generation, and molecular reasoning. "
            "Be concise, helpful, and practical. Explain tradeoffs clearly and steer the user toward actionable next steps. "
            "When optimization is requested, preserve local chemical similarity unless the user explicitly asks for broader exploration. "
            "If the user specifies an optimization backend, honor it. Otherwise use the current default strategy. "
            "Whenever you mention a concrete molecule, candidate, analog, scaffold, or generated structure, call display_smiles_batch. "
            "Do this even for a single molecule and even if the molecule already appears in the prose."
        )
        return Agent(
            name="FlemingCodexSDK",
            instructions=instructions,
            tools=self._build_tools(),
            model=self.model,
            model_settings=ModelSettings(parallel_tool_calls=True),
        )

    def run(self, prompt: str, optimization_strategy: str | None = None, max_turns: int = 20):
        self.clear_run_state()
        context = FlemingSessionContext(
            optimization_strategy=optimization_strategy or self.optimization_strategy
        )
        if optimization_strategy:
            self.optimization_utils.set_optimization_strategy(optimization_strategy)
        return Runner.run_sync(
            self.agent,
            prompt,
            context=context,
            max_turns=max_turns,
        )

    def clear_run_state(self) -> None:
        self._display_items = []
        self._run_notes = []

    def get_display_items(self) -> list[dict[str, str]]:
        return list(self._display_items)

    def get_run_notes(self) -> list[str]:
        return list(self._run_notes)

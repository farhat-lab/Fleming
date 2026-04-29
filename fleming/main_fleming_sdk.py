from __future__ import annotations
from typing import Any

from gradio import ChatMessage

from codex_sdk import FlemingCodexSDK
from utils.utils import detect_smiles_in_text


class FlemingSDKInterface:
    """Adapter that exposes the codex_sdk agent through the legacy Gradio event protocol."""

    def __init__(
        self,
        config_path: str,
        model: str = "gpt-5.4",
        optimization_strategy: str = "auto",
        max_turns: int = 20,
    ) -> None:
        self.config_path = config_path
        self.model = model
        self.optimization_strategy = optimization_strategy
        self.max_turns = max_turns
        self.agent_name = "Fleming SDK"
        self.system_prompt = (
            "You are Fleming, a medicinal chemistry copilot for tuberculosis antibiotic design."
        )
        self.conversation: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.history: list[str] = []
        self._interrupt_requested = False

        self.sdk = FlemingCodexSDK(
            config_path=config_path,
            model=model,
            optimization_strategy=optimization_strategy,
        )

    def reset(self) -> None:
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        self.history = []
        self._interrupt_requested = False
        self.sdk.clear_run_state()

    def generate_image(self, smiles: str):
        return self.sdk.optimization_utils.generate_image(smiles)

    def _display_mentions_from_text(self, text: str) -> list[dict[str, str]]:
        mentioned_smiles = []
        for smiles, _, _ in detect_smiles_in_text(text or ""):
            if smiles not in mentioned_smiles:
                mentioned_smiles.append(smiles)
        if not mentioned_smiles:
            return []
        payload = self.sdk.optimization_utils.display_smiles_batch(mentioned_smiles)
        if isinstance(payload, dict):
            return payload.get("items", [])
        return []

    @staticmethod
    def _format_runtime_error(exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        if "OPENAI_API_KEY" in message or "api key" in message.lower():
            return (
                "The Codex SDK backend is configured, but no OpenAI API key is available. "
                "Set `OPENAI_API_KEY` in the environment, then retry."
            )
        if "authentication" in message.lower():
            return (
                "The Codex SDK request could not authenticate with OpenAI. "
                "Check `OPENAI_API_KEY` and any proxy settings, then retry."
            )
        return f"The Codex SDK backend failed: {message}"

    def _build_prompt(self, message: str, uploaded_csv_path: str | None = None) -> str:
        lines = [
            "You are helping the user inside the Fleming chat interface.",
            "Respond like a capable copilot: be direct, practical, and pleasant to work with.",
            "If you mention any concrete molecule, call display_smiles_batch for it.",
            "Do not rely on the interface to infer molecule mentions from prose alone.",
        ]
        if uploaded_csv_path:
            lines.append(
                f"The user uploaded a CSV at {uploaded_csv_path}. "
                "The SDK chat backend does not handle batch CSV workflows; explain that clearly."
            )
        if len(self.conversation) > 1:
            lines.append("Recent conversation:")
            for item in self.conversation[-8:]:
                role = item.get("role", "user").upper()
                content = item.get("content", "").strip()
                if content:
                    lines.append(f"{role}: {content}")
        lines.append(f"USER: {message}")
        return "\n".join(lines)

    def run_gradio_chat(
        self,
        message: str,
        history: list,
        temperature: float,
        max_new_tokens: int,
        max_token: int,
        call_agent: bool,
        conversation_state: Any,
        max_round: int = 20,
        seed: int | None = None,
        uploaded_csv_path: str | None = None,
        use_synspace: bool = False,
    ):
        if self._interrupt_requested:
            return

        self.sdk.optimization_strategy = "synspace" if use_synspace else "auto"
        self.conversation.append({"role": "user", "content": message})

        strategy = "synspace" if use_synspace else self.sdk.optimization_strategy
        yield "main", ChatMessage(role="user", content=message)
        yield (
            "plan",
            "1. Understand the user's chemistry request.\n"
            "2. Use the SDK tools for inhibition, ADMET, optimization, generation, or reasoning as needed.\n"
            "3. Display concrete molecules in the side panel when they are relevant.\n"
            "4. Return a concise recommendation with supporting details.",
        )
        yield (
            "inner",
            ChatMessage(
                role="assistant",
                content=(
                    f"[SDK] Running {self.model} with optimization strategy `{strategy}`."
                ),
            ),
        )

        prompt = self._build_prompt(message, uploaded_csv_path=uploaded_csv_path)
        try:
            result = self.sdk.run(
                prompt,
                optimization_strategy=strategy,
                max_turns=min(max_round, self.max_turns),
            )
        except Exception as exc:
            failure = self._format_runtime_error(exc)
            self.conversation.append({"role": "assistant", "content": failure})
            yield "inner", ChatMessage(role="assistant", content=f"[SDK Error] {failure}")
            yield "main", ChatMessage(role="assistant", content=f"[{self.agent_name}] {failure}")
            return

        if self._interrupt_requested:
            return

        tool_notes = self.sdk.get_run_notes()
        if tool_notes:
            for note in tool_notes:
                yield "inner", ChatMessage(role="assistant", content=f"[SDK Tool] {note}")

        final_output = getattr(result, "final_output", None) or str(result)
        displayed_items = self.sdk.get_display_items()
        if not displayed_items:
            displayed_items = self._display_mentions_from_text(final_output)
        if displayed_items:
            yield "molecules", displayed_items
            yield (
                "inner",
                ChatMessage(
                    role="assistant",
                    content=f"[SDK] Displayed {len(displayed_items)} molecule(s).",
                ),
            )

        self.conversation.append({"role": "assistant", "content": final_output})
        yield "main", ChatMessage(role="assistant", content=f"[{self.agent_name}] {final_output}")

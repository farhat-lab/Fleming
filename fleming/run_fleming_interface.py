import random
import datetime
import sys
import threading
import queue
sys.stdout.flush()
try:
    import spaces
except ImportError:
    spaces = None
import gradio as gr
import gradio_client.utils as gradio_client_utils
import os
from PIL import Image
import base64
import io
import csv
import json
from starlette.templating import Jinja2Templates

# Google OAuth (optional) – requires: pip install authlib itsdangerous uvicorn
try:
    from fastapi import FastAPI, Depends, Request as FastAPIRequest
    from starlette.config import Config as StarletteConfig
    from starlette.responses import RedirectResponse
    from starlette.middleware.sessions import SessionMiddleware
    from authlib.integrations.starlette_client import OAuth as _OAuth, OAuthError
    import uvicorn as _uvicorn
    _OAUTH_PACKAGES_AVAILABLE = True
except ImportError:
    _OAUTH_PACKAGES_AVAILABLE = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fleming")


def _patch_gradio_schema_parser():
    """
    Gradio 4.44.x can emit JSON schema fragments like `additionalProperties: true`
    for Chatbot message/file metadata, but gradio_client's parser assumes a dict.
    That breaks API metadata generation on app load for `type="messages"` chatbots.
    """
    original = getattr(gradio_client_utils, "_json_schema_to_python_type", None)
    if original is None or getattr(original, "_fleming_patched", False):
        return

    def _safe_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        if isinstance(schema, dict) and schema.get("additionalProperties") is True:
            schema = dict(schema)
            schema["additionalProperties"] = {}
        return original(schema, defs)

    _safe_json_schema_to_python_type._fleming_patched = True
    gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


_patch_gradio_schema_parser()


def _patch_starlette_template_response():
    """
    Gradio 4.44 calls `TemplateResponse(name, context)` but Starlette 1.0 expects
    `TemplateResponse(request, name, context)`. Normalize the old Gradio call style.
    """
    original = getattr(Jinja2Templates, "TemplateResponse", None)
    if original is None or getattr(original, "_fleming_patched", False):
        return

    def _compat_template_response(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            context = context or {}
            request = context.get("request")
            remaining = args[2:]
            return original(self, request, name, context, *remaining, **kwargs)
        return original(self, *args, **kwargs)

    _compat_template_response._fleming_patched = True
    Jinja2Templates.TemplateResponse = _compat_template_response


_patch_starlette_template_response()

from utils.utils import detect_smiles_in_text, is_likely_smiles
try:
    from main_fleming import Fleming
except Exception as exc:
    Fleming = None
    _LEGACY_IMPORT_ERROR = exc
else:
    _LEGACY_IMPORT_ERROR = None

try:
    from main_fleming_sdk import FlemingSDKInterface
except Exception as exc:
    FlemingSDKInterface = None
    _SDK_IMPORT_ERROR = exc
else:
    _SDK_IMPORT_ERROR = None

import pandas as pd

os.environ["VLLM_USE_V1"] = "0" # Disable v1 API for now since it does not support logits processors.

# Determine the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Set an environment variable
#HF_TOKEN = os.environ.get("HF_TOKEN", None)


# Create the image path - use absolute path for reliability
img_path = os.path.join(current_dir, 'img', 'Fleming.png')

def display_image(image_path):
    # Load and return the image
    img = Image.open(image_path)
    return img

DESCRIPTION = f'''
<div style="text-align: center;">
  <h1 style="font-size: 32px; margin-bottom: 10px;">Fleming: An AI Agent for Drug-Discovery in Tuberculosis</h1>
  <p style="font-size: 18px; opacity: 0.78; max-width: 900px; margin: 0 auto;">
    A medicinal-chemistry copilot for screening, optimization, and molecule-level reasoning.
    Ask for inhibition estimates, ADMET tradeoffs, analog ideas, or concrete next-step suggestions.
  </p>
</div>
'''
INTRO = """
This is the intro that goes here
"""

LICENSE = """
License goes here
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 8px; opacity: 0.7;">Fleming</h1>
   <p style="font-size: 18px; margin-bottom: 4px; opacity: 0.72;">Try prompts like:</p>
   <p style="font-size: 18px; margin-bottom: 4px; opacity: 0.58;">"Estimate inhibition and ADMET liabilities for CC1=CC=CC=C1"</p>
   <p style="font-size: 18px; margin-bottom: 4px; opacity: 0.58;">"Suggest close analogs with better permeability"</p>
   <p style="font-size: 18px; margin-bottom: 4px; opacity: 0.58;">"Compare Molecule A and Molecule B, then show the best candidates"</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}

#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}

.small-button button {
    font-size: 12px !important;
    padding: 4px 8px !important;
    height: 6px !important;
    width: 4px !important;
}

.gradio-accordion {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}

/* Disabled button styling */
button[disabled], button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    background-color: #f5f5f5 !important;
    color: #9ca3af !important;
    border-color: #d1d5db !important;
}

/* Make sure disabled buttons are clearly grayed out */
.gr-button[disabled] {
    opacity: 0.4 !important;
    pointer-events: none !important;
}

/* Processing animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.processing-indicator {
    animation: pulse 1.5s ease-in-out infinite;
    background: linear-gradient(45deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
}

.processing-spinner::before {
    content: "⚙️ ";
    display: inline-block;
    animation: spin 2s linear infinite;
    margin-right: 8px;
}

/* Loading dots animation */
@keyframes loading-dots {
    0%, 20% { content: "Processing"; }
    40% { content: "Processing."; }
    60% { content: "Processing.."; }
    80%, 100% { content: "Processing..."; }
}

.loading-text::after {
    content: "";
    animation: loading-dots 2s infinite;
}

/* Custom animated progress bar */
#progress-display-container .progress-wrapper {
    position: relative;
    padding-bottom: 18px; /* Space for the timer text */
}

#progress-display-container .progress-container {
    width: 100%;
    background-color: #e5e7eb;
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
    margin-top: 8px;
    margin-bottom: 4px;
}

#progress-display-container .progress-bar {
    height: 100%;
    background-color: #3b82f6;
    border-radius: 9999px;
    transition: width 0.2s ease-in-out;
    background-image: linear-gradient(
        45deg,
        rgba(255, 255, 255, 0.15) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255, 255, 255, 0.15) 50%,
        rgba(255, 255, 255, 0.15) 75%,
        transparent 75%,
        transparent
    );
    background-size: 40px 40px;
    animation: progress-bar-stripes 2s linear infinite;
}

@keyframes progress-bar-stripes {
    from {
        background-position: 40px 0;
    }
    to {
        background-position: 0 0;
    }
}

#progress-display-container .timer-text {
    position: absolute;
    bottom: 0;
    right: 0;
    font-size: 12px;
    color: #888;
}

.chat-timer {
    text-align: center !important;
    color: #888 !important;
    font-size: 14px !important;
    padding-top: 5px !important;
}
"""

chat_css = """
.gr-button { font-size: 20px !important; }  /* Enlarges button icons */
.gr-button svg { width: 32px !important; height: 32px !important; } /* Enlarges SVG icons */

/* Molecule image styling */
.chatbot img {
    max-width: 300px !important;
    max-height: 300px !important;
    border-radius: 8px !important;
    margin: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

/* SMILES code styling */
.chatbot code {
    background-color: #f0f8ff !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
    color: #2563eb !important;
}

/* Molecule gallery styling */
#molecule-gallery {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

#molecule-gallery .gallery-item {
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

#molecule-gallery img {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 4px !important;
}

/* Molecule section styling */
.chatbot .detected-molecules {
    background-color: #f8fafc !important;
    border-left: 4px solid #3b82f6 !important;
    padding: 12px !important;
    margin: 10px 0 !important;
    border-radius: 0 8px 8px 0 !important;
}


"""

model_name = ''

os.environ["TOKENIZERS_PARALLELISM"] = "false"

new_tool_files = {
    'new_tool': os.path.join(current_dir, 'data', 'new_tool.json'),
}

#config_path = "configs/fleming.yaml"
config_path = "config.yaml"

# Backends:
# - legacy: `main_fleming.Fleming` (default for this iteration)
# - sdk: `main_fleming_sdk.FlemingSDKInterface` (optional; requires extra deps/files)
fleming_real = None
fleming_sdk = None

def _get_default_backend_mode() -> str:
    mode = os.environ.get("FLEMING_BACKEND", "").strip().lower()
    if mode in {"legacy", "sdk"}:
        return mode
    # Default to legacy unless explicitly requested.
    return "legacy"

current_fleming_mode = _get_default_backend_mode()
fleming_mode_locked = True  # Track if mode has been locked

# Add a global set to track all displayed normalized SMILES
all_displayed_smiles = set()

# Persistent molecule labeling within a session
molecule_label_registry = {}
used_molecule_labels = set()
_next_label_index = 0

def _index_to_label(index: int) -> str:
    """Convert 0-based index to labels A..Z, AA..AZ, BA.. etc (Excel-style)."""
    label = ""
    while True:
        label = chr(ord('A') + (index % 26)) + label
        index = index // 26 - 1
        if index < 0:
            break
    return label

def get_or_assign_label_for_smiles(normalized_smiles: str) -> str:
    """Get an existing label for a SMILES or assign the next available label."""
    global _next_label_index
    if normalized_smiles in molecule_label_registry:
        return molecule_label_registry[normalized_smiles]
    # Find next unused label
    while True:
        candidate = _index_to_label(_next_label_index)
        _next_label_index += 1
        if candidate not in used_molecule_labels:
            molecule_label_registry[normalized_smiles] = candidate
            used_molecule_labels.add(candidate)
            return candidate

def get_current_fleming(uploaded_csv=None):
    """Get the currently selected Fleming instance"""
    global fleming_real
    global fleming_sdk
    global current_fleming_mode

    # Legacy backend is always used for CSV/batch flows.
    if uploaded_csv is not None or current_fleming_mode == "legacy":
        if fleming_real is None:
            if Fleming is None:
                raise RuntimeError(
                    "The legacy Fleming backend is unavailable in this environment. "
                    f"Import error: {_LEGACY_IMPORT_ERROR}"
                )
            fleming_real = Fleming(config_path)
        return fleming_real

    # Optional SDK backend.
    if current_fleming_mode == "sdk":
        if FlemingSDKInterface is None:
            raise RuntimeError(
                "FLEMING_BACKEND=sdk was requested but the SDK backend is unavailable. "
                f"Import error: {_SDK_IMPORT_ERROR}"
            )
        if fleming_sdk is None:
            fleming_sdk = FlemingSDKInterface(config_path)
        return fleming_sdk

    # Should never happen, but keep a safe default.
    current_fleming_mode = "legacy"
    return get_current_fleming(uploaded_csv=uploaded_csv)


# Update the global fleming variable to use the current selection
fleming = get_current_fleming()

# Add a global flag to prevent concurrent executions
_is_processing = False
_active_request_signature = None
_last_completed_request_signature = None
_last_completed_request_time = 0.0

# Initialize interruption flags only for instantiated backends.
for fleming_instance in [inst for inst in [fleming_real, fleming_sdk] if inst is not None]:
    fleming_instance._interrupt_requested = False

# Set interrupt flags for subagents
for fleming_instance in [inst for inst in [fleming_real] if inst is not None]:
    subagents = ['admet_agent', 'inhibition_agent', 'optimization_agent', 'molecular_reasoner_agent', 'generate_agent']
    for agent_name in subagents:
        if hasattr(fleming_instance, agent_name):
            agent = getattr(fleming_instance, agent_name)
            if hasattr(agent, '_interrupt_requested'):
                agent._interrupt_requested = False
            else:
                agent._interrupt_requested = False

def normalize_smiles(smiles):
    """
    Normalize SMILES string for consistent duplicate detection.
    Uses RDKit if available, otherwise returns cleaned SMILES.
    """
    if not smiles:
        return ""
    
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        
        # Suppress RDKit warnings
        RDLogger.DisableLog('rdApp.*')
        
        # Parse and canonicalize SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Keep stereochemistry so stereoisomers are treated as distinct
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # Re-enable logging
            RDLogger.EnableLog('rdApp.*')
            
            return canonical_smiles
        else:
            # Re-enable logging
            RDLogger.EnableLog('rdApp.*')
            
            # Fallback to simple normalization
            return smiles.strip().upper()
            
    except ImportError:
        # RDKit not available, use simple normalization
        return smiles.strip().upper()
    except Exception:
        # Error in RDKit processing, use simple normalization
        return smiles.strip().upper()

def extract_smiles_from_label(label):
    """
    Extract SMILES string from molecule label using various patterns.
    """
    if not label:
        return None
    
    import re
    
    # Common label patterns used in the molecule gallery
    patterns = [
        # "Name (SMILES)" or "Name (SMILES...)"
        r'\(([A-Za-z0-9\[\]()=#+\-\\\/\.@]{6,})\)',
        # "Compound: SMILES" or "Molecule: SMILES"
        r'(?:Compound|Molecule):\s*([A-Za-z0-9\[\]()=#+\-\\\/\.@]{6,})',
        # Just the SMILES if it's the whole label
        r'^([A-Za-z0-9\[\]()=#+\-\\\/\.@]{6,})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, label)
        if match:
            potential_smiles = match.group(1).strip()
            # Remove trailing "..." if present
            if potential_smiles.endswith('...'):
                potential_smiles = potential_smiles[:-3]
            
            # Validate extracted SMILES
            if is_likely_smiles(potential_smiles):
                return potential_smiles
    
    return None

def enhance_message_with_molecules(content, role="assistant", molecule_renderer=None):
    """
    Enhance a message with molecule images if SMILES are detected.
    This is purely for display purposes and doesn't affect the LLM conversation.
    Returns (enhanced_content, molecule_data_list)
    """
    detected_smiles = detect_smiles_with_context(content)
    
    if not detected_smiles:
        return content, []
    
    # Collect molecule data
    valid_molecules = []
    
    for smiles_data in detected_smiles:
        smiles = smiles_data['smiles']
        context_name = smiles_data.get('context_name', None)
        
        try:
            renderer = molecule_renderer or fleming
            image_base64 = renderer.generate_image(smiles)
            
            if image_base64 is not None:
                # Create a meaningful label
                if context_name:
                    label = f"{context_name} ({smiles[:20]}{'...' if len(smiles) > 20 else ''})"
                else:
                    # Use a better default format
                    label = f"Compound: {smiles[:20]}{'...' if len(smiles) > 20 else ''}"
                
                valid_molecules.append({
                    'smiles': smiles,
                    'image_base64': image_base64,
                    'position': smiles_data.get('position', (0, 0)),
                    'label': label
                })
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            continue
    
    return content, valid_molecules

def detect_smiles_with_context(text):
    """
    Detect SMILES in normal prose and preserve nearby chemistry labels when present.
    Returns a list of dictionaries with smiles, context_name, and position.
    """
    import re

    if not text:
        return []

    context_by_span = {}
    context_patterns = [
        r'((?:Molecule|Substructure|Analog|Candidate|Lead|Seed)\s+[A-Za-z0-9]+)\s*:\s*([A-Za-z0-9\[\]()=#+\-\\\/\.@%]{3,200})',
        r'((?:Molecule|Substructure|Analog|Candidate|Lead|Seed)\s+[A-Za-z0-9]+)\s*\(\s*([A-Za-z0-9\[\]()=#+\-\\\/\.@%]{3,200})\s*\)',
    ]
    for pattern in context_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context_name = match.group(1).strip().title()
            smiles = match.group(2).strip()
            start_pos = match.start(2)
            end_pos = start_pos + len(smiles)
            context_by_span[(start_pos, end_pos, smiles)] = context_name

    detected_molecules = []
    processed_smiles = set()

    for smiles, start_pos, end_pos in detect_smiles_in_text(text):
        if smiles in processed_smiles:
            continue
        context_name = context_by_span.get((start_pos, end_pos, smiles))
        detected_molecules.append({
            'smiles': smiles,
            'context_name': context_name,
            'position': (start_pos, end_pos)
        })
        processed_smiles.add(smiles)

    return detected_molecules



def chat_with_molecule_panel(message, history, temperature, max_new_tokens, max_tokens, call_agent, conversation, max_round, seed):
    """
    Chat function that updates both the conversation and the molecule panel.
    This function is deprecated - use custom_chat_handler instead.
    """
    print("Warning: chat_with_molecule_panel called but is deprecated - use custom_chat_handler instead")
    return history, []

def enhanced_fleming_chat(*args, **kwargs):
    """
    Wrapper function that enhances Fleming's chat responses with molecule images.
    Returns both the chat history and molecule data for side panel display.
    Only shows the most recent molecule detected.
    """
    # This function is no longer used - the custom_chat_handler handles everything directly
    # Keeping it for backward compatibility but it should not be called
    print("Warning: enhanced_fleming_chat called but is deprecated - use custom_chat_handler instead")
    return


def reset_conversation():
    """Reset the conversation history, state, and interrupt any running agents"""
    global fleming, _is_processing, current_fleming_mode, fleming_mode_locked, all_displayed_smiles
    global molecule_label_registry, used_molecule_labels, _next_label_index
    
    # Clear processing flag during reset
    _is_processing = False
    print("🧹 Cleared processing flag during reset")
    
    # Unlock Fleming mode when resetting
    fleming_mode_locked = False
    print("🔓 Fleming mode unlocked - can be changed again")
    
    # Clear the global SMILES set
    all_displayed_smiles.clear()
    print("🧹 Cleared all_displayed_smiles set during reset")

    # Clear molecule label registry
    molecule_label_registry.clear()
    used_molecule_labels.clear()
    _next_label_index = 0
    print("🧹 Cleared molecule_label_registry and reset label index")
    
    # Get the current Fleming instance
    current_fleming = get_current_fleming()
    
    try:
        # Set interruption flag to stop any running agents (with safe attribute setting)
        if hasattr(current_fleming, '_interrupt_requested'):
            current_fleming._interrupt_requested = True
        else:
            current_fleming._interrupt_requested = True
        
        # Give a brief moment for running agents to see the interruption flag
        import time
        time.sleep(0.1)
        
        # Reset Fleming's internal conversation state
        if hasattr(current_fleming, 'conversation'):
            if hasattr(current_fleming, 'system_prompt'):
                current_fleming.conversation = [{"role": "system", "content": current_fleming.system_prompt}]
            else:
                current_fleming.conversation = []
        
        if hasattr(current_fleming, 'history'):
            current_fleming.history = []
        
        # Reset Fleming's waiting state if it's waiting for user input
        if hasattr(current_fleming, '_is_waiting_for_user_input'):
            current_fleming._is_waiting_for_user_input = False
        else:
            current_fleming._is_waiting_for_user_input = False
            
        if hasattr(current_fleming, '_question_for_user_input'):
            current_fleming._question_for_user_input = None
        
        # Helper function to safely reset agent attributes
        def safe_reset_agent(agent, agent_name):
            """Safely reset an agent's attributes with error handling"""
            try:
                # Set interrupt flag
                if hasattr(agent, '_interrupt_requested'):
                    agent._interrupt_requested = True
                else:
                    agent._interrupt_requested = True
                    
                # Reset conversation
                if hasattr(agent, 'conversation'):
                    if hasattr(agent, 'system_prompt'):
                        agent.conversation = [{"role": "system", "content": agent.system_prompt}]
                    else:
                        agent.conversation = []
                        
                # Reset other attributes
                for attr, default_value in [
                    ('history', []),
                    ('_is_waiting_for_user_input', False),
                    ('_question_for_user_input', None),
                    ('parent_conversation', []),
                    ('outputs', []),
                    ('outputs_str', ''),
                    ('last_outputs', []),
                    ('function_call_messages', [])
                ]:
                    if hasattr(agent, attr):
                        setattr(agent, attr, default_value)
                
                print(f"✅ Reset {agent_name} successfully")
                
            except Exception as e:
                print(f"⚠️ Error resetting {agent_name}: {e}")
        
        # Reset subagents for both real and dummy Fleming
        if hasattr(current_fleming, '_active_subagent_calls'):
            current_fleming._active_subagent_calls.clear()
            print("🧹 Cleared active subagent calls tracking")
        
        # Reset all sub-agents
        subagents = [
            ('admet_agent', 'ADMET Agent'),
            ('inhibition_agent', 'Inhibition Agent'),
            ('optimization_agent', 'Optimization Agent'),
            ('molecular_reasoner_agent', 'Molecular Reasoner Agent'),
            ('generate_agent', 'Generate Agent')
        ]
        
        for agent_attr, agent_name in subagents:
            if hasattr(current_fleming, agent_attr):
                safe_reset_agent(getattr(current_fleming, agent_attr), agent_name)
        
        # Reset Fleming's own state variables
        for attr, default_value in [
            ('outputs', []),
            ('outputs_str', ''),
            ('last_outputs', []),
            ('function_call_messages', [])
        ]:
            if hasattr(current_fleming, attr):
                setattr(current_fleming, attr, default_value)
        
        # Clear the interruption flag after reset
        current_fleming._interrupt_requested = False
        
        # Clear subagent interruption flags
        for agent_attr, _ in subagents:
            if hasattr(current_fleming, agent_attr):
                agent = getattr(current_fleming, agent_attr)
                if hasattr(agent, '_interrupt_requested'):
                    agent._interrupt_requested = False
        
        mode_name = "Fleming" if current_fleming_mode == "real" else "Fleming"
        print(f"🔄 {mode_name} reset successfully")
        
    except Exception as e:
        print(f"Warning: Error during reset: {e}")
        import traceback
        traceback.print_exc()
    
    # Return empty chatbot history, empty conversation state, empty molecules, re-enabled buttons
    return [], [], [], gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="🗑️ Reset Conversation")

conversation_state = gr.State([])

# Gradio block
disclaimer_text = """
**Disclaimer:** Fleming is a living tool in active development.
"""
with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    # --- Main Application Screen ---
    with gr.Column(visible=True) as main_app_group:
        progress_display = gr.HTML("<div class='progress-wrapper'><div class='progress-container'><div class='progress-bar'></div></div><div class='timer-text'></div></div>", visible=False, elem_id="progress-display-container")

        with gr.Accordion("Current Plan & Status", open=True, visible=False) as plan_accordion:
            plan_display = gr.Markdown("Waiting for plan...")

        with gr.Row():
            with gr.Column(scale=9):
                msg_input = gr.Textbox(
                    placeholder="Ask Fleming to analyze, compare, optimize, or generate TB-focused molecules...",
                    show_label=False,
                    scale=1
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=1):
                reset_btn = gr.Button("🗑️ Reset", variant="secondary", scale=1)

        gr.Markdown("""On the **left** is the working conversation with Fleming. In the **middle** you can inspect tool use and intermediate reasoning. On the **right** any molecules Fleming highlights will be rendered so you can compare structures visually.""")
        gr.Markdown("For batch ranking, upload a CSV with a single `SMILES` column.")
        csv_upload = gr.File(label="Upload SMILES CSV", file_types=[".csv"])
        use_synspace_checkbox = gr.Checkbox(label="Use Synspace for Neighbors", value=False)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=400, 
                    placeholder="Fleming will answer here with recommendations, tradeoffs, and next-step suggestions.",
                    label='Main Conversation'
                )
            
            with gr.Column(scale=3):
                inner_chatbot = gr.Chatbot(
                    height=400,
                    placeholder="Optional audit trail: tool calls, intermediate checks, and molecule-display events appear here.",
                    label="Fleming's Inner Thoughts (Optional)"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Detected Molecules")
                gr.Markdown("*New molecules appear first*")
                molecule_gallery = gr.Gallery(
                    label="Molecule Structures",
                    show_label=False,
                    elem_id="molecule-gallery",
                    columns=1,
                    rows=3,
                    height=400,
                    object_fit="contain",
                    value=[]
                )

        with gr.Accordion("", open=True):
            gr.Markdown("""**Limitations:** Fleming is a research product. Its main capabilities are built around predicting molecular properties (inhibition, ADMET, druglikeness, etc.). While Fleming has advanced literature review capabilities to supplement its main tools, there are certain areas where it does not perform well. Notably, Fleming is not scoped to understand the mechanism of action of molecules.""")

        with gr.Accordion("Batch Processing", open=False):
            batch_summary_display = gr.Markdown("Batch processing status will appear here.")
            batch_dataframe_display = gr.DataFrame(visible=False, label="Batch Results")
            download_csv_button = gr.File(label="Download Ranked CSV", visible=False)

    gr.Markdown(LICENSE)

    # ------------------
    # All Event Handlers
    # ------------------

    # Helper to hard-interrupt and clear all agents/sub-agents without changing UI
    def hard_interrupt_all_agents():
        global _is_processing
        try:
            _is_processing = False
            import time
            time.sleep(0.1)

            subagent_names = [
                'admet_agent', 'inhibition_agent', 'optimization_agent',
                'molecular_reasoner_agent', 'generate_agent'
            ]

            for fleming_instance in [inst for inst in [fleming_real, fleming_sdk] if inst is not None]:
                setattr(fleming_instance, '_interrupt_requested', True)
                if hasattr(fleming_instance, '_active_subagent_calls'):
                    try:
                        fleming_instance._active_subagent_calls.clear()
                    except Exception:
                        pass
                if hasattr(fleming_instance, 'conversation'):
                    if hasattr(fleming_instance, 'system_prompt'):
                        fleming_instance.conversation = [{"role": "system", "content": fleming_instance.system_prompt}]
                    else:
                        fleming_instance.conversation = []
                if hasattr(fleming_instance, 'history'):
                    fleming_instance.history = []
                for attr in ['outputs', 'outputs_str', 'last_outputs', 'function_call_messages']:
                    if hasattr(fleming_instance, attr):
                        setattr(fleming_instance, attr, [] if attr != 'outputs_str' else '')
                if hasattr(fleming_instance, '_is_waiting_for_user_input'):
                    fleming_instance._is_waiting_for_user_input = False
                if hasattr(fleming_instance, '_question_for_user_input'):
                    fleming_instance._question_for_user_input = None

                for agent_attr in subagent_names:
                    if hasattr(fleming_instance, agent_attr):
                        agent = getattr(fleming_instance, agent_attr)
                        setattr(agent, '_interrupt_requested', True)
                        if hasattr(agent, 'conversation'):
                            if hasattr(agent, 'system_prompt'):
                                agent.conversation = [{"role": "system", "content": agent.system_prompt}]
                            else:
                                agent.conversation = []
                        if hasattr(agent, 'history'):
                            agent.history = []
                        if hasattr(agent, '_is_waiting_for_user_input'):
                            agent._is_waiting_for_user_input = False

                if hasattr(fleming_instance, 'reset'):
                    try:
                        fleming_instance.reset()
                    except Exception:
                        pass

            # brief pause to propagate interrupt flags
            time.sleep(0.1)

        except Exception as e:
            print(f"⚠️ Error during hard interrupt: {e}")

    # --- Handler Functions ---
    def custom_chat_handler(message, history, inner_history, current_gallery, uploaded_csv, use_synspace):
        global _is_processing, all_displayed_smiles
        global _active_request_signature, _last_completed_request_signature, _last_completed_request_time
        import time
        start_time = time.time()
        
        current_fleming = get_current_fleming(uploaded_csv)
        uploaded_csv_name = getattr(uploaded_csv, "name", None) if uploaded_csv else None
        request_signature = (message.strip(), uploaded_csv_name, bool(use_synspace))
        
        # State for batch components
        batch_summary = "Batch processing status will appear here."
        batch_df_value = None
        batch_df_visible = False
        download_path = None
        download_visible = False
        
        # State for plan
        current_plan = "Waiting for plan..."
        plan_visible = False

        if _is_processing:
            if request_signature == _active_request_signature:
                print("🚫 Duplicate request ignored while current request is still processing.")
                yield (
                    history, inner_history, current_gallery,
                    gr.update(interactive=False, value="Processing..."), gr.update(interactive=False, value="🚫 Reset Disabled"),
                    gr.update(visible=True),
                    batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                    gr.update(value=download_path, visible=download_visible),
                    gr.update(value=current_plan, visible=plan_visible)
                )
                return

            print("🚫 New request arrived while processing, interrupting previous request...")
            if hasattr(current_fleming, '_interrupt_requested'):
                current_fleming._interrupt_requested = True
            else:
                current_fleming._interrupt_requested = True

            time.sleep(0.1)
            _is_processing = False
        
        if (
            request_signature == _last_completed_request_signature
            and (time.time() - _last_completed_request_time) < 3
        ):
            print("🚫 Duplicate request ignored immediately after completion.")
            yield (
                history, inner_history, current_gallery,
                gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="🗑️ Reset Conversation"),
                gr.update(visible=False),
                batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                gr.update(value=download_path, visible=download_visible),
                gr.update(value=current_plan, visible=plan_visible)
            )
            return
        
        if not message.strip() and not getattr(current_fleming, '_is_waiting_for_user_input', False):
            # Return current state if message is empty
            yield (
                history, inner_history, current_gallery, 
                gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="🗑️ Reset Conversation"), 
                gr.update(visible=False),
                batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                gr.update(value=download_path, visible=download_visible),
                gr.update(value=current_plan, visible=plan_visible)
            )
            return

        _is_processing = True
        _active_request_signature = request_signature
        mode_name = "Fleming SDK" if current_fleming is fleming_sdk else "Fleming"
        print(f"🔄 Starting request processing with {mode_name}")
        
        main_chat_history = list(history)
        inner_chat_history = list(inner_history)
        accumulated_molecules = list(current_gallery) if current_gallery is not None else []
        

        html_content = "<div class='progress-wrapper'><div class='progress-container'><div class='progress-bar' style='width: 0%;'></div></div><div class='timer-text'>0s / 300s</div></div>"
        yield (
            main_chat_history, inner_chat_history, accumulated_molecules, 
            gr.update(interactive=False, value="Processing..."), gr.update(interactive=False, value="🚫 Reset Disabled"), 
            gr.update(visible=True, value=html_content),
            batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
            gr.update(value=download_path, visible=download_visible),
            gr.update(value=current_plan, visible=plan_visible)
        )

        result_queue = queue.Queue()

        def agent_thread_func():
            try:
                if hasattr(current_fleming, '_interrupt_requested'):
                    current_fleming._interrupt_requested = False

                agent_generator = current_fleming.run_gradio_chat(
                    message, main_chat_history, 0.3, 1024, 81920, False, conversation_state, 30, 100, uploaded_csv_path=uploaded_csv.name if uploaded_csv else None, use_synspace=use_synspace
                )

                for output in agent_generator:
                    if not _is_processing:
                        print("🚫 Processing interrupted, stopping agent generator")
                        break
                    result_queue.put(output)
            except Exception as e:
                result_queue.put(e)
            finally:
                result_queue.put(None)  # Sentinel value

        thread = threading.Thread(target=agent_thread_func)
        thread.start()

        try:
            while thread.is_alive() or not result_queue.empty():
                try:
                    output = result_queue.get_nowait()

                    if output is None:
                        break
                    
                    if isinstance(output, Exception):
                        raise output
                
                    panel, *data = output
                    
                    if panel == 'main':
                        main_chat_history.append(data[0])
                    elif panel == 'inner':
                        inner_chat_history.append(data[0])
                    elif panel == 'molecules':
                        try:
                            items = data[0] if data else []
                            new_images = []
                            for item in items:
                                smiles = item.get('smiles')
                                if not smiles:
                                    continue
                                normalized_smiles = normalize_smiles(smiles)
                                if normalized_smiles and normalized_smiles not in all_displayed_smiles:
                                    # Assign persistent label per SMILES
                                    assigned_label = get_or_assign_label_for_smiles(normalized_smiles)
                                    image_base64 = current_fleming.generate_image(smiles)
                                    if image_base64 is None:
                                        continue
                                    try:
                                        image_bytes = base64.b64decode(image_base64)
                                        image = Image.open(io.BytesIO(image_bytes))
                                        label = f"Molecule {assigned_label}: {smiles[:20]}{'...' if len(smiles) > 20 else ''}"
                                        new_images.append((image, label))
                                        all_displayed_smiles.add(normalized_smiles)
                                    except Exception as e:
                                        print(f"Error processing structured molecule image: {e}")
                                else:
                                    if normalized_smiles:
                                        print(f"🔄 Skipping duplicate molecule (global): {smiles}")
                            if new_images:
                                accumulated_molecules = new_images + accumulated_molecules
                                if len(accumulated_molecules) > 20:
                                    accumulated_molecules = accumulated_molecules[:20]
                        except Exception as e:
                            print(f"Error handling molecules payload: {e}")
                            continue
                    elif panel == 'batch':
                        df_update, summary_update = data
                        batch_summary = summary_update
                        batch_df_value = df_update
                        batch_df_visible = True
                    elif panel == 'download':
                        download_path = data[0]
                        download_visible = True
                    elif panel == 'plan':
                        current_plan = data[0]
                        plan_visible = True
                     
                    if panel in ['main', 'inner']:
                        content = data[0].content if hasattr(data[0], 'content') else str(data[0])
                        _, molecules = enhance_message_with_molecules(content, molecule_renderer=current_fleming)
                        if molecules:
                            new_images = []
                            for mol_data in molecules:
                                normalized_smiles = normalize_smiles(mol_data['smiles'])
                                if normalized_smiles and normalized_smiles not in all_displayed_smiles:
                                    try:
                                        image_bytes = base64.b64decode(mol_data['image_base64'])
                                        image = Image.open(io.BytesIO(image_bytes))
                                        label = mol_data.get('label', f"Compound: {mol_data['smiles'][:20]}...")
                                        new_images.append((image, label))
                                        all_displayed_smiles.add(normalized_smiles)
                                    except Exception as e:
                                        print(f"Error processing image for display: {e}")
                            if new_images:
                                accumulated_molecules = new_images + accumulated_molecules
                                if len(accumulated_molecules) > 20:
                                    accumulated_molecules = accumulated_molecules[:20]
                except queue.Empty:
                    pass # No new data, just update timer

                elapsed_secs = int(time.time() - start_time)
                new_max = max(300, elapsed_secs + 5)

                progress_percentage = min((elapsed_secs / new_max) * 100, 100)
                timer_text = f"{elapsed_secs}s / {new_max}s"
                html_content = f"<div class='progress-wrapper'><div class='progress-container'><div class='progress-bar' style='width: {progress_percentage}%;'></div></div><div class='timer-text'>{timer_text}</div></div>"


                yield (
                    main_chat_history, inner_chat_history, accumulated_molecules, 
                    gr.update(interactive=False, value="Processing..."), gr.update(interactive=False, value="🚫 Reset Disabled"), 
                    gr.update(visible=True, value=html_content),
                    batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                    gr.update(value=download_path, visible=download_visible),
                    gr.update(value=current_plan, visible=plan_visible)
                )
                
                time.sleep(1)
        except Exception as e:
            print(f"Error in chat handler: {e}")
            import traceback
            traceback.print_exc()
            inner_chat_history.append(gr.ChatMessage(role="system", content=f"An error occurred: {e}"))
        
        finally:
            _is_processing = False
            _last_completed_request_signature = request_signature
            _last_completed_request_time = time.time()
            _active_request_signature = None
            print("✅ Request processing completed")
            if hasattr(current_fleming, '_interrupt_requested'):
                current_fleming._interrupt_requested = False
            
            yield (
                main_chat_history, inner_chat_history, accumulated_molecules, 
                gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="🗑️ Reset Conversation"), 
                gr.update(visible=False),
                batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                gr.update(value=download_path, visible=download_visible),
                gr.update(value=current_plan, visible=plan_visible)
            )

    # --- Event Listeners ---
    send_event = send_btn.click(
        custom_chat_handler,
        inputs=[msg_input, chatbot, inner_chatbot, molecule_gallery, csv_upload, use_synspace_checkbox],
        outputs=[chatbot, inner_chatbot, molecule_gallery, send_btn, reset_btn, progress_display, batch_summary_display, batch_dataframe_display, download_csv_button, plan_display],
    ).then(lambda: (None, ""), outputs=[csv_upload, msg_input])
    
    submit_event = msg_input.submit(
        custom_chat_handler,
        inputs=[msg_input, chatbot, inner_chatbot, molecule_gallery, csv_upload, use_synspace_checkbox],
        outputs=[chatbot, inner_chatbot, molecule_gallery, send_btn, reset_btn, progress_display, batch_summary_display, batch_dataframe_display, download_csv_button, plan_display],
    ).then(lambda: (None, ""), outputs=[csv_upload, msg_input])

    reset_btn.click(
        reset_conversation,
        outputs=[chatbot, conversation_state, molecule_gallery, send_btn, reset_btn],
        cancels=[send_event, submit_event]
    ).then(lambda: ([], "Batch processing status will appear here.", gr.update(value=None, visible=False), gr.update(value=None, visible=False), gr.update(value="Waiting for plan...", visible=False)), outputs=[inner_chatbot, batch_summary_display, batch_dataframe_display, download_csv_button, plan_display])

AUTH0_DOMAIN        = os.environ.get("AUTH0_DOMAIN", "")
AUTH0_CLIENT_ID     = os.environ.get("AUTH0_CLIENT_ID", "")
AUTH0_CLIENT_SECRET = os.environ.get("AUTH0_CLIENT_SECRET", "")
AUTH0_SECRET_KEY    = os.environ.get("AUTH0_SECRET_KEY", "")

USE_AUTH0 = (
    _OAUTH_PACKAGES_AVAILABLE
    and bool(AUTH0_DOMAIN)
    and bool(AUTH0_CLIENT_ID)
    and bool(AUTH0_CLIENT_SECRET)
    and bool(AUTH0_SECRET_KEY)
)

if USE_AUTH0:
    import datetime as _dt

    _fastapi_app = FastAPI()


    _fastapi_app.add_middleware(SessionMiddleware, secret_key=AUTH0_SECRET_KEY)

    # ---- Auth0 OIDC client ---------------------------------------------------
    _starlette_cfg = StarletteConfig(environ={
        'AUTH0_CLIENT_ID':     AUTH0_CLIENT_ID,
        'AUTH0_CLIENT_SECRET': AUTH0_CLIENT_SECRET,
    })
    _oauth_client = _OAuth(_starlette_cfg)
    _oauth_client.register(
        name='auth0',
        server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

    _LOG_DIR  = os.path.join(current_dir, 'logs')
    _LOG_FILE = os.path.join(_LOG_DIR, 'user_activity.log')
    os.makedirs(_LOG_DIR, exist_ok=True)

    @_fastapi_app.middleware('http')
    async def _track_user_activity(request: FastAPIRequest, call_next):
        """Log authenticated requests to /gradio for usage auditing."""
        response = await call_next(request)
        if request.url.path.startswith('/gradio'):
            user_info = request.session.get('user', {})
            username  = user_info.get('name') or user_info.get('email') or 'anonymous'
            ip        = request.client.host if request.client else 'unknown'
            timestamp = _dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            method    = request.method
            path      = request.url.path
            status    = response.status_code
            line      = f"{timestamp} | {username:40s} | {ip:15s} | {method} {path} | {status}\n"
            try:
                with open(_LOG_FILE, 'a') as _lf:
                    _lf.write(line)
            except Exception:
                pass  
        return response

    def _get_current_user(request: FastAPIRequest):
        """Return the logged-in user's name/email, or None if not authenticated."""
        user = request.session.get('user')
        if user:
            return user.get('name') or user.get('email')
        return None

    @_fastapi_app.get('/')
    def _root(user: str = Depends(_get_current_user)):
        return RedirectResponse(url='/gradio' if user else '/login-demo')

    @_fastapi_app.route('/logout')
    async def _logout(request: FastAPIRequest):
        request.session.pop('user', None)
        return_to = str(request.base_url)
        return RedirectResponse(
            url=f'https://{AUTH0_DOMAIN}/v2/logout?returnTo={return_to}&client_id={AUTH0_CLIENT_ID}'
        )

    @_fastapi_app.route('/login')
    async def _login(request: FastAPIRequest):
        redirect_uri = request.url_for('_oauth_callback')
        return await _oauth_client.auth0.authorize_redirect(request, redirect_uri)

    @_fastapi_app.route('/auth')
    async def _oauth_callback(request: FastAPIRequest):
        try:
            token = await _oauth_client.auth0.authorize_access_token(request)
        except OAuthError:
            return RedirectResponse(url='/')
        request.session['user'] = dict(token).get('userinfo', {})
        return RedirectResponse(url='/')
    with gr.Blocks(title="Fleming – Sign In") as _login_demo:
        gr.Markdown(
            "## Fleming\n\n"
            "**An AI Agent for Drug Discovery in Tuberculosis**\n\n"
            "Sign in with your institutional account to continue."
        )
        gr.Button("Sign in with Auth0", link="/login", variant="primary")

    _fastapi_app = gr.mount_gradio_app(_fastapi_app, _login_demo, path="/login-demo")
    _fastapi_app = gr.mount_gradio_app(_fastapi_app, demo, path="/gradio", auth_dependency=_get_current_user)


if __name__ == "__main__":
    if USE_AUTH0:
        port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
        print(f"Starting Fleming with Auth0 at http://0.0.0.0:{port}", flush=True)
        print(f"  Main app  : http://localhost:{port}/gradio", flush=True)
        print(f"  Login page: http://localhost:{port}/login-demo", flush=True)
        print(f"  User log  : {_LOG_FILE}", flush=True)
        _uvicorn.run(_fastapi_app, host="0.0.0.0", port=port)
    else:
        if not _OAUTH_PACKAGES_AVAILABLE:
            print(
                "Note: Auth0 packages unavailable. "
                "Install with: pip install authlib itsdangerous uvicorn",
                flush=True,
            )
        else:
            print(
                "Note: Auth0 disabled. "
                "Set AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_SECRET_KEY to enable.",
                flush=True,
            )
        _, local_url, share_url = demo.launch(share=True, server_name="0.0.0.0", css=css)
        print("Gradio local URL:", local_url, flush=True)
        print("Gradio share URL:", share_url, flush=True)

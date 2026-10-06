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

/* Disabled button styling - uses theme variables so it reads correctly in both light and dark mode */
button[disabled], button:disabled {
    opacity: 0.7 !important;
    cursor: not-allowed !important;
    background: var(--background-fill-secondary) !important;
    color: var(--body-text-color-subdued) !important;
    border-color: var(--border-color-primary) !important;
}

/* Make sure disabled buttons are clearly grayed out */
.gr-button[disabled] {
    opacity: 0.7 !important;
    pointer-events: none !important;
}

/* --- Mode selection screen --- */
.mode-card {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    padding: 24px !important;
    background: var(--background-fill-secondary) !important;
    transition: border-color 0.15s ease;
}

.mode-card:hover {
    border-color: #3b82f6 !important;
}

.mode-card h3 {
    margin-top: 0 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

.back-button button {
    font-size: 13px !important;
    opacity: 0.8;
}

/* Chat input row - anchored to the bottom of the conversation column and highlighted
   so it reads clearly as the primary place to start interacting with Fleming. */
.chat-input-row {
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    margin-top: 8px !important;
    background: var(--background-fill-secondary) !important;
}

.chat-input-row textarea, .chat-input-row input {
    border: none !important;
    background: transparent !important;
}
"""

css += """

/* Custom animated progress bar */
#progress-display-container .progress-wrapper {
    position: relative;
    padding-bottom: 18px; /* Space for the timer text */
}

#progress-display-container .progress-container {
    width: 100%;
    background-color: var(--border-color-primary);
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
        rgba(255, 255, 255, 0.1) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255, 255, 255, 0.1) 50%,
        rgba(255, 255, 255, 0.1) 75%,
        transparent 75%,
        transparent
    );
    background-size: 32px 32px;
    animation: progress-bar-stripes 3.5s linear infinite;
}

@keyframes progress-bar-stripes {
    from {
        background-position: 32px 0;
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
    color: var(--body-text-color-subdued);
}

.chat-timer {
    text-align: center !important;
    color: var(--body-text-color-subdued) !important;
    font-size: 14px !important;
    padding-top: 5px !important;
}

/* Plan & status panel (middle column, above the progress trace) - theme-aware so it stays readable in dark mode */
#plan-display-container {
    background-color: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-left: 4px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin-bottom: 10px !important;
    color: var(--body-text-color) !important;
    max-height: 220px;
    overflow-y: auto;
}

#plan-display-container * {
    color: inherit !important;
}

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
    background-color: var(--background-fill-secondary) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
    color: #2563eb !important;
}

.dark .chatbot code {
    color: #60a5fa !important;
}

/* Molecule gallery styling - theme-aware container; structure images themselves keep their white
   drawing canvas from RDKit, which reads fine on either a light or dark card background. */
#molecule-gallery {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

#molecule-gallery .gallery-item {
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
    background: #ffffff !important;
}

#molecule-gallery img {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 4px !important;
}

/* Molecule section styling */
.chatbot .detected-molecules {
    background-color: var(--background-fill-secondary) !important;
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
# Batch runs do not use _is_processing (that belongs to the chat handler),
# so Stop/Hard Reset signal the batch drain loop through its own flag.
_batch_stop_requested = False
# Plausibility costs one LLM call per molecule, so the batch screen caps it.
PLAUSIBILITY_MAX_N = 100
_active_request_signature = None
_last_completed_request_signature = None
_last_completed_request_time = 0.0
# Monotonically increasing counter identifying the "current" request. Used so that a stale
# (already-interrupted) request's cleanup code can never clobber the state of a newer request
# that has since started - see custom_chat_handler / agent_thread_func below.
_request_generation = 0

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


CHAT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs")


def log_chat_event(kind, payload):
    """Append one event to today's chat log.

    One JSONL file per day, one JSON object per line, appended and flushed as it happens -
    so a session that crashes or is killed still leaves everything up to that point on
    disk. Logging never raises: a failure here must not take down a conversation.
    """
    try:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)
        path = os.path.join(
            CHAT_LOG_DIR,
            f"chat_{datetime.datetime.now().strftime('%Y-%m-%d')}.jsonl")
        record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "mode": current_fleming_mode,
            **payload,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:                                       # noqa: BLE001
        print(f"chat log write failed ({exc!r}) - continuing")


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
    return [], [], [], gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="Reset")

conversation_state = gr.State([])

# Gradio block
disclaimer_text = """
**Disclaimer:** Fleming is a living tool in active development.
"""
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown(DESCRIPTION)

    # --- Mode Selection Screen ---
    with gr.Column(visible=True) as mode_selection_group:
        gr.Markdown("## How would you like to work with Fleming today?")
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes=["mode-card"]):
                gr.Markdown("### Batch Mode")
                gr.Markdown(
                    "Upload a CSV of SMILES strings for deterministic, large-scale ranking, "
                    "optimization, and plausibility screening — no chat required. Best for "
                    "**10 or more molecules** at once. Batch mode was used to generate the antibiotic candidates that were validated in vitro"
                )
                batch_mode_btn = gr.Button("Enter Batch Mode", variant="primary", size="lg")
            with gr.Column(elem_classes=["mode-card"]):
                gr.Markdown("### Copilot Mode")
                gr.Markdown(
                    "Chat with Fleming directly. Ask for inhibition estimates, ADMET tradeoffs, "
                    "analog ideas, optimization, or molecule generation, and watch its plan and "
                    "reasoning as it works. Best for **fewer than 10 molecules** at a time."
                )
                copilot_mode_btn = gr.Button("Enter Copilot Mode", variant="primary", size="lg")

    # --- Copilot (Chat) Screen ---
    with gr.Column(visible=False) as copilot_group:
        back_to_modes_btn = gr.Button("Back to mode selection", size="sm", elem_classes=["back-button"])
        pending_message_buffer = gr.Textbox(value="", visible=False, show_label=False)
        no_csv_state = gr.State(None)

        gr.Markdown("""On the **left** is the working conversation with Fleming - type your question in the box at the bottom of that column to get started. In the **middle** you can review its plan and inspect tool use and intermediate reasoning as it works. On the **right** any molecules Fleming highlights will be rendered so you can compare structures visually.""")
        use_synspace_checkbox = gr.Checkbox(label="Use Synspace for Neighbors", value=False)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=400, 
                    placeholder="Fleming will answer here with recommendations, tradeoffs, and next-step suggestions.",
                    label='Main Conversation',
                    type="messages"
                )
                with gr.Row(elem_classes=["chat-input-row"]):
                    msg_input = gr.Textbox(
                        placeholder="Start here: ask Fleming to analyze, compare, optimize, or generate TB-focused molecules...",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                    stop_btn = gr.Button("Stop", variant="stop", scale=0, min_width=70)
                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary", size="sm")
                    hard_reset_btn = gr.Button("Hard Reset", variant="secondary", size="sm")

            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("#### Plan")
                    plan_display = gr.Markdown(
                        "Waiting for plan...",
                        visible=False,
                        elem_id="plan-display-container",
                    )
                with gr.Group():
                    gr.Markdown("#### Progress & Reasoning")
                    progress_display = gr.HTML(
                        "<div class='progress-wrapper'><div class='progress-container'><div class='progress-bar'></div></div><div class='timer-text'></div></div>",
                        visible=False,
                        elem_id="progress-display-container"
                    )
                    inner_chatbot = gr.Chatbot(
                        height=340,
                        placeholder="Optional audit trail: tool calls, intermediate checks, and molecule-display events appear here.",
                        label="Tool Calls & Reasoning",
                        type="messages",
                        show_label=False,
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("### Relevant Molecules")
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

    # --- Batch Mode Screen ---
    with gr.Column(visible=False) as batch_group:
        back_to_modes_btn_batch = gr.Button("Back to mode selection", size="sm", elem_classes=["back-button"])
        gr.Markdown("## Batch Mode")
        gr.Markdown(
            "Upload a CSV of molecules. Fleming will deterministically rank "
            "molecules by predicted TB inhibition and ADMET, optionally generate optimized analogs "
            "for the top candidates, and run a plausibility check on the best results — this pipeline "
            "runs directly without an LLM driving each step."
        )
        with gr.Group():
            gr.Markdown(
                "**CSV requirements:**\n"
                "- Must contain a column named exactly `SMILES` (case-sensitive). Any other columns are ignored.\n"
                "- One valid SMILES string per row.\n"
                "- File must be `.csv` format, comma-separated, with a header row.\n"
                "- Duplicate SMILES are automatically removed. Rows with invalid/unparseable SMILES may fail "
                "during scoring, so please double-check your SMILES are valid before uploading.\n\n"
                "Example:\n"
                "```\n"
                "SMILES\n"
                "CCO\n"
                "c1ccccc1\n"
                "CC(=O)Oc1ccccc1C(=O)O\n"
                "```"
            )
        batch_csv_upload = gr.File(label="Upload SMILES CSV", file_types=[".csv"])
        batch_steps_checkboxes = gr.CheckboxGroup(
            choices=["Ranking (Inhibition + ADMET)", "Optimization", "Plausibility Check"],
            value=["Ranking (Inhibition + ADMET)", "Optimization", "Plausibility Check"],
            label="Pipeline steps to run",
        )
        with gr.Row():
            batch_top_n_optimize = gr.Number(value=10, precision=0, minimum=1, label="Top N molecules to optimize")
            batch_top_n_plausibility = gr.Number(
                value=10, precision=0, minimum=1, maximum=PLAUSIBILITY_MAX_N,
                label=f"Top N molecules for plausibility check (max {PLAUSIBILITY_MAX_N})")
        with gr.Row():
            run_batch_btn = gr.Button("Run Batch Analysis", variant="primary", size="lg")
            stop_btn_batch = gr.Button("Stop", variant="stop", scale=0, min_width=90)
        hard_reset_btn_batch = gr.Button("Hard Reset", variant="secondary", size="sm")
        batch_progress_display = gr.HTML(
            "<div class='progress-wrapper'><div class='progress-container'><div class='progress-bar'></div></div><div class='timer-text'></div></div>",
            visible=False,
            elem_id="progress-display-container"
        )
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
        global _request_generation
        import time
        start_time = time.time()
        max_inner_history = 80
        generator_closing = False
        
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
                    gr.update(interactive=False, value="Processing..."), gr.update(interactive=False, value="Reset"),
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
                gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="Reset"),
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
                gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="Reset"), 
                gr.update(visible=False),
                batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                gr.update(value=download_path, visible=download_visible),
                gr.update(value=current_plan, visible=plan_visible)
            )
            return

        _is_processing = True
        _active_request_signature = request_signature
        _request_generation += 1
        my_generation = _request_generation
        mode_name = "Fleming SDK" if current_fleming is fleming_sdk else "Fleming"
        print(f"🔄 Starting request processing with {mode_name}")
        log_chat_event("user_message", {"backend": mode_name, "message": message,
                                        "generation": my_generation})
        
        main_chat_history = list(history)
        inner_chat_history = list(inner_history)
        accumulated_molecules = list(current_gallery) if current_gallery is not None else []

        def _append_unique_chat_message(chat_history, chat_message):
            if not isinstance(chat_message, gr.ChatMessage):
                chat_history.append(chat_message)
                return True
            if chat_history:
                last_item = chat_history[-1]
                if (
                    isinstance(last_item, gr.ChatMessage)
                    and getattr(last_item, "role", None) == getattr(chat_message, "role", None)
                    and getattr(last_item, "content", None) == getattr(chat_message, "content", None)
                ):
                    return False
            chat_history.append(chat_message)
            return True

        def _normalize_inner_chat_message(chat_message):
            if not isinstance(chat_message, gr.ChatMessage):
                return gr.ChatMessage(role="assistant", content=str(chat_message))
            content = getattr(chat_message, "content", "")
            role = getattr(chat_message, "role", "assistant")
            if role != "assistant":
                role = "assistant"
            return gr.ChatMessage(role=role, content=content)

        def _trim_inner_history():
            nonlocal inner_chat_history
            if len(inner_chat_history) > max_inner_history:
                inner_chat_history = inner_chat_history[-max_inner_history:]

        def _build_progress_html(elapsed_secs):
            # Indeterminate bar (we don't know total task duration) - the CSS stripe animation
            # conveys "in progress"; only the elapsed time is shown, with no artificial countdown.
            timer_text = f"Working... {elapsed_secs}s elapsed"
            return (
                f"<div class='progress-wrapper'><div class='progress-container'>"
                f"<div class='progress-bar' style='width: 100%;'></div></div>"
                f"<div class='timer-text'>{timer_text}</div></div>"
            )

        def _ui_payload(progress_value):
            return (
                main_chat_history, inner_chat_history, accumulated_molecules,
                gr.update(interactive=False, value="Processing..."), gr.update(interactive=False, value="Reset"),
                gr.update(visible=True, value=progress_value),
                batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                gr.update(value=download_path, visible=download_visible),
                gr.update(value=current_plan, visible=plan_visible)
            )
        
        if message and message.strip():
            _append_unique_chat_message(main_chat_history, gr.ChatMessage(role="user", content=message.strip()))
        _append_unique_chat_message(
            inner_chat_history,
            gr.ChatMessage(role="assistant", content="[Fleming] Starting analysis and building plan...")
        )
        _trim_inner_history()

        html_content = _build_progress_html(0)
        yield _ui_payload(html_content)

        result_queue = queue.Queue()

        def agent_thread_func():
            agent_generator = None
            try:
                if hasattr(current_fleming, '_interrupt_requested'):
                    current_fleming._interrupt_requested = False

                agent_generator = current_fleming.run_gradio_chat(
                    message, main_chat_history, 0.3, 1024, 81920, False, conversation_state, 30, 100, uploaded_csv_path=uploaded_csv.name if uploaded_csv else None, use_synspace=use_synspace
                )

                for output in agent_generator:
                    if not _is_processing or _request_generation != my_generation:
                        print(f"🚫 Processing interrupted, stopping agent generator (generation {my_generation}, current {_request_generation})")
                        break
                    result_queue.put(output)
            except Exception as e:
                result_queue.put(e)
            finally:
                if agent_generator is not None:
                    try:
                        agent_generator.close()
                    except Exception:
                        pass
                result_queue.put(None)  # Sentinel value

        thread = threading.Thread(target=agent_thread_func)
        thread.start()

        try:
            last_progress_bucket = -1
            last_render_signature = None
            while thread.is_alive() or not result_queue.empty():
                state_changed = False
                try:
                    output = result_queue.get_nowait()

                    if output is None:
                        break
                    
                    if isinstance(output, Exception):
                        raise output
                
                    panel, *data = output
                    
                    if panel == 'main':
                        state_changed = _append_unique_chat_message(main_chat_history, data[0]) or state_changed
                    elif panel == 'inner':
                        state_changed = _append_unique_chat_message(inner_chat_history, _normalize_inner_chat_message(data[0])) or state_changed
                        _trim_inner_history()
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
                                state_changed = True
                        except Exception as e:
                            print(f"Error handling molecules payload: {e}")
                            continue
                    elif panel == 'batch':
                        df_update, summary_update = data
                        batch_summary = summary_update
                        batch_df_value = df_update
                        batch_df_visible = True
                        state_changed = True
                    elif panel == 'download':
                        download_path = data[0]
                        download_visible = True
                        state_changed = True
                    elif panel == 'plan':
                        current_plan = data[0]
                        plan_visible = True
                        state_changed = True
                     
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
                                state_changed = True
                except queue.Empty:
                    pass

                elapsed_secs = int(time.time() - start_time)
                progress_bucket = elapsed_secs // 2
                if state_changed or progress_bucket != last_progress_bucket:
                    html_content = _build_progress_html(elapsed_secs)
                    render_signature = (
                        len(main_chat_history),
                        len(inner_chat_history),
                        len(accumulated_molecules),
                        batch_summary,
                        batch_df_visible,
                        download_visible,
                        current_plan,
                        plan_visible,
                        progress_bucket,
                    )
                    if render_signature != last_render_signature:
                        last_render_signature = render_signature
                        last_progress_bucket = progress_bucket
                        yield _ui_payload(html_content)

                time.sleep(0.25)
        except GeneratorExit:
            generator_closing = True
            _is_processing = False
            if hasattr(current_fleming, '_interrupt_requested'):
                current_fleming._interrupt_requested = True
            raise
        except Exception as e:
            print(f"Error in chat handler: {e}")
            import traceback
            traceback.print_exc()
            _append_unique_chat_message(inner_chat_history, gr.ChatMessage(role="assistant", content=f"An error occurred: {e}"))
            _trim_inner_history()
        
        finally:
            # Only clear the shared processing state if a newer request hasn't already taken
            # over in the meantime. Without this guard, a stale/interrupted request's cleanup
            # could clobber `_is_processing`/`_active_request_signature` right out from under a
            # newer, still-running request, causing it to spuriously self-interrupt.
            if _request_generation == my_generation:
                _is_processing = False
                _active_request_signature = None
                if hasattr(current_fleming, '_interrupt_requested'):
                    current_fleming._interrupt_requested = False
            _last_completed_request_signature = request_signature
            _last_completed_request_time = time.time()
            print(f"✅ Request processing completed (generation {my_generation}, current {_request_generation})")
            log_chat_event("turn_complete", {
                "generation": my_generation,
                "chat": [getattr(m, "content", m) for m in main_chat_history],
                "inner": [getattr(m, "content", m) for m in inner_chat_history],
                "plan": current_plan,
                "molecules": len(accumulated_molecules),
            })

            if not generator_closing:
                yield (
                    main_chat_history, inner_chat_history, accumulated_molecules,
                    gr.update(interactive=True, value="Send"), gr.update(interactive=True, value="Reset"),
                    gr.update(visible=False),
                    batch_summary, gr.update(value=batch_df_value, visible=batch_df_visible),
                    gr.update(value=download_path, visible=download_visible),
                    gr.update(value=current_plan, visible=plan_visible)
                )

    # --- Mode Selection Navigation ---
    def _enter_copilot_mode():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def _enter_batch_mode():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def _request_stop():
        """Ask every running agent and worker thread to stop at its next checkpoint.

        Three things have to happen, because a Python thread cannot be killed from
        outside: the agents get their stop flag, `_is_processing` goes false, and the
        generation counter is bumped. The agent worker loop already breaks when either
        of the latter two changes, so bumping the counter is what actually unblocks it.
        A request already in flight still has to return first - seconds, not instantly.

        The conversation is deliberately left intact so the partial work stays readable.
        """
        global _is_processing, _request_generation, _batch_stop_requested
        _batch_stop_requested = True
        _request_generation += 1          # invalidates the in-flight request's generation
        stopped = []
        for inst in (fleming_real, fleming_sdk):
            if inst is None:
                continue
            if hasattr(inst, "request_stop"):
                inst.request_stop(True)
            else:
                inst._interrupt_requested = True
            stopped.append(type(inst).__name__)
        _is_processing = False
        print(f"\U0001f6d1 Stop requested on: {', '.join(stopped) or 'nothing running'}")
        return gr.update()

    def _hard_reset():
        """Full reset: clear every agent's state and return to the mode-selection screen.

        Goes further than Reset, which only clears the chat. This also drops
        the cached Fleming instances so no sub-agent carries a stale conversation, plan or
        molecule registry into the next session.
        """
        global fleming, fleming_real, fleming_sdk, current_fleming_mode, fleming_mode_locked

        _request_stop()
        try:
            reset_conversation()
        except Exception as exc:                                   # noqa: BLE001
            print(f"hard reset: reset_conversation raised {exc!r}; continuing")

        # Drop the instances so the next turn builds them fresh. get_current_fleming()
        # rebuilds on demand, so this is safe even though it is the nuclear option.
        fleming_real = None
        fleming_sdk = None
        fleming_mode_locked = False
        current_fleming_mode = "legacy"
        fleming = None
        print("\U0001f501 Hard reset: agents dropped, returning to mode selection")

        return (
            gr.update(visible=True),    # mode_selection_group
            gr.update(visible=False),   # copilot_group
            gr.update(visible=False),   # batch_group
            [],                         # chatbot
            [],                         # inner_chatbot
            [],                         # conversation_state
            None,                       # molecule_gallery
            gr.update(value="Waiting for plan...", visible=False),   # plan_display
            "Batch processing status will appear here.",             # batch_summary_display
            gr.update(value=None, visible=False),                    # batch_dataframe_display
            gr.update(value=None, visible=False),                    # download_csv_button
            gr.update(interactive=True),                             # send_btn
            gr.update(interactive=True),                             # reset_btn
        )

    def _back_to_mode_selection():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    _mode_group_outputs = [mode_selection_group, copilot_group, batch_group]
    copilot_mode_btn.click(_enter_copilot_mode, outputs=_mode_group_outputs)
    batch_mode_btn.click(_enter_batch_mode, outputs=_mode_group_outputs)
    back_to_modes_btn.click(_back_to_mode_selection, outputs=_mode_group_outputs)
    back_to_modes_btn_batch.click(_back_to_mode_selection, outputs=_mode_group_outputs)

    # Stop needs no outputs - it only sets a flag the running generator reads.
    for _b in (stop_btn, stop_btn_batch):
        _b.click(_request_stop, outputs=None, queue=False)

    _hard_reset_outputs = _mode_group_outputs + [
        chatbot, inner_chatbot, conversation_state, molecule_gallery, plan_display,
        batch_summary_display, batch_dataframe_display, download_csv_button,
        send_btn, reset_btn,
    ]
    for _b in (hard_reset_btn, hard_reset_btn_batch):
        _b.click(_hard_reset, outputs=_hard_reset_outputs, queue=False)

    def _on_browser_close():
        """Stop everything when the last viewer closes the tab.

        Gradio fires `unload` per browser session, but this app keeps its agents in module
        globals rather than per-session state, so a reset here affects everyone. That is
        the right behaviour for the single-user case this is deployed for, and the wrong
        behaviour if two people ever use one server at once - in that case one person
        closing a tab would stop the other's run. Guarded on FLEMING_RESET_ON_CLOSE so it
        can be turned off without a code change.
        """
        if os.environ.get("FLEMING_RESET_ON_CLOSE", "1").strip().lower() in ("0", "false", "no"):
            print("Browser closed; FLEMING_RESET_ON_CLOSE is off, leaving state alone")
            return
        print("\U0001f6aa Browser closed - stopping work and clearing state")
        log_chat_event("browser_close", {"reason": "unload"})
        try:
            _hard_reset()
        except Exception as exc:                                   # noqa: BLE001
            print(f"unload reset failed: {exc!r}")

    demo.unload(_on_browser_close)

    # --- Batch Mode Handler ---
    _BATCH_STEP_LABEL_TO_KEY = {
        "Ranking (Inhibition + ADMET)": "ranking",
        "Optimization": "optimization",
        "Plausibility Check": "plausibility",
    }

    def run_batch_mode_handler(csv_file, selected_steps, top_n_optimize, top_n_plausibility):
        """
        Runs the deterministic batch pipeline directly (no LLM/chat loop involved) and streams
        progress back to the Batch Mode screen.
        """
        import time as _time

        # A Stop from a previous run must not kill this one.
        global _batch_stop_requested
        _batch_stop_requested = False

        def _progress_html(elapsed_secs):
            return (
                f"<div class='progress-wrapper'><div class='progress-container'>"
                f"<div class='progress-bar' style='width: 100%;'></div></div>"
                f"<div class='timer-text'>Working... {elapsed_secs}s elapsed</div></div>"
            )

        if csv_file is None:
            yield (
                "Please upload a CSV file with a `SMILES` column before running.",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=True, value="Run Batch Analysis"),
                gr.update(visible=False),
            )
            return

        steps_to_run = [_BATCH_STEP_LABEL_TO_KEY[s] for s in (selected_steps or []) if s in _BATCH_STEP_LABEL_TO_KEY]
        if not steps_to_run:
            steps_to_run = ["ranking"]

        current_fleming = get_current_fleming(csv_file)
        # The agent-level stop flag is cleared when a chat turn starts, but a batch run
        # has no such entry point - without this, a Stop pressed during an earlier chat
        # would end this batch at its first phase guard.
        if hasattr(current_fleming, "request_stop"):
            current_fleming.request_stop(False)
        csv_path = getattr(csv_file, "name", csv_file)

        start_time = _time.time()
        result_queue = queue.Queue()

        def batch_thread_func():
            try:
                generator = current_fleming.batchmode_provided_csv(
                    csv_path=csv_path,
                    steps_to_run=steps_to_run,
                    top_n_to_optimize=int(top_n_optimize or 100),
                    top_n_to_plausibility=min(int(top_n_plausibility or 10),
                                              PLAUSIBILITY_MAX_N),
                )
                for item in generator:
                    # The batch generator stops itself between molecules, but the drain
                    # loop also has to notice, or a Stop leaves this thread streaming
                    # already-computed items into a reset UI.
                    if _batch_stop_requested:
                        print("\U0001f6ab Batch interrupted - stopping drain loop")
                        break
                    result_queue.put(item)
            except Exception as exc:
                result_queue.put(exc)
            finally:
                result_queue.put(None)

        thread = threading.Thread(target=batch_thread_func)
        thread.start()

        batch_summary = "Starting batch processing..."
        batch_df_value = None
        batch_df_visible = False
        download_path = None
        download_visible = False

        yield (
            batch_summary,
            gr.update(value=batch_df_value, visible=batch_df_visible),
            gr.update(value=download_path, visible=download_visible),
            gr.update(interactive=False, value="Processing..."),
            gr.update(value=_progress_html(0), visible=True),
        )

        while thread.is_alive() or not result_queue.empty():
            try:
                item = result_queue.get(timeout=0.25)
            except queue.Empty:
                elapsed = int(_time.time() - start_time)
                yield (
                    batch_summary,
                    gr.update(value=batch_df_value, visible=batch_df_visible),
                    gr.update(value=download_path, visible=download_visible),
                    gr.update(interactive=False, value="Processing..."),
                    gr.update(value=_progress_html(elapsed), visible=True),
                )
                continue

            if item is None:
                break
            if isinstance(item, Exception):
                batch_summary = f"Batch processing failed: {item}"
                break

            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], pd.DataFrame):
                batch_df_value, batch_summary = item
                batch_df_visible = True
            elif isinstance(item, tuple) and len(item) == 2 and item[0] == 'download':
                download_path = item[1]
                download_visible = True

            elapsed = int(_time.time() - start_time)
            yield (
                batch_summary,
                gr.update(value=batch_df_value, visible=batch_df_visible),
                gr.update(value=download_path, visible=download_visible),
                gr.update(interactive=False, value="Processing..."),
                gr.update(value=_progress_html(elapsed), visible=True),
            )

        yield (
            batch_summary,
            gr.update(value=batch_df_value, visible=batch_df_visible),
            gr.update(value=download_path, visible=download_visible),
            gr.update(interactive=True, value="Run Batch Analysis"),
            gr.update(visible=False),
        )

    run_batch_btn.click(
        run_batch_mode_handler,
        inputs=[batch_csv_upload, batch_steps_checkboxes, batch_top_n_optimize, batch_top_n_plausibility],
        outputs=[batch_summary_display, batch_dataframe_display, download_csv_button, run_batch_btn, batch_progress_display],
    )

    # --- Copilot Chat Event Listeners ---
    def _stash_and_clear_message(message):
        return message, ""

    send_event = send_btn.click(
        _stash_and_clear_message,
        inputs=[msg_input],
        outputs=[pending_message_buffer, msg_input],
        queue=False,
    ).then(
        custom_chat_handler,
        inputs=[pending_message_buffer, chatbot, inner_chatbot, molecule_gallery, no_csv_state, use_synspace_checkbox],
        outputs=[chatbot, inner_chatbot, molecule_gallery, send_btn, reset_btn, progress_display, batch_summary_display, batch_dataframe_display, download_csv_button, plan_display],
    ).then(lambda: "", outputs=[msg_input])
    
    submit_event = msg_input.submit(
        _stash_and_clear_message,
        inputs=[msg_input],
        outputs=[pending_message_buffer, msg_input],
        queue=False,
    ).then(
        custom_chat_handler,
        inputs=[pending_message_buffer, chatbot, inner_chatbot, molecule_gallery, no_csv_state, use_synspace_checkbox],
        outputs=[chatbot, inner_chatbot, molecule_gallery, send_btn, reset_btn, progress_display, batch_summary_display, batch_dataframe_display, download_csv_button, plan_display],
    ).then(lambda: "", outputs=[msg_input])

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
        _, local_url, share_url = demo.launch(share=True, server_name="0.0.0.0")
        print("Gradio local URL:", local_url, flush=True)
        print("Gradio share URL:", share_url, flush=True)

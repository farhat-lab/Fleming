# Fleming: A Conversational Chemist Agent

Fleming is a multi-agent system designed for pharmaceutical sciences, capable of answering questions about molecules, predicting properties (ADMET, inhibition), and generating new molecular structures.

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11+ for the current `pqapi`, `admet-ai`, and `codex_sdk` stack
- `venv` is sufficient; Conda is optional

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd fleming
    ```

2.  **Create and populate the runtime environment:**
    On HMS/cluster systems with environment modules, the supported path is Conda via Lmod:

    ```bash
    module load conda/miniforge3/24.11.3-0
    bash scripts/bootstrap_fleming_conda.sh
    conda activate "$(pwd)/.conda-env/fleming-codex-sdk"
    python scripts/check_fleming_env.py --scope codex-sdk
    ```

    This installs from [environment.conda.yml](/n/data1/hms/dbmi/farhat/anz226/Fleming/fleming/environment.conda.yml:1), which is the supported runtime for:
    - the Gradio Fleming app
    - the new `codex_sdk` entrypoint
    - optimization with `immediate`, `synspace`, or `auto`

3.  **Alternative `venv` path:**
    If Conda is unavailable on your machine, there is also a Python `venv` bootstrap:
    ```bash
    PYTHON_BIN=python3.11 bash scripts/bootstrap_fleming_env.sh
    python scripts/check_fleming_env.py --scope codex-sdk
    ```

    *Note: The `molecule_diffusion` module has specific requirements. If you plan to use the Molecular Generation Agent, ensure you have a compatible PyTorch and RDKit version installed (see `molecule_diffusion/e3_diffusion_for_molecules/requirements.txt` for reference).*

### 3. Configuration

1.  **Copy the example config:**
    ```bash
    cp config.example.yaml config.yaml
    ```

2.  **Edit `config.yaml`:**
    - Add your **OpenAI API Key** (or Azure configuration).
    - Add your **PQApi Token** (if using Pharmacist Quality features).
    - Verify paths for models and data files.

### 4. Running Fleming

To launch the Gradio interface:

```bash
python run_fleming_interface.py
```

## 🧩 External Tools & Models

Fleming relies on several external tools and models which may need to be downloaded or configured separately.

### ChemMCP (for SIDER, Solubility, Toxicity)

The ADMET Agent uses scripts in `utils/external_tools/chemmcp_tools/` which depend on the `ChemMCP` library.

1.  **Download/Install ChemMCP:**
    Ensure the `ChemMCP` library is available. The scripts expect to find it in `utils/ChemMCP`.
    
    If you have the source code for ChemMCP, place it at:
    `utils/ChemMCP/`

2.  **Environment:**
    If ChemMCP requires a conflicting environment, create a separate conda env (e.g., `chemtools`) and specify the run command in `config.yaml`:
    ```yaml
    chemmcp_python_cmd: "conda run -n chemtools python"
    ```

### Molecule Diffusion (Generation Agent)

The generation agent uses a diffusion model located in `molecule_diffusion/`.
- Ensure the model checkpoints are present in `molecule_diffusion/e3_diffusion_for_molecules/outputs/TB_Diffusion_resume/`.
- If using `synthemol`, additional setup may be required.

### Codex SDK Variant

There is also a `codex_sdk` implementation that exposes Fleming through typed SDK tools instead of the legacy prompt-parsed `Tool-call:` loop.

1. After activating the bootstrapped environment, run a single prompt:
   ```bash
   python run_codex_sdk_fleming.py --message "Optimize CC1=CC=CC=C1 for lower toxicity" --optimization-strategy immediate
   ```
2. Or start an interactive session:
   ```bash
   python run_codex_sdk_fleming.py --interactive --optimization-strategy synspace
   ```

The `codex_sdk` path keeps the same high-level tool families:
- ADMET assessment
- inhibition prediction
- molecular optimization
- molecular generation
- molecular reasoning / description

The optimization tool now supports explicit backend selection:
- `immediate`: STONED-like immediate chemical space exploration using SELFIES edits
- `synspace`: the original synspace neighborhood generator
- `llm`: LLM-generated close analogs
- `auto`: try `immediate`, then `synspace`, then `llm`

### SCScore & SAScore

These tools are included in `utils/external_tools/`.
- **SCScore**: Included in `utils/external_tools/scscore`. Uses pre-trained models in `models/`.
- **SAScore**: Included in `utils/external_tools/sascore`. Requires `fpscores.pkl.gz` (included).

## 📂 Project Structure

- `main_fleming.py`: Core agent logic.
- `run_fleming_interface.py`: Entry point for the Gradio UI.
- `utils/`: Helper modules for different agents.
  - `admet_utils.py`: Handling ADMET properties.
  - `molecular_generation_utils.py`: Logic for diffusion models.
  - `env_manager.py`: Manages subprocess calls to external environments.
  - `external_tools/`: Third-party scripts (ChemMCP wrappers, SCScore).
- `molecule_diffusion/`: Code for the E3 Diffusion model.

## 🛠 Troubleshooting

- **Import Errors**: Ensure you are running from the project root.
- **Model Not Found**: Check `config.yaml` paths and ensure model checkpoints are downloaded.
- **CUDA/GPU**: Molecular generation is faster with a GPU. Ensure PyTorch is installed with CUDA support.

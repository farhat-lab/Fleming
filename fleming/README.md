# Fleming: A Conversational Chemist Agent

Fleming is a multi-agent system designed for pharmaceutical sciences, capable of answering questions about molecules, predicting properties (ADMET, inhibition), and generating new molecular structures.

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- Conda (Miniconda or Anaconda) recommended

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd fleming
    ```

2.  **Create a Conda environment:**
    It is recommended to use a single environment for most tasks, although some external tools may require specific environments (see "External Tools" below).

    ```bash
    conda create -n fleming python=3.10
    conda activate fleming
    ```

3.  **Install Dependencies:**
    
    Install the core requirements:
    ```bash
    pip install -r requirements.txt
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


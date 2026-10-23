# Fleming: A Conversational Chemist Agent

Fleming is a multi-agent system designed for pharmaceutical sciences, capable of answering questions about molecules, predicting properties (ADMET, inhibition), and generating new molecular structures.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda (Miniconda or Anaconda) recommended

### Installation

1.  **Clone the repository:**
```bash
git clone <repo-url>
cd fleming
```

2.  **Create a Conda environment:**

It is recommended to use a single environment for most tasks, although some external tools may require specific environments (see "External Tools" below). Conda is ideal for this project because it needs RDKit, PyTorch with CUDA, and other scientific libraries that have complex non-Python dependencies.

Use HomeBrew to install Anaconda on macOS
```bash
brew install --cask anaconda
/opt/homebrew/anaconda3/bin/conda init zsh
source ~/.zshrc
```

Create a new environment
```bash
conda env create -f environment.fleming.yml
conda activate fleming
```

3.  **Install Dependencies:**
    
Install the core requirements:
```bash
pip install -r requirements-core.txt
```

See the [Development](#development) section below for more details

*Note: The `molecule_diffusion` module has specific requirements. If you plan to use the Molecular Generation Agent, ensure you have a compatible PyTorch and RDKit version installed (see `molecule_diffusion/e3_diffusion_for_molecules/requirements.txt` for reference).*

### Configuration

1.  **Copy the example config:**
```bash
cp config.example.yaml config.yaml
```

2.  **Edit `config.yaml`:**
- Add your **OpenAI API Key** (or Azure configuration).
- Add your **PQApi Token** (if using Pharmacist Quality features).
- Verify paths for models and data files.

### Running Fleming

To launch the Gradio interface:

```bash
python run_fleming_interface.py
```

By default the UI uses the legacy backend (`main_fleming.Fleming`). If you still have the optional SDK backend available, you can select it with `FLEMING_BACKEND=sdk`.

To run the automatic CSV pipeline from the command line:

```bash
python run_fleming_batch_cli.py path/to/input.csv
```

The CSV must include a `SMILES` column. By default this runs `ranking`, `optimization`, and `plausibility`, then writes:

- `<input>_fleming_results.csv`
- step-specific files such as `<input>_fleming_results_ranking_results.csv`

Example with explicit output and step control:

```bash
python run_fleming_batch_cli.py data/molecules.csv \
  --output outputs/molecules_ranked.csv \
  --steps ranking optimization \
  --top-n-to-optimize 50
```

### Containerized interface-only environment

If you only need to run `run_fleming_interface.py` and do not need the diffusion stack or ChemMCP/chemtools, use the environment-only container recipe under [docker/interface/README.md](/n/data1/hms/dbmi/farhat/anz226/Fleming/fleming/docker/interface/README.md).

That workflow:

- targets O2-style `Singularity` / `Apptainer` usage
- stores the built `.sif` in `/n/scratch/users/a/anz226/`
- keeps the Fleming source tree, model checkpoints, and `config.yaml` outside the image
- runs the app by bind-mounting the repo into the container at runtime

## 🧩 External Tools & Models

Fleming relies on several external tools and models which may need to be downloaded or configured separately.

### ChemMCP (for SIDER, Solubility, Toxicity)

The ADMET Agent uses scripts in `utils/external_tools/chemmcp_tools/` which depend on the `ChemMCP` library.

1.  **Download/Install ChemMCP:**
Ensure the `ChemMCP` library is available locally. The wrappers support either layout:
- `utils/ChemMCP/`
- `utils/external_tools/ChemMCP/`

If you have the source code for ChemMCP, place it at:
`utils/ChemMCP/` (recommended)

2.  **Environment:**
If ChemMCP requires a conflicting environment, create a separate conda env (e.g., `chemtools`) and specify the run command in `config.yaml`:
```yaml
chemmcp_python_cmd: "conda run -n chemtools python"
```
A starter environment file is provided at `environment.chemtools.yml`.

### Molecule Diffusion (Generation Agent)

The generation agent uses a diffusion model located in `molecule_diffusion/`.
- Ensure the model checkpoints are present in `molecule_diffusion/e3_diffusion_for_molecules/outputs/TB_Diffusion_resume/`.
- If using `synthemol`, additional setup may be required.

### SCScore & SAScore

These tools are included in `utils/external_tools/`.
- **SCScore**: Included in `utils/external_tools/scscore`. Uses pre-trained models in `models/`.
- **SAScore**: Included in `utils/external_tools/sascore`. Requires `fpscores.pkl.gz` (included).

## 📊 Data Files

- **`data/chembl_abx_cleaned.csv`**: A set of antibiotics and antibacterial candidate compounds pulled from [ChEMBL](https://www.ebi.ac.uk/chembl/), including approved antibiotics as well as investigational/candidate compounds (not filtered to only fully-approved drugs). This is used as the reference set for **novelty scoring** — new/generated molecules are compared (via Tanimoto similarity on Morgan fingerprints) against this set, so novelty reflects structural dissimilarity to known antibiotics and antibacterial candidates. Configured via `chembl_antibiotics_data` in `config.yaml`.

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
- **Optional deps**: If you see warnings about missing `pqapi`, `PyTDC/tdc`, `langchain_community`, `chemicalconverters`, or `synspace`, those features are disabled but the core app should still run.
- **Missing IUPAC names**: The `describe_molecule` tool (and `get_iupac`) optionally use the `chemicalconverters` module (pip package **`chemical-converters`**) to compute IUPAC names. It is **not installed by default** because it pins old, narrow dependency versions (`transformers==4.36.2`, `torch==2.1.2`, `rdkit==2023.3.3`) that conflict with the rest of this project's stack. If it's missing (or fails to load its model), IUPAC names are reported as unavailable and the LLM instead reasons directly from the SMILES string - the description quality is not meaningfully affected.

## Development
Fleming runs in a conda environment. Here is a process to help manage the environment and corresponding YAML file

1. Install with conda

```bash
conda activate fleming
conda install package-name
```

Use pip for packages not in conda:

```bash
pip install package-name
```

2. Test to ensure the package is installed correctly
3. Add to `environment.fleming.yml` and update

Edit the YAML file:

```yaml
dependencies:
  - new-package
  - pip:
    - pip-only-package
```

Then update:
```bash
conda env update -f environment.fleming.yml
```

4. Recreate the environmnet for major changes (e.g. python version upgrade)
```bash
conda deactivate
conda env remove -n fleming
conda env create -f environment.fleming.yml
```
import shutil
import subprocess
import os
import yaml

def get_project_root() -> str:
    """
    Return the project root directory (repo root), assuming this file lives in `utils/`.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_conda_command():
    """
    Detect available conda environment manager and return the appropriate command.
    
    Returns:
        str: 'micromamba', 'mamba', or 'conda' depending on what's available
    """
    # Check in order of preference: micromamba, mamba, conda
    for cmd in ['micromamba', 'mamba', 'conda']:
        if shutil.which(cmd) is not None:
            return cmd
    
    # If none found, return conda as fallback
    return 'conda'

def get_chemmcp_python_cmd(config_path="config.yaml"):
    """
    Get the command to run python for ChemMCP tools from config.
    Defaults to 'conda run -n chemtools python'.
    """
    default_cmd = f"{get_conda_command()} run -n chemtools python"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'chemmcp_python_cmd' in config:
                    return config['chemmcp_python_cmd']
        except Exception as e:
            print(f"Warning: Could not read config for chemmcp command: {e}")
            
    return default_cmd

def run_chemmcp_script(script_relative_path, args, config_path="config.yaml"):
    """
    Run a ChemMCP script using the configured environment.
    
    Args:
        script_relative_path: Path to script relative to project root (e.g. 'utils/external_tools/chemmcp_tools/sider.py')
        args: List of arguments to pass to the script
        config_path: Path to config file
        
    Returns:
        subprocess.CompletedProcess
    """
    python_cmd = get_chemmcp_python_cmd(config_path)
    
    # Split python command into parts (e.g. ['conda', 'run', '-n', 'chemtools', 'python'])
    # This handles spaces in the command string properly for subprocess
    import shlex
    cmd_parts = shlex.split(python_cmd)
    
    # Append script path and args
    full_cmd = cmd_parts + [script_relative_path] + args
    
    project_root = get_project_root()
    env = os.environ.copy()
    # Make local sources importable for the ChemMCP wrappers and any vendored tools.
    extra_paths = [
        project_root,
        os.path.join(project_root, "utils"),
        os.path.join(project_root, "utils", "external_tools"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([p for p in extra_paths if p] + ([existing] if existing else []))

    return subprocess.run(
        full_cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True
    )

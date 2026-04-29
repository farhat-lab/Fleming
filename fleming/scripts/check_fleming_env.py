from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CODEX_SDK_REQUIRED_MODULES = [
    "openai",
    "agents",
    "gradio",
    "imageio",
    "torch",
    "rdkit",
    "chemprop",
    "admet_ai",
]

FULL_REQUIRED_MODULES = [
    *CODEX_SDK_REQUIRED_MODULES,
    "tdc",
    "pqapi",
    "langchain_community",
    "langchain_experimental",
    "chemicalconverters",
    "selfies",
]

OPTIONAL_MODULES = [
    "synspace",
    "selfies",
    "chemicalconverters",
    "pqapi",
    "tdc",
    "langchain_community",
    "langchain_experimental",
]

REQUIRED_PATHS = [
    ROOT / "config.yaml",
    ROOT / "model_checkpoints" / "model.pt",
    ROOT / "molecule_diffusion" / "e3_diffusion_for_molecules" / "outputs" / "TB_Diffusion_resume" / "args.pickle",
]


def check_modules(modules: list[str], required: bool) -> list[str]:
    errors: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"[ok] module: {module_name}")
        except Exception as exc:
            level = "error" if required else "warn"
            print(f"[{level}] module: {module_name} -> {exc}")
            if required:
                errors.append(f"missing module {module_name}: {exc}")
    return errors


def check_paths(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.exists():
            print(f"[ok] path: {path}")
        else:
            print(f"[error] path: {path} is missing")
            errors.append(f"missing path {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Fleming runtime environment.")
    parser.add_argument(
        "--scope",
        choices=["codex-sdk", "full"],
        default="codex-sdk",
        help="Validation profile to run.",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version}")
    print(f"Repo root: {ROOT}")
    print(f"Validation scope: {args.scope}")
    print()

    required_modules = (
        FULL_REQUIRED_MODULES if args.scope == "full" else CODEX_SDK_REQUIRED_MODULES
    )

    errors = []
    errors.extend(check_modules(required_modules, required=True))
    check_modules(OPTIONAL_MODULES, required=False)
    errors.extend(check_paths(REQUIRED_PATHS))

    print()
    if errors:
        print("Environment check failed.")
        for error in errors:
            print(f" - {error}")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

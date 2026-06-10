import argparse
import os
import sys

DEFAULT_STEPS = ["ranking", "optimization", "plausibility"]
STEP_CHOICES = ["ranking", "optimization", "plausibility"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Fleming automatic mode from the CLI on a CSV with a SMILES column."
    )
    parser.add_argument("csv", help="Path to the input CSV. Must include a 'SMILES' column.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to Fleming config. Defaults to config.yaml in the repo root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the final output CSV. Defaults to '<input>_fleming_results.csv'.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_CHOICES,
        default=DEFAULT_STEPS,
        help="Pipeline steps to run. Default: ranking optimization plausibility",
    )
    parser.add_argument(
        "--top-n-to-optimize",
        type=int,
        default=100,
        help="Number of top-ranked molecules to retain before optimization.",
    )
    parser.add_argument(
        "--top-n-to-plausibility",
        type=int,
        default=100,
        help="Number of molecules to send to plausibility assessment.",
    )
    return parser


def resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.path.dirname(__file__), path_value)


def default_output_path(csv_path: str) -> str:
    base, ext = os.path.splitext(csv_path)
    if not ext:
        ext = ".csv"
    return f"{base}_fleming_results{ext}"


def validate_args(args):
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    if args.top_n_to_optimize <= 0:
        raise ValueError("--top-n-to-optimize must be greater than 0.")
    if args.top_n_to_plausibility <= 0:
        raise ValueError("--top-n-to-plausibility must be greater than 0.")


def main():
    args = build_parser().parse_args()

    try:
        validate_args(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    config_path = resolve_path(args.config)
    output_path = os.path.abspath(args.output or default_output_path(os.path.abspath(args.csv)))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        from utils.automatic_mode_utils import Batch_Mode_Utils
    except ModuleNotFoundError as exc:
        print(
            f"[error] Missing Python dependency while loading Fleming automatic mode: {exc}",
            file=sys.stderr,
        )
        return 1

    agent = Batch_Mode_Utils(config_path, agent_name="Automatic Mode CLI", gradio=False, prompts=None)

    print(f"[info] Input CSV: {os.path.abspath(args.csv)}")
    print(f"[info] Config: {config_path}")
    print(f"[info] Steps: {', '.join(args.steps)}")
    print(f"[info] Output CSV: {output_path}")

    final_df, final_summary = agent.run_complete_pipeline(
        steps_to_run=args.steps,
        top_n_to_optimize=args.top_n_to_optimize,
        top_n_to_plausibility=args.top_n_to_plausibility,
        describe_mol_function=agent.describe_molecule,
        output_file=output_path,
        smiles_csv_path=os.path.abspath(args.csv),
    )

    if final_df.empty:
        print("[error] Pipeline completed without results.", file=sys.stderr)
        if final_summary:
            print(final_summary, file=sys.stderr)
        return 1

    print("")
    print(final_summary)
    print("")
    print(f"[ok] Final results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Arm 3 of 3: the models alone, with no LLM anywhere in the loop.

This is the baseline the whole benchmark is measured against. It calls no API, costs
nothing, and takes about a minute for all 150 molecules, so run it first and run it often.

Two rankings come out of one pass:

  chemprop          ranked by the Chemprop Mtb inhibition score alone. This is the
                    comparison you asked for: can the agent beat the inhibition model at
                    retrieving compounds that are active AND non-cytotoxic, when the model
                    only knows about activity?
  chemprop_proxy    ranked by inhibition x (1 - HepG2 cytotoxicity), using the human
                    cytotoxicity model as a stand-in for Vero. This is the harder baseline,
                    and the more honest one: it is what the existing models can do about
                    BOTH axes without any language model. If Fleming does not beat this, the
                    reasoning is not what is adding value - the extra model call is.

Neither model was trained on Vero cells. The proxy arm uses HepG2 (human hepatoma) because
it is the cytotoxicity model with the most training data of the three, not because it is
the closest tissue to monkey kidney epithelium; `--proxy-line` switches it, and
`--proxy-line mean` averages all three, which is the least arbitrary choice.

    python run_chemprop_baseline.py data/mtb_150_smiles.csv --output-dir results/
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import arm_io  # noqa: E402
import scoring  # noqa: E402
from fleming_harness import FLEMING_DIR, prepare_environment  # noqa: E402

PROXY_LINES = ("hepg2", "hskmc", "imr90", "mean", "max", "none")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", default=os.path.join(_HERE, "results"))
    parser.add_argument("--tag", default="", help="Suffix for the output filenames.")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--tb-col", default="Y_TB")
    parser.add_argument("--vero-col", default="Y_Vero")
    parser.add_argument("--group-col", default="selection_group")
    parser.add_argument("--config", default=None,
                        help="Fleming config YAML (default: fleming/config.yaml).")
    parser.add_argument("--proxy-line", choices=PROXY_LINES, default="hepg2",
                        help="Which cytotoxicity model stands in for Vero in the proxy arm. "
                        "'mean' averages all three; 'max' takes the worst; 'none' skips the "
                        "proxy arm entirely.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output files instead of refusing.")
    return parser


def main():
    args = build_parser().parse_args()
    input_csv = os.path.abspath(args.input_csv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows = arm_io.read_input(input_csv, args.smiles_col, args.tb_col, args.vero_col,
                             args.group_col)
    smiles_list = [row["SMILES"] for row in rows]
    print(f"[info] arm              {arm_io.ARM_CHEMPROP} (no API calls, no cost)")
    print(f"[info] input            {input_csv} ({len(rows)} molecules)")

    tag = f"_{args.tag}" if args.tag else ""
    inhibition_out = os.path.join(output_dir, f"chemprop{tag}.csv")
    proxy_out = os.path.join(output_dir, f"chemprop_proxy{tag}.csv")
    for path in (inhibition_out, proxy_out):
        if os.path.exists(path) and not args.overwrite:
            print(f"[error] {path} exists. Pass --overwrite to replace it.", file=sys.stderr)
            return 2

    # Fleming's config points at its checkpoints with paths relative to the fleming/
    # directory, so the process has to run from there.
    prepare_environment()
    config_path = os.path.abspath(args.config) if args.config else \
        os.path.join(FLEMING_DIR, "config.yaml")

    from prompts import inhibition_agent_prompts  # noqa: E402
    from utils.inhibition_utils import InhibitionUtils  # noqa: E402

    # InhibitionUtils, not ADMETUtils: it gives the same load_models/make_prediction but
    # without pulling in ADMET-AI, which would cost a minute and several GB for nothing.
    agent = InhibitionUtils(config_path, "Chemprop Baseline", False, inhibition_agent_prompts)

    started = time.time()
    print("[info] scoring Mtb inhibition ...")
    inhibition, inhibition_unc = agent.get_chemprop_prediction(smiles_list)
    inhibition = [float(v) for v in inhibition]
    inhibition_unc = [float(v) for v in inhibition_unc]
    print(f"[info] done in {time.time() - started:.0f}s")

    cytotox = {}
    if args.proxy_line != "none":
        project_root = FLEMING_DIR
        paths = agent.config.get("cytotox_model_paths", [
            "model_checkpoints/hepg2_model.pt",
            "model_checkpoints/hskmc_model.pt",
            "model_checkpoints/imr90_model.pt",
        ])
        paths = [p if os.path.isabs(p) else os.path.join(project_root, p) for p in paths]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"[error] cytotoxicity checkpoints not found: {missing}", file=sys.stderr)
            return 2
        for name, path in zip(("hepg2", "hskmc", "imr90"), paths):
            print(f"[info] scoring {name} cytotoxicity ...")
            models = agent.load_models([path], "cpu")
            preds, _ = agent.make_prediction(models, smiles_list)
            cytotox[name] = [float(v) for v in preds]

    def proxy(i):
        if args.proxy_line == "none" or not cytotox:
            return None
        if args.proxy_line == "mean":
            return sum(values[i] for values in cytotox.values()) / len(cytotox)
        if args.proxy_line == "max":
            return max(values[i] for values in cytotox.values())
        return cytotox[args.proxy_line][i]

    inhibition_rows, proxy_rows = [], []
    for i, row in enumerate(rows):
        shared = {
            "status": arm_io.STATUS_COMPLETE,
            "score_source": "model",
            "raw_inhibition": round(inhibition[i], 6),
            "inhibition_uncertainty": round(inhibition_unc[i], 6),
            "raw_hepg2": round(cytotox["hepg2"][i], 6) if cytotox else "",
            "raw_hskmc": round(cytotox["hskmc"][i], 6) if cytotox else "",
            "raw_imr90": round(cytotox["imr90"][i], 6) if cytotox else "",
            "n_api_calls": 0, "n_web_searches": 0, "n_lit_reviews": 0,
            "cost_usd": 0.0, "runtime_sec": "",
            "tools_used": "get_chemprop_prediction",
        }

        # Arm: inhibition alone. `vero_score` is deliberately left blank - this arm has no
        # opinion at all about cytotoxicity, and writing a 0 there would be a claim.
        out = arm_io.blank_row(row, arm_io.ARM_CHEMPROP)
        out.update(shared)
        out.update({
            "tb_score": round(inhibition[i], 6),
            "vero_score": "",
            "promise_score": "",
            "ranking_score": round(inhibition[i], 6),
            "ranking_mode": "inhibition_only",
            "tiebreak": "",
        })
        inhibition_rows.append(out)

        if args.proxy_line != "none":
            proxy_value = proxy(i)
            rank, mode = scoring.ranking_score(inhibition[i], proxy_value, None, mode="product")
            out = arm_io.blank_row(row, f"{arm_io.ARM_CHEMPROP}_proxy")
            out.update(shared)
            out.update({
                "tb_score": round(inhibition[i], 6),
                "vero_score": round(proxy_value, 6),
                "promise_score": "",
                "ranking_score": rank,
                "ranking_mode": f"product({args.proxy_line})",
                "tiebreak": scoring.tiebreak_score(inhibition[i], proxy_value),
                "tools_used": "get_chemprop_prediction;get_cytotoxicity_copilot_mode",
            })
            proxy_rows.append(out)

    for path, out_rows in ((inhibition_out, inhibition_rows), (proxy_out, proxy_rows)):
        if not out_rows:
            continue
        if os.path.exists(path):
            os.remove(path)
        for out in out_rows:
            arm_io.write_row(path, out)
        print(f"[ok] {len(out_rows)} rows -> {path}")

    print("")
    print(f"total wall clock: {(time.time() - started) / 60:.1f} min, $0.00")
    print("Next:  python evaluate.py " + inhibition_out +
          (" " + proxy_out if proxy_rows else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

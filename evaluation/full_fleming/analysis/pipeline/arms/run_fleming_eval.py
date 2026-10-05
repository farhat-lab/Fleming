#!/usr/bin/env python
"""
Arm 1 of 3: the full Fleming system.

Runs every subagent over a CSV of molecules, scoring TB whole-cell activity and Vero
cytotoxicity, and asking for one 0-10 promise score that weighs the two together.

One Fleming run per molecule, fresh conversation each time. The molecule always produces
exactly one output row whatever happens to it, and `status` says which kind of row it is -
anything other than "complete" is an exclusion you should count and report, because these
failures are content-triggered and therefore not missing-at-random.

Input CSV: a SMILES column, plus `Y_TB` and `Y_Vero` labels. Labels are carried through
untouched and never shown to the agent - only the SMILES reaches the prompt.

Results are appended row by row, so a long sweep resumes after an interruption: re-run the
same command and it skips the rows already in the output file. The full transcript of each
molecule is written as it happens, so an aborted run still leaves a record of what the
agent did and what it cost.

Example:
    python run_fleming_eval.py data/mtb_150_smiles.csv \
        --output results/fleming_pilot.csv --limit 3
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import arm_io  # noqa: E402
import scoring  # noqa: E402
from fleming_harness import (  # noqa: E402
    FLEMING_DIR,
    HarnessConfig,
    FlemingRunner,
    read_quota,
    resolve_base_url,
)

DEFAULT_PROMPT_FILE = os.path.join(_HERE, "prompts", "fleming_vero.txt")
FOLLOWUP_PROMPT_FILE = os.path.join(_HERE, "prompts", "score_block_followup.txt")

import threading  # noqa: E402

_LOCAL = threading.local()


def get_runner(cfg):
    """
    One Fleming per worker thread, built on first use.

    Building it loads three ADMET-AI models, so it is reused across molecules (with a full
    conversation reset between them) rather than rebuilt. That also means each extra worker
    costs another full copy of those models in RAM - see the --workers note in the README
    before raising it.
    """
    runner = getattr(_LOCAL, "runner", None)
    if runner is None:
        runner = FlemingRunner(cfg)
        _LOCAL.runner = runner
    return runner


def run_one(row, cfg, args, prompt_template, followup_prompt, transcript_dir):
    started = time.time()
    index, smiles = row["index"], row["SMILES"]
    instruction = prompt_template.replace("{SMILES}", smiles)

    transcript = arm_io.TranscriptWriter(transcript_dir, index, {
        "arm": arm_io.ARM_FLEMING, "index": index, "SMILES": smiles, "model": cfg.model,
        "planner_model": cfg.resolved_planner_model(),
        "literature_model": cfg.resolved_literature_model(),
        "max_rounds": cfg.max_rounds, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }) if transcript_dir else None

    out = arm_io.blank_row(row, arm_io.ARM_FLEMING)
    try:
        runner = get_runner(cfg)
        result = runner.run(smiles, instruction,
                            on_event=(transcript.write if transcript else None))

        answer = result["answer"]
        parsed = scoring.parse_answer(answer)
        # The block is the whole point of the row, and the analysis behind it has already
        # been paid for, so ask once for just the block rather than discarding the molecule.
        if parsed["score_source"] in ("none", "prose") and answer.strip() and not args.no_followup:
            followup_text = runner.ask_for_score_block(followup_prompt)
            if followup_text:
                if transcript:
                    transcript.write({"kind": "score_followup", "role": "assistant",
                                      "content": followup_text})
                parsed = scoring.parse_answer(answer, followup_text)

        status = result["status"]
        if status == arm_io.STATUS_COMPLETE and parsed["score_source"] in ("none", "prose"):
            status = arm_io.STATUS_NO_BLOCK
        # A score block is not enough on its own. If Fleming never reached a subagent or a
        # quantitative tool, the numbers are the orchestrator's unaided opinion, which is
        # the control arm, not this one. Flagged so it is excluded rather than compared.
        if status == arm_io.STATUS_COMPLETE and not arm_io.ran_any_evidence_tool(
                result["tools_used"], result["subagents_used"]):
            status = arm_io.STATUS_NO_TOOLS

        arm_io.apply_scores(out, parsed, mode=args.rank_by, weight=args.rank_weight)
        arm_io.apply_usage(out, result["usage"])
        out.update({
            "status": status,
            "subagents_used": ";".join(result["subagents_used"]),
            "tools_used": ";".join(result["tools_used"]),
            "error": result["error"],
            "plan": result["plan"],
            "answer": answer,
        })
    except Exception as exc:
        out["status"] = arm_io.STATUS_FAILED
        out["error"] = f"{type(exc).__name__}: {exc}"
        if transcript:
            transcript.write({"kind": "error", "content": traceback.format_exc()[-4000:]})
        print(f"[{index}] FAILED: {out['error']}", flush=True)

    out["runtime_sec"] = round(time.time() - started, 1)
    if transcript:
        transcript.write({"kind": "result", **{k: out[k] for k in
                          ("status", "score_source", "tb_score", "vero_score",
                           "promise_score", "ranking_score", "cost_usd", "runtime_sec")}})
        transcript.close(complete=True)
    return out


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the full Fleming system over a CSV, scoring TB activity and Vero "
                    "cytotoxicity together.")
    parser.add_argument("input_csv", help="Input CSV with a SMILES column.")
    parser.add_argument("--output", required=True, help="Output CSV. Appended to and resumable.")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--tb-col", default="Y_TB", help="1 = TB hit. Carried through only.")
    parser.add_argument("--vero-col", default="Y_Vero", help="1 = Vero-cytotoxic. Blank = never counter-screened.")
    parser.add_argument("--group-col", default="selection_group")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run at most this many molecules. Applied AFTER the resume "
                        "filter, so repeated calls with --limit walk through the file in "
                        "batches instead of re-checking the same finished rows.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Molecules in parallel. Each worker holds its own full Fleming "
                        "(three ADMET-AI models), so raise this with the job's memory.")
    parser.add_argument("--start", type=int, default=None, help="Skip rows before this index.")
    parser.add_argument("--end", type=int, default=None, help="Skip rows at or after this index.")

    parser.add_argument("--model", default=None, help="Model for every agent turn.")
    parser.add_argument("--planner-model", default=None,
                        help="Model for the plan turn, re-plans and the final summary only.")
    parser.add_argument("--literature-model", default=None,
                        help="Model for literature/web-search calls. Defaults to o4-mini; "
                        "pass an empty string to use --model for those too.")
    parser.add_argument("--reasoning", default=None)
    parser.add_argument("--planner-reasoning", default=None)
    parser.add_argument("--base-url", default=None,
                        help="API base URL. Pass an empty string to force direct OpenAI.")
    parser.add_argument("--config", default=None,
                        help="Fleming config YAML (default: fleming/config.yaml).")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Cap on Fleming's own ReAct rounds per molecule.")
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-budget", type=float, default=None)
    parser.add_argument("--literature-backend", choices=("web", "pqa"), default="web")
    parser.add_argument("--canned-lit-context", choices=("keep", "drop"), default="keep",
                        help="Whether literature_review keeps the hard-coded liver-toxicity "
                        "paragraph Fleming injects into every call (see the README).")

    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE,
                        help="Per-molecule instruction. Must contain {SMILES}.")
    parser.add_argument("--rank-by", choices=scoring.RANKING_MODES, default="promise",
                        help="What the ranking_score column holds. 'promise' uses the "
                        "agent's own 0-10 score; the others compute it from the two axes.")
    parser.add_argument("--rank-weight", type=float, default=0.5,
                        help="Weight on TB activity for --rank-by weighted.")
    parser.add_argument("--no-followup", action="store_true",
                        help="Do not make the extra call that asks for the score block when "
                        "an answer came back without one.")
    parser.add_argument("--no-transcripts", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run - molecules, models, prompt - and exit "
                        "without calling any model.")
    return parser


def build_config(args):
    cfg = HarnessConfig()
    for attribute, value in (
        ("model", args.model), ("planner_model", args.planner_model),
        ("reasoning", args.reasoning), ("planner_reasoning", args.planner_reasoning),
        ("max_rounds", args.max_rounds), ("request_timeout", args.request_timeout),
        ("max_retries", args.max_retries), ("retry_budget", args.retry_budget),
    ):
        if value is not None:
            setattr(cfg, attribute, value)
    # Empty string means "use the main model", which is different from "not specified".
    if args.literature_model is not None:
        cfg.literature_model = args.literature_model or None
    cfg.base_url = args.base_url
    cfg.config_path = os.path.abspath(args.config) if args.config else None
    cfg.literature_backend = args.literature_backend
    cfg.canned_lit_context = args.canned_lit_context
    return cfg


def main():
    args = build_parser().parse_args()

    # Resolve every path before the harness chdirs into fleming/ for Fleming's own relative
    # config paths. A relative --output would otherwise land in the wrong place.
    input_csv = os.path.abspath(args.input_csv)
    output_csv = os.path.abspath(args.output)
    prompt_file = os.path.abspath(args.prompt_file)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(prompt_file) as handle:
        prompt_template = handle.read().strip()
    if "{SMILES}" not in prompt_template:
        print(f"[error] {prompt_file} does not contain the {{SMILES}} placeholder.",
              file=sys.stderr)
        return 2
    with open(FOLLOWUP_PROMPT_FILE) as handle:
        followup_prompt = handle.read().strip()

    rows = arm_io.read_input(input_csv, args.smiles_col, args.tb_col, args.vero_col,
                             args.group_col)
    if args.start is not None:
        rows = [row for row in rows if row["index"] >= args.start]
    if args.end is not None:
        rows = [row for row in rows if row["index"] < args.end]
    done = arm_io.already_done(output_csv)
    pending = [row for row in rows if row["index"] not in done]
    # --limit AFTER the resume filter: applying it first makes repeated batching impossible,
    # because every call would re-examine the same already-finished rows.
    if args.limit is not None:
        pending = pending[:args.limit]

    cfg = build_config(args)
    transcript_dir = None if args.no_transcripts else \
        os.path.splitext(output_csv)[0] + "_transcripts"

    print(f"[info] arm              {arm_io.ARM_FLEMING}")
    print(f"[info] input            {input_csv} ({len(rows)} molecules)")
    print(f"[info] output           {output_csv}")
    print(f"[info] already done     {len(done)}")
    print(f"[info] to run now       {len(pending)}")
    print(f"[info] prompt           {prompt_file}")
    print(f"[info] model            {cfg.model} (planner {cfg.resolved_planner_model()}, "
          f"literature {cfg.resolved_literature_model()})")
    print(f"[info] reasoning        {cfg.reasoning} (planner {cfg.planner_reasoning})")
    print(f"[info] max rounds       {cfg.max_rounds}")
    print(f"[info] rank by          {args.rank_by}"
          + (f" (weight {args.rank_weight})" if args.rank_by == "weighted" else ""))
    print(f"[info] literature       backend={cfg.literature_backend}, "
          f"canned context={cfg.canned_lit_context}")
    print(f"[info] workers          {args.workers}")
    print(f"[info] transcripts      {transcript_dir or 'disabled'}")
    print(f"[info] fleming source   {FLEMING_DIR}")

    if args.dry_run:
        print("")
        print("--- instruction for the first pending molecule ---")
        if pending:
            print(prompt_template.replace("{SMILES}", pending[0]["SMILES"]))
        print("--- dry run: nothing was called ---")
        return 0
    if not pending:
        print("[ok] Nothing to do - every molecule is already in the output file.")
        return 0

    # The gateway's own accounting is the authoritative cost figure: the true spend of this
    # run is the drop in `remaining` across it, with no price table in the loop.
    api_key = os.environ.get("OPENAI_API_KEY")
    gateway_base_url = resolve_base_url(api_key, cfg.base_url, {}) if api_key else ""
    quota_before = read_quota(gateway_base_url, api_key) if api_key else None
    if quota_before:
        print(f"[info] gateway budget   {quota_before['remaining']:.2f} "
              f"{quota_before['unit'] or 'USD'} remaining of {quota_before['limit']} "
              f"per {quota_before['interval']}")

    results = []
    started = time.time()

    def report(position, out):
        print(f"[{position}/{len(pending)}] idx={out['index']} {out['status']} "
              f"rank={out['ranking_score']} promise={out['promise_score']} "
              f"(tb={out['tb_score']}, vero={out['vero_score']}) "
              f"{out['runtime_sec']}s ${out['cost_usd'] or '?'}", flush=True)

    if args.workers <= 1:
        for position, row in enumerate(pending, 1):
            out = run_one(row, cfg, args, prompt_template, followup_prompt, transcript_dir)
            arm_io.write_row(output_csv, out)
            results.append(out)
            report(position, out)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, row, cfg, args, prompt_template,
                                   followup_prompt, transcript_dir): row for row in pending}
            for position, future in enumerate(as_completed(futures), 1):
                out = future.result()
                arm_io.write_row(output_csv, out)
                results.append(out)
                report(position, out)

    arm_io.cost_report(results, arm_io.ARM_FLEMING)
    print(f"wall clock for this batch  {(time.time() - started) / 60:.1f} min")

    if quota_before:
        quota_after = read_quota(gateway_base_url, api_key)
        if quota_after and quota_after.get("remaining") is not None:
            spent = quota_before["remaining"] - quota_after["remaining"]
            print(f"gateway-billed spend       ${spent:.4f} "
                  f"({quota_after['remaining']:.2f} remaining) - this is the authoritative "
                  "figure; the estimate above is from the price table")

    print(f"[ok] results -> {output_csv}")
    failed = [row for row in results if row["status"] != arm_io.STATUS_COMPLETE]
    return 1 if failed and len(failed) == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

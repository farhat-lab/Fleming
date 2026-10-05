#!/usr/bin/env python
"""
Arm 2 of 3: an LLM with web search, and nothing else.

The control for the full Fleming arm. Same molecules, same question, same required answer
block, same retry policy, same cost accounting - the only difference is what the system has
to answer with. Fleming has a quantitative Mtb inhibition model, three human cytotoxicity
models, ADMET predictions, a druglikeness filter and five subagents; this arm has the
structure, whatever it already knows, and a web search. Any gap between the two rankings is
what the tools bought.

One Responses API call per molecule, with the web-search tool available so the model can
search as many times as it decides to within that call. If the answer comes back without
the score block, one follow-up call asks for just the block.

    python run_llm_only_eval.py data/mtb_150_smiles.csv \
        --output results/llm_only.csv --limit 3
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
    HarnessConfig,
    MODEL_PRICES,
    UsageTracker,
    WEB_SEARCH_TOOL_TYPES,
    _build_client,
    call_with_retry,
    read_quota,
    resolve_base_url,
    web_search_tool,
)

DEFAULT_PROMPT_FILE = os.path.join(_HERE, "prompts", "llm_only_vero.txt")
FOLLOWUP_PROMPT_FILE = os.path.join(_HERE, "prompts", "score_block_followup.txt")


def extract_text(response):
    """
    The assistant text from a Responses API result.

    `response.output_text` concatenates every text part with no separator, so a model that
    emits two message items produces two sentences glued together on one line - which can
    weld the score block onto the end of a paragraph and defeat the line-anchored parser.
    Join the message items with newlines instead.
    """
    parts = []
    try:
        for item in (getattr(response, "output", None) or []):
            if getattr(item, "type", None) != "message":
                continue
            for piece in (getattr(item, "content", None) or []):
                text = getattr(piece, "text", None)
                if text:
                    parts.append(text)
    except Exception:
        pass
    if parts:
        return "\n".join(parts)
    return getattr(response, "output_text", "") or ""


class LLMOnlyArm:
    """One model, one web-search tool, one molecule at a time."""

    def __init__(self, cfg, model, search_context="medium", verbose=True):
        self.cfg = cfg
        self.model = model
        self.search_context = search_context
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set OPENAI_API_KEY in the environment or fleming/.env.")
        self.api_key = api_key
        self.base_url = resolve_base_url(api_key, cfg.base_url, {})
        self.client = _build_client(api_key, self.base_url, cfg.request_timeout)
        self.tracker = UsageTracker(cfg.prices)
        # o4-mini takes `web_search_preview`; newer models take `web_search`. Start with
        # preview and switch automatically if the API rejects it, so changing the model does
        # not also mean editing this.
        self._tool_types = list(WEB_SEARCH_TOOL_TYPES)
        if verbose:
            print(f"[llm_only] model={self.model} reasoning={cfg.reasoning} "
                  f"endpoint={self.base_url or 'api.openai.com (direct)'}", flush=True)

    def _respond(self, prompt, label):
        tool_types = list(self._tool_types)
        while True:
            kwargs = {
                "model": self.model,
                "input": prompt,
                "tools": web_search_tool(tool_types[0], self.search_context),
            }
            if self.cfg.reasoning:
                kwargs["reasoning"] = {"effort": self.cfg.reasoning}
            try:
                return call_with_retry(
                    lambda: self.client.responses.create(**kwargs),
                    label=label, cfg=self.cfg, tracker=self.tracker,
                    record=lambda t, r: t.record_responses(r, "llm_only", self.model, label),
                    base_url=self.base_url, api_key=self.api_key, who="llm_only")
            except Exception as exc:
                # A rejected tool-type spelling is a different problem from a refusal: fix
                # the spelling and retry immediately rather than treating the call as failed.
                if "web_search" in str(exc) and len(tool_types) > 1:
                    tool_types.pop(0)
                    self._tool_types = tool_types
                    print(f"Warning: [llm_only] retrying {label} with tool type "
                          f"'{tool_types[0]}' ({str(exc)[:120]})", flush=True)
                    continue
                raise

    def run(self, instruction, followup_prompt, do_followup=True, on_event=None):
        self.tracker.reset()
        started = time.time()

        def emit(kind, content):
            if on_event is not None:
                try:
                    on_event({"t": round(time.time() - started, 2), "kind": kind,
                              "content": content})
                except Exception:
                    pass

        emit("instruction", instruction)
        response = self._respond(instruction, "answer")
        answer = extract_text(response)
        emit("answer", answer)

        parsed = scoring.parse_answer(answer)
        if parsed["score_source"] in ("none", "prose") and answer.strip() and do_followup:
            # The model already did the work; one cheap call for just the block beats
            # discarding the molecule. Recorded as `score_source=followup`.
            followup = self._respond(
                f"{instruction}\n\nYour previous answer was:\n\n{answer}\n\n{followup_prompt}",
                "score_followup")
            followup_text = extract_text(followup)
            emit("score_followup", followup_text)
            parsed = scoring.parse_answer(answer, followup_text)

        status = arm_io.STATUS_COMPLETE
        if not answer.strip():
            status = arm_io.STATUS_NO_ANSWER
        elif parsed["score_source"] in ("none", "prose"):
            status = arm_io.STATUS_NO_BLOCK
        return {"status": status, "answer": answer, "parsed": parsed,
                "usage": self.tracker.snapshot(),
                "runtime_sec": round(time.time() - started, 1)}


_LOCAL = __import__("threading").local()


def get_arm(cfg, model, search_context):
    """One client per worker thread. Cheap to build, but no reason to rebuild per molecule."""
    arm = getattr(_LOCAL, "arm", None)
    if arm is None:
        arm = LLMOnlyArm(cfg, model, search_context, verbose=False)
        _LOCAL.arm = arm
    return arm


def run_one(row, cfg, args, prompt_template, followup_prompt, transcript_dir):
    started = time.time()
    index, smiles = row["index"], row["SMILES"]
    instruction = prompt_template.replace("{SMILES}", smiles)

    transcript = arm_io.TranscriptWriter(transcript_dir, index, {
        "arm": arm_io.ARM_LLM_ONLY, "index": index, "SMILES": smiles, "model": args.model,
        "reasoning": cfg.reasoning, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }) if transcript_dir else None

    out = arm_io.blank_row(row, arm_io.ARM_LLM_ONLY)
    try:
        arm = get_arm(cfg, args.model, args.search_context)
        result = arm.run(instruction, followup_prompt, not args.no_followup,
                         on_event=(transcript.write if transcript else None))
        arm_io.apply_scores(out, result["parsed"], mode=args.rank_by, weight=args.rank_weight)
        arm_io.apply_usage(out, result["usage"])
        out.update({"status": result["status"], "answer": result["answer"],
                    "tools_used": "web_search" if out["n_web_searches"] else ""})
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
        description="Run the LLM + web-search control arm over a CSV of molecules.")
    parser.add_argument("input_csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--tb-col", default="Y_TB")
    parser.add_argument("--vero-col", default="Y_Vero")
    parser.add_argument("--group-col", default="selection_group")
    parser.add_argument("--limit", type=int, default=None,
                        help="Applied AFTER the resume filter, so repeated calls batch.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Molecules in parallel. This arm holds no models, so it can run "
                        "far wider than the Fleming arm.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.6-sol",
                        help="The model under test. Defaults to the same model that does "
                        "Fleming's reasoning, so the arms differ in tools, not in model.")
    parser.add_argument("--reasoning", default=None)
    parser.add_argument("--search-context", default="medium",
                        choices=("low", "medium", "high"),
                        help="Web-search context size. Measured as cost-neutral on the "
                        "inhibition benchmark - low/medium/high were within 0%% of each other.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-budget", type=float, default=None)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--rank-by", choices=scoring.RANKING_MODES, default="promise")
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--no-followup", action="store_true")
    parser.add_argument("--no-transcripts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

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

    cfg = HarnessConfig()
    for attribute, value in (("reasoning", args.reasoning),
                             ("request_timeout", args.request_timeout),
                             ("max_retries", args.max_retries),
                             ("retry_budget", args.retry_budget)):
        if value is not None:
            setattr(cfg, attribute, value)
    cfg.base_url = args.base_url
    cfg.prices = {k: dict(v) for k, v in MODEL_PRICES.items()}

    rows = arm_io.read_input(input_csv, args.smiles_col, args.tb_col, args.vero_col,
                             args.group_col)
    if args.start is not None:
        rows = [row for row in rows if row["index"] >= args.start]
    if args.end is not None:
        rows = [row for row in rows if row["index"] < args.end]
    done = arm_io.already_done(output_csv)
    pending = [row for row in rows if row["index"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]

    transcript_dir = None if args.no_transcripts else \
        os.path.splitext(output_csv)[0] + "_transcripts"

    print(f"[info] arm              {arm_io.ARM_LLM_ONLY}")
    print(f"[info] input            {input_csv} ({len(rows)} molecules)")
    print(f"[info] output           {output_csv}")
    print(f"[info] already done     {len(done)}")
    print(f"[info] to run now       {len(pending)}")
    print(f"[info] prompt           {prompt_file}")
    print(f"[info] model            {args.model} (reasoning {cfg.reasoning})")
    print(f"[info] web search       enabled, context={args.search_context}")
    print(f"[info] rank by          {args.rank_by}")
    print(f"[info] workers          {args.workers}")
    print(f"[info] transcripts      {transcript_dir or 'disabled'}")

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

    api_key = os.environ.get("OPENAI_API_KEY")
    gateway_base_url = resolve_base_url(api_key, cfg.base_url, {}) if api_key else ""
    quota_before = read_quota(gateway_base_url, api_key) if api_key else None
    if quota_before:
        print(f"[info] gateway budget   {quota_before['remaining']:.2f} "
              f"{quota_before['unit'] or 'USD'} remaining")

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

    arm_io.cost_report(results, arm_io.ARM_LLM_ONLY)
    print(f"wall clock for this batch  {(time.time() - started) / 60:.1f} min")
    if quota_before:
        quota_after = read_quota(gateway_base_url, api_key)
        if quota_after and quota_after.get("remaining") is not None:
            print(f"gateway-billed spend       "
                  f"${quota_before['remaining'] - quota_after['remaining']:.4f}")
    print(f"[ok] results -> {output_csv}")
    failed = [row for row in results if row["status"] != arm_io.STATUS_COMPLETE]
    return 1 if failed and len(failed) == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

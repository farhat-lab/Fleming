#!/usr/bin/env python
"""
The output schema and file handling shared by every arm of this benchmark.

Three arms are compared - full Fleming, an LLM with web search only, and the chemprop
inhibition model alone - and they must produce byte-identical columns, or the comparison
quietly becomes a comparison of parsers. So the schema, the row builder, the resume logic
and the cost report live here once, and each driver only supplies the part that is actually
different: how it gets an answer for a molecule.
"""

import csv
import json
import os
import threading

import scoring

# `arm` is recorded on every row so results can be concatenated without losing track of
# which system produced which ranking.
ARM_FLEMING = "fleming"
ARM_LLM_ONLY = "llm_only"
ARM_CHEMPROP = "chemprop"

STATUS_COMPLETE = "complete"            # a final answer with a parseable score block
STATUS_NO_BLOCK = "no_score_block"      # an answer, but no scores even after the follow-up
STATUS_NO_ANSWER = "no_answer"          # finished without a final answer
STATUS_BLOCKED = "blocked"              # the system reported it could not do the task
STATUS_AWAITING_USER = "awaiting_user"  # stopped on a question (should not happen)
STATUS_FAILED = "failed"                # aborted on an exception
# An agent run in which no subagent and no quantitative tool ever ran. It may still carry a
# full score block - the follow-up call will happily extract one - but those scores came
# from the orchestrator reasoning on its own, with no model output and no literature behind
# them. That is the LLM-only arm by a more expensive route, and counting it as an agent
# result would silently contaminate the comparison this benchmark exists to make.
STATUS_NO_TOOLS = "no_tools"

# Tools that do not constitute evidence: bookkeeping and display, not measurement.
NON_EVIDENCE_TOOLS = frozenset({
    "FINISHED", "ask_user", "display_smiles", "display_smiles_batch",
    "display_molecule", "display_molecules_batch", "run_python_code",
})


def ran_any_evidence_tool(tools_used, subagents_used):
    """Did this run actually gather evidence, rather than only talk and display?"""
    if subagents_used:
        return True
    return any(tool not in NON_EVIDENCE_TOOLS for tool in (tools_used or []))

OUTPUT_FIELDS = [
    # Identity and ground truth. Labels are carried through untouched and never shown to
    # any arm - only the SMILES reaches a prompt.
    "index", "SMILES", "Y_TB", "Y_Vero", "selection_group", "target",
    "arm", "status", "score_source",
    # What the arm said.
    # `promise_score` is the 0-10 selectivity axis (labelled SELECTIVITY_SCORE in the
    # current prompt, PROMISE_SCORE in earlier ones); `plausibility_score` is the separate
    # 0-10 "worth advancing as an antibiotic" axis added later.
    "tb_score", "vero_score", "promise_score", "plausibility_score",
    # What it is ranked by.
    "ranking_score", "ranking_mode", "tiebreak",
    # What the underlying models said, where the arm had any. Blank for llm_only by design.
    "raw_inhibition", "inhibition_uncertainty",
    "raw_hepg2", "raw_hskmc", "raw_imr90", "hepg2_uncertainty",
    # What actually ran.
    "subagents_used", "tools_used", "n_lit_reviews", "n_web_searches",
    "n_api_calls", "calls_by_agent",
    "n_llm_retries", "n_bio_policy", "n_refusals", "n_timeouts",
    "n_ask_user", "n_tool_errors",
    # Cost.
    "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
    "cost_usd", "unpriced_calls", "runtime_sec",
    # Text.
    "error", "rationale", "answer", "plan",
]

LIT_TOOL_LABELS = ("brief_literature_review", "literature_review", "pqa_molecular_analysis")

_WRITE_LOCK = threading.Lock()


# ======================================================================================
# Input
# ======================================================================================

def read_input(path, smiles_col="SMILES", tb_col="Y_TB", vero_col="Y_Vero",
               group_col="selection_group"):
    """
    Load the molecule list.

    A blank `Y_Vero` is kept as a blank, not coerced to 0 - in mtb_150 the TB non-hits were
    never counter-screened, and `scoring.target_label` is what decides what that means.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        if smiles_col not in reader.fieldnames:
            raise ValueError(
                f"{path} has no {smiles_col!r} column (found: {', '.join(reader.fieldnames)}). "
                "Use --smiles-col to point at the right one.")
        rows = []
        for index, row in enumerate(reader):
            smiles = (row.get(smiles_col) or "").strip()
            if not smiles:
                continue
            if smiles.isdigit():
                # mtb_150.csv ships integers in its SMILES column; they are row ids from the
                # source frame, not structures. Fail loudly rather than sending "2224" to a
                # model and getting a confident answer about nothing.
                raise ValueError(
                    f"{path} row {index}: the {smiles_col!r} column contains {smiles!r}, "
                    "which is an integer, not a SMILES string. Rebuild the input with "
                    "make_mtb150_input.py, which recovers the structures from the PubChem "
                    "bulk file by SID.")
            rows.append({
                "index": index,
                "SMILES": smiles,
                "Y_TB": (row.get(tb_col) or "").strip(),
                "Y_Vero": (row.get(vero_col) or "").strip(),
                "selection_group": (row.get(group_col) or "").strip(),
            })
    return rows


def already_done(path):
    """Indices already in the output file, so a resumed run skips them."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                done.add(int(row["index"]))
            except (KeyError, TypeError, ValueError):
                continue
    return done


# ======================================================================================
# Output
# ======================================================================================

def write_row(path, row):
    """Append one row, flushed and fsynced, so a killed job keeps every finished molecule."""
    with _WRITE_LOCK:
        exists = os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def blank_row(input_row, arm):
    """A row with identity, ground truth and arm filled in, and everything else empty."""
    out = {field: "" for field in OUTPUT_FIELDS}

    def _num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    y_tb, y_vero = _num(input_row.get("Y_TB")), _num(input_row.get("Y_Vero"))
    target = scoring.target_label(y_tb, y_vero)
    out.update({
        "index": input_row["index"],
        "SMILES": input_row["SMILES"],
        "Y_TB": input_row.get("Y_TB", ""),
        "Y_Vero": input_row.get("Y_Vero", ""),
        "selection_group": input_row.get("selection_group") or scoring.group_of(y_tb, y_vero),
        "target": "" if target is None else target,
        "arm": arm,
    })
    return out


def apply_scores(out, parsed, mode="promise", weight=0.5):
    """Put a parse_answer() result and the derived ranking value onto a row."""
    rank, used_mode = scoring.ranking_score(
        parsed["tb_score"], parsed["vero_score"], parsed["promise_score"],
        mode=mode, weight=weight)
    out.update({
        "score_source": parsed["score_source"],
        "tb_score": parsed["tb_score"],
        "vero_score": parsed["vero_score"],
        "promise_score": parsed["promise_score"],
        "plausibility_score": parsed["plausibility_score"],
        "ranking_score": rank,
        "ranking_mode": used_mode,
        "tiebreak": scoring.tiebreak_score(parsed["tb_score"], parsed["vero_score"]),
        "raw_inhibition": parsed["raw_inhibition"],
        "inhibition_uncertainty": parsed["inhibition_uncertainty"],
        "raw_hepg2": parsed["raw_hepg2"],
        "raw_hskmc": parsed["raw_hskmc"],
        "raw_imr90": parsed["raw_imr90"],
        "hepg2_uncertainty": parsed["hepg2_uncertainty"],
        "rationale": parsed["rationale"],
    })
    return out


def apply_usage(out, usage):
    """Put a UsageTracker snapshot onto a row."""
    out.update({
        "n_lit_reviews": sum(1 for call in usage.get("calls", [])
                             if call.get("label") in LIT_TOOL_LABELS),
        "n_web_searches": usage["n_web_searches"],
        "n_api_calls": usage["n_api_calls"],
        "calls_by_agent": json.dumps(usage.get("calls_by_agent", {})),
        "n_llm_retries": usage["n_llm_retries"],
        "n_bio_policy": usage["n_bio_policy"],
        "n_refusals": usage["n_refusals"],
        "n_timeouts": usage["n_timeouts"],
        "n_ask_user": usage["n_ask_user"],
        "n_tool_errors": usage["n_tool_errors"],
        "input_tokens": usage["input_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost_usd": usage["cost_usd"] if usage["cost_usd"] is not None else "",
        "unpriced_calls": usage["unpriced_calls"],
    })
    return out


class TranscriptWriter:
    """
    A molecule's transcript, written step by step rather than at the end.

    A run is many minutes and real money; if the job dies at molecule 60 the first 59
    transcripts must still be on disk, and so must the partial 60th. Each entry is flushed
    and fsynced as it is produced, and the file is renamed from `.jsonl.partial` to `.jsonl`
    only once the molecule finishes - so an interrupted molecule is identifiable by its
    filename rather than looking like a complete one.
    """

    def __init__(self, directory, index, header):
        os.makedirs(directory, exist_ok=True)
        self.final_path = os.path.join(directory, f"{index:05d}.jsonl")
        self.path = self.final_path + ".partial"
        self._handle = open(self.path, "w")
        self.write({"kind": "header", **header})

    def write(self, entry):
        try:
            self._handle.write(json.dumps(entry, default=str) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except Exception:
            pass

    def close(self, complete=True):
        try:
            self._handle.close()
        except Exception:
            pass
        if complete:
            try:
                os.replace(self.path, self.final_path)
            except Exception:
                pass


# ======================================================================================
# Reporting
# ======================================================================================

def cost_report(rows, arm):
    """Cost and behaviour summary. Says what it left out rather than quietly rounding it in."""
    def total(field):
        return sum(int(float(row[field])) for row in rows
                   if str(row.get(field, "")).strip() not in ("", "None"))

    n = len(rows)
    if not n:
        return
    priced = [row for row in rows if str(row.get("cost_usd", "")).strip() not in ("", "None")]
    complete = [row for row in rows if row["status"] == STATUS_COMPLETE]

    print("")
    print("--- run summary -------------------------------------------------------")
    print(f"arm                        {arm}")
    print(f"molecules attempted        {n}")
    by_status = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<24} {count}")
    print(f"api calls                  {total('n_api_calls')} "
          f"({total('n_api_calls') / n:.1f} per molecule)")
    print(f"literature/web calls       {total('n_lit_reviews')} lit, "
          f"{total('n_web_searches')} web-search tool calls")
    print(f"retries                    {total('n_llm_retries')} "
          f"({total('n_bio_policy')} bio-policy refusals, {total('n_timeouts')} timeouts)")
    if total("n_ask_user"):
        print(f"ask_user auto-answered     {total('n_ask_user')}")
    print(f"tokens                     {total('input_tokens')} in "
          f"({total('cached_tokens')} cached), {total('output_tokens')} out")
    runtimes = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
    if runtimes:
        print(f"wall clock                 {sum(runtimes) / 60:.1f} min total, "
              f"{sum(runtimes) / len(runtimes) / 60:.1f} min per molecule")

    if priced:
        spend = sum(float(row["cost_usd"]) for row in priced)
        print(f"estimated cost             ${spend:.4f} over {len(priced)} molecules "
              f"(${spend / len(priced):.4f} each)")
        if total("unpriced_calls"):
            print(f"  NOT INCLUDED: {total('unpriced_calls')} calls on models with no price "
                  "entry. Add them to MODEL_PRICES in fleming_harness.py.")
    elif arm != ARM_CHEMPROP:
        print("estimated cost             no prices known for the models used; tokens only")

    if complete:
        sources = {}
        for row in complete:
            sources[row["score_source"]] = sources.get(row["score_source"], 0) + 1
        print(f"score provenance           {sources}")
        agents = {}
        for row in complete:
            for name in (row["subagents_used"] or "").split(";"):
                if name:
                    agents[name] = agents.get(name, 0) + 1
        if agents:
            print("subagents used             " + ", ".join(
                f"{name} {count}/{len(complete)}" for name, count in sorted(agents.items())))
    print("-----------------------------------------------------------------------")

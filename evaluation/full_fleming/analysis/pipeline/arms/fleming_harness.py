#!/usr/bin/env python
"""
Headless harness around the *real* Fleming multi-agent system.

  1. Transport. Fleming's `llm_infer` builds a bare `OpenAI(api_key=...)` client against
     api.openai.com. Here it is rebuilt against the Harvard HUIT gateway (or direct
     OpenAI), with a retry policy, a per-request timeout, and token/cost accounting for
     every call any agent makes.
  2. Models. `utils.utils.MODEL / PLANNER_MODEL / PLANNER_REASONING` are patched, so all
     agents run on the model this harness was configured with rather than the app defaults.
  3. Non-interactivity. `ask_user` cannot block a sweep, so it returns a fixed instruction
     to proceed with defaults, and the harness counts how often that happened.
  4. Termination. Two of Fleming's loops are unbounded in the app (the refusal retry in
     `llm_infer` is a bare `while True`, and a subagent can ask the user forever). Both get
     hard caps here. 
Everything else - the planner, the plan gate, the tool dispatch by `eval()`, the parallel
subagent runner, the summarization prompt - is untouched, so results describe Fleming,
not this file.

Usage:
    from fleming_harness import HarnessConfig, FlemingRunner

    runner = FlemingRunner(HarnessConfig(model="gpt-5.6-sol"))
    result = runner.run("CCOc1ccccc1", "Assess TB inhibition and liver cytotoxicity ...")
    print(result["answer"], result["usage"]["cost_usd"])
"""

import json
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
# evaluation/full_fleming -> evaluation -> repo root -> fleming/
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
# Overridable, because this file is also vendored into analysis_code/ for publication,
# where the two-levels-up assumption no longer points at the app.
FLEMING_DIR = os.environ.get("FLEMING_DIR") or os.path.join(_REPO_ROOT, "fleming")

try:
    import dotenv
    # load_dotenv() only walks up from the working directory, which misses the repo's key
    # file when this module is imported from somewhere else. Look in the obvious places.
    dotenv.load_dotenv()
    for _candidate in (os.path.join(_HERE, ".env"), os.path.join(FLEMING_DIR, ".env")):
        if os.path.exists(_candidate):
            dotenv.load_dotenv(_candidate, override=False)
except Exception:
    pass

from openai import OpenAI  # noqa: E402  (after dotenv so env vars are set)

# --------------------------------------------------------------------------------------
# Defaults. Every one is overridable by environment variable (FLEMING_EVAL_*) and by the
# corresponding HarnessConfig field, which is what the CLI flags set.
# --------------------------------------------------------------------------------------
MODEL = os.environ.get("FLEMING_EVAL_MODEL", "gpt-5.6-sol")
PLANNER_MODEL = os.environ.get("FLEMING_EVAL_PLANNER_MODEL") or None
REASONING_EFFORT = os.environ.get("FLEMING_EVAL_REASONING", "medium")
PLANNER_REASONING = os.environ.get("FLEMING_EVAL_PLANNER_REASONING", "high")
# Literature retrieval is one call but a large share of the spend, because the retrieved
# pages are billed as input tokens. On the inhibition benchmark moving it to o4-mini cut
# cost per molecule from $0.68 to $0.20 with no measurable change in the answers (score
# correlation 0.994), so the hybrid split is the default here: everything on MODEL except
# the literature/web-search calls. Set FLEMING_EVAL_LITERATURE_MODEL to the empty string,
# or pass --literature-model "", to put literature back on MODEL too.
_LITERATURE_MODEL_ENV = os.environ.get("FLEMING_EVAL_LITERATURE_MODEL")
LITERATURE_MODEL = "o4-mini" if _LITERATURE_MODEL_ENV is None else (_LITERATURE_MODEL_ENV or None)

HARVARD_BASE_URL = "https://go.apis.huit.harvard.edu/ais-openai-direct/v2"
BASE_URL_ENV = os.environ.get("FLEMING_EVAL_BASE_URL")  # None = unset, "" = direct OpenAI
QUOTA_PATH = "/apigee/quota"

# Fleming's own default is max_round=20. A full dual-objective run needs the plan turn,
# two subagent calls, a display call and the summary, so ~8-12 rounds is typical.
MAX_ROUNDS = int(os.environ.get("FLEMING_EVAL_MAX_ROUNDS", "14"))
REQUEST_TIMEOUT = float(os.environ.get("FLEMING_EVAL_TIMEOUT", "900"))
MAX_LLM_RETRIES = int(os.environ.get("FLEMING_EVAL_MAX_RETRIES", "3"))
RETRY_SLEEP_BUDGET = float(os.environ.get("FLEMING_EVAL_RETRY_BUDGET", "900"))
# Timeouts get a tighter cap than refusals: a refusal is free and often clears on a later
# attempt, but a timeout has already burned a full request_timeout and resending the same
# slow query usually burns another.
MAX_TIMEOUT_RETRIES = int(os.environ.get("FLEMING_EVAL_MAX_TIMEOUT_RETRIES", "2"))
# The app retries a refusal-shaped reply forever (`while True` in llm_infer). Capped here.
MAX_REFUSAL_RETRIES = int(os.environ.get("FLEMING_EVAL_MAX_REFUSAL_RETRIES", "3"))
# A subagent that keeps calling ask_user would otherwise never finish.
MAX_ASK_USER = int(os.environ.get("FLEMING_EVAL_MAX_ASK_USER", "3"))

WEB_SEARCH_TOOL_TYPES = ["web_search_preview", "web_search"]
WEB_SEARCH_USER_LOCATION = {
    "type": "approximate", "country": "US", "city": "Boston",
    "region": "Massachusetts", "timezone": "America/New_York",
}

# USD per 1M tokens. Priced per call against the model that served it, because a run can
# mix models (flagship reasoning, cheap literature search) and billing everything at one
# rate overstates a mixed run several-fold. Verify against the current price sheet before
# quoting any number from these.
MODEL_PRICES = {
    "gpt-5.6-sol":  {"input_per_mtok": 4.00, "cached_input_per_mtok": 0.40,
                     "output_per_mtok": 20.00, "web_search_per_call": None},
    "gpt-5.6":      {"input_per_mtok": 4.00, "cached_input_per_mtok": 0.40,
                     "output_per_mtok": 20.00, "web_search_per_call": None},
    "o4-mini":      {"input_per_mtok": 1.10, "cached_input_per_mtok": 0.275,
                     "output_per_mtok": 4.40, "web_search_per_call": None},
}

# Fleming's `ask_user` has no answer available in a batch sweep. Returning an error would
# make it retry; returning this makes it proceed, and `n_ask_user` records that it happened
# so a molecule whose result depended on a missing answer is identifiable.
ASK_USER_REPLY = (
    "No interactive user is available in this batch evaluation, so no additional input can "
    "be provided. Proceed with the task using reasonable defaults and the tools you have, "
    "and state any assumption you had to make in your final answer. Do not ask again."
)

# The known-bad path in PQA_Agent.get_general_lit_info: it returns a hard-coded paragraph
# about liver-toxicity mechanisms and never reaches `pqapi.agent_query`, so every
# `literature_review(...)` call is prepended with that same canned text as if it had been
# retrieved. That is a constant, not per-molecule label leakage, but it is fabricated
# provenance and it happens to be about exactly the endpoint this benchmark scores.
CANNED_LIT_MARKER = "there are several key structural features and mechanisms that contribute to liver toxicity"


def _env_float(name):
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else None
    except ValueError:
        print(f"Warning: {name}={raw!r} is not a number; ignoring.")
        return None


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class HarnessConfig:
    """Everything the harness needs to know. The CLI builds one of these."""
    model: str = MODEL
    planner_model: Optional[str] = PLANNER_MODEL
    literature_model: Optional[str] = LITERATURE_MODEL
    reasoning: str = REASONING_EFFORT
    planner_reasoning: str = PLANNER_REASONING
    base_url: Optional[str] = None          # None = resolve automatically
    config_path: Optional[str] = None       # None = fleming/config.yaml
    max_rounds: int = MAX_ROUNDS
    request_timeout: float = REQUEST_TIMEOUT
    max_retries: int = MAX_LLM_RETRIES
    retry_budget: float = RETRY_SLEEP_BUDGET
    prices: Dict[str, Dict[str, Optional[float]]] = field(default_factory=lambda: {
        k: dict(v) for k, v in MODEL_PRICES.items()})
    # "web"  -> the PQA-backed tools (literature_review, pqa_molecular_analysis) are routed
    #           to the web-search implementation. This is the default because PQA calls take
    #           ~5 minutes each and the token in fleming/config.yaml has expired.
    # "pqa"  -> leave them exactly as the app has them.
    literature_backend: str = "web"
    # Whether to keep the canned liver-toxicity paragraph described at CANNED_LIT_MARKER.
    # "keep" is faithful to the app; "drop" removes the fabricated provenance.
    canned_lit_context: str = "keep"
    search_context_size: str = "medium"
    verbose: bool = True

    def resolved_planner_model(self) -> str:
        return self.planner_model or self.model

    def resolved_literature_model(self) -> str:
        return self.literature_model or self.model


# ======================================================================================
# Usage / cost accounting, shared across every agent in one molecule's run
# ======================================================================================

class UsageTracker:
    """
    Token, call and cost accounting for one molecule.

    Every agent instance in the Fleming tree holds a reference to the same tracker, so a
    subagent's literature search and Fleming's own summary land in one total. Subagents run
    in their own threads (`run_parallel_agents`), hence the lock.
    """

    def __init__(self, price_table: Dict[str, dict]):
        self.price_table = {k: dict(v) for k, v in (price_table or {}).items()}
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.calls: List[dict] = []
            self.retries = 0
            self.bio_policy_hits = 0
            self.refusals = 0
            self.timeouts = 0
            self.ask_user_calls = 0
            self.tool_errors = 0
            self.unpriced_calls = 0

    # ------------------------------------------------------------------ recording

    def record_chat(self, response, agent: str, model: str, label: str):
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        cached = reasoning = 0
        try:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        except Exception:
            cached = 0
        try:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        except Exception:
            reasoning = 0
        self._append(agent, model, label, prompt, cached, completion, reasoning, 0)

    def record_responses(self, response, agent: str, model: str, label: str):
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        cached = reasoning = 0
        try:
            details = getattr(usage, "input_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        except Exception:
            cached = 0
        try:
            details = getattr(usage, "output_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        except Exception:
            reasoning = 0
        web_calls = 0
        try:
            for item in (getattr(response, "output", None) or []):
                if "web_search" in str(getattr(item, "type", "")):
                    web_calls += 1
        except Exception:
            pass
        self._append(agent, model, label, input_tokens, cached, output_tokens,
                     reasoning, web_calls)

    def _append(self, agent, model, label, input_tokens, cached, output_tokens,
                reasoning, web_calls):
        with self._lock:
            self.calls.append({
                "agent": agent, "model": model, "label": label,
                "input_tokens": input_tokens, "cached_input_tokens": cached,
                "output_tokens": output_tokens, "reasoning_tokens": reasoning,
                "web_search_calls": web_calls,
            })

    def note_retry(self, is_bio_policy: bool = False, is_timeout: bool = False):
        with self._lock:
            self.retries += 1
            if is_bio_policy:
                self.bio_policy_hits += 1
            if is_timeout:
                self.timeouts += 1

    def note_refusal(self):
        with self._lock:
            self.refusals += 1

    def note_ask_user(self):
        with self._lock:
            self.ask_user_calls += 1

    def note_tool_error(self):
        with self._lock:
            self.tool_errors += 1

    # ------------------------------------------------------------------ costing

    def _prices_for(self, model_name):
        """Rates for one model id, tolerating dated suffixes like o4-mini-2025-04-16."""
        if model_name and model_name in self.price_table:
            return self.price_table[model_name]
        # Longest prefix wins, so "gpt-5.6-sol" is not matched by a "gpt-5.6" entry.
        best = None
        for known, prices in self.price_table.items():
            if known and model_name and str(model_name).startswith(known):
                if best is None or len(known) > len(best[0]):
                    best = (known, prices)
        return best[1] if best else None

    def cost_usd(self):
        """
        Dollar cost of everything recorded, priced per call against the serving model.
        None when no rates are known for anything that ran; `unpriced_calls` says how many
        calls were skipped, so a partial total is never mistaken for a complete one.
        """
        with self._lock:
            calls = list(self.calls)
        self.unpriced_calls = 0
        total = 0.0
        priced_any = False
        for call in calls:
            prices = self._prices_for(call.get("model")) or {}
            if all(prices.get(k) is None for k in
                   ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok",
                    "web_search_per_call")):
                self.unpriced_calls += 1
                continue
            priced_any = True
            cached = call.get("cached_input_tokens", 0) or 0
            cached_rate = prices.get("cached_input_per_mtok")
            if cached_rate is None:
                # No separate cached rate: bill every input token at the standard rate.
                uncached, cached, cached_rate = call.get("input_tokens", 0) or 0, 0, 0.0
            else:
                uncached = max((call.get("input_tokens", 0) or 0) - cached, 0)
            total += uncached / 1_000_000 * (prices.get("input_per_mtok") or 0.0)
            total += cached / 1_000_000 * cached_rate
            total += (call.get("output_tokens", 0) or 0) / 1_000_000 * (prices.get("output_per_mtok") or 0.0)
            total += (call.get("web_search_calls", 0) or 0) * (prices.get("web_search_per_call") or 0.0)
        return round(total, 6) if priced_any else None

    def snapshot(self) -> dict:
        with self._lock:
            calls = list(self.calls)
            retries, bio, refusals = self.retries, self.bio_policy_hits, self.refusals
            timeouts, asks, tool_errors = self.timeouts, self.ask_user_calls, self.tool_errors
        cost = self.cost_usd()
        by_agent = {}
        for call in calls:
            by_agent[call["agent"]] = by_agent.get(call["agent"], 0) + 1
        input_tokens = sum(c["input_tokens"] for c in calls)
        output_tokens = sum(c["output_tokens"] for c in calls)
        return {
            "n_api_calls": len(calls),
            "input_tokens": input_tokens,
            "cached_tokens": sum(c["cached_input_tokens"] for c in calls),
            "output_tokens": output_tokens,
            "reasoning_tokens": sum(c["reasoning_tokens"] for c in calls),
            "total_tokens": input_tokens + output_tokens,
            "n_web_searches": sum(c["web_search_calls"] for c in calls),
            "n_llm_retries": retries,
            "n_bio_policy": bio,
            "n_refusals": refusals,
            "n_timeouts": timeouts,
            "n_ask_user": asks,
            "n_tool_errors": tool_errors,
            "cost_usd": cost,
            "unpriced_calls": self.unpriced_calls,
            "calls_by_agent": by_agent,
            "calls": calls,
        }


# ======================================================================================
# Patching the real Fleming
# ======================================================================================

_PATCH_LOCK = threading.Lock()
_PATCHED = {"done": False}

REFUSAL_PHRASES = (
    "I'm sorry, I can't assist with that",
    "I'm unable to view the image",
    "I'm unable to provide a definitive answer",
)


def resolve_base_url(api_key: str, configured: Optional[str], config: dict) -> str:
    """
    Endpoint resolution, in order: explicit argument, FLEMING_EVAL_BASE_URL (set it to the
    empty string to force direct OpenAI), the config file, then auto-detect - a key that
    does not look like an OpenAI `sk-` key is assumed to be a gateway key.
    """
    if configured is not None:
        base = configured
    elif BASE_URL_ENV is not None:
        base = BASE_URL_ENV
    elif config.get("openai_base_url") is not None:
        base = config["openai_base_url"]
    elif api_key and not api_key.startswith("sk-"):
        base = HARVARD_BASE_URL
    else:
        base = ""
    return (base or "").rstrip("/")


def _build_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": 1}
    if base_url:
        kwargs["base_url"] = base_url + "/"
        # The Harvard gateway authenticates on this header; the SDK's own
        # `Authorization: Bearer` is ignored there.
        kwargs["default_headers"] = {"api-key": api_key}
    return OpenAI(**kwargs)


def _retry_delay(attempt: int, is_bio_policy: bool) -> float:
    return min(5 * (2 ** (attempt - 1)), 60) if is_bio_policy else min(2 ** attempt, 30)


def _check_budget_exhausted(base_url: str, api_key: str, message: str):
    """
    On the Harvard gateway a 429 means the app's monthly budget is spent, not that we are
    going too fast. Retrying cannot succeed, so fail loudly and immediately.
    """
    if not base_url:
        return
    if "429" not in message and "Too Many Requests" not in message:
        return
    detail = ""
    info = read_quota(base_url, api_key)
    if info and info.get("remaining") is not None:
        detail = (f" Gateway reports {info['remaining']:.2f} {info.get('unit') or 'USD'} "
                  f"remaining of {info.get('limit')} per {info.get('interval')}.")
    raise RuntimeError(
        "The Harvard gateway returned 429, which means this app's budget for the period is "
        f"exhausted - it will not clear by retrying.{detail} Raise the limit in the API "
        "portal app description, or wait for the period to reset."
    )


def read_quota(base_url: str, api_key: str) -> Optional[dict]:
    """
    Spend against the gateway budget, or None if there is no gateway.

    This is the authoritative cost figure: the gateway bills at the provider's own rates
    and reports what is left, so a run's true spend is the drop in `remaining` across it -
    no price table to keep in sync with a price sheet.
    """
    if not base_url:
        return None
    try:
        import urllib.request
        request = urllib.request.Request(base_url + QUOTA_PATH,
                                         headers={"api-key": api_key})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        print(f"Warning: could not read the gateway quota ({str(exc)[:160]})")
        return None
    quota = payload.get("quota") or {}
    try:
        return {
            "app": payload.get("apigee_app_name"),
            "limit": float(quota.get("limit")) if quota.get("limit") else None,
            "unit": quota.get("limit_unit"),
            "interval": f"{quota.get('interval', '')} {quota.get('unit', '')}".strip(),
            "remaining": float(payload.get("remaining_limit")),
            "remaining_pct": float(payload.get("remaining_percentage")),
        }
    except (TypeError, ValueError):
        return None


def call_with_retry(make_call, *, label, cfg, tracker=None, record=None,
                    base_url="", api_key="", who=""):
    """
    One API call under this benchmark's retry policy, used by every arm.

    Two separate limits, because the count alone is the wrong knob: a refusal that is
    deterministic on content will not clear however many times it is resent, and with
    backoff each extra attempt costs wall clock. `cfg.max_retries` caps the attempts;
    `cfg.retry_budget` caps the TOTAL seconds the call may spend, attempts included, so
    raising the count cannot silently turn into hours of waiting. Timeouts get their own,
    tighter cap: a refusal is free and often clears on a later attempt, but a timeout has
    already burned a full request_timeout and resending the same slow query usually burns
    another.

    It lives here, at module level, so the Fleming arm and the LLM-only arm retry
    identically - a difference in retry policy between two arms is a difference in how much
    each one gets to try, which would land in the results as if it were a difference in
    capability.
    """
    started = time.time()
    attempt = 0
    timeouts = 0
    while True:
        attempt += 1
        try:
            response = make_call()
            if tracker is not None and record is not None:
                record(tracker, response)
            return response
        except Exception as exc:
            message = str(exc)
            is_bio_policy = "bio_policy" in message or "biological risk" in message
            _check_budget_exhausted(base_url, api_key, message)
            is_timeout = ("timed out" in message.lower() or "timeout" in message.lower()
                          or type(exc).__name__ in ("APITimeoutError", "Timeout"))
            if is_timeout:
                timeouts += 1
            delay = _retry_delay(attempt, is_bio_policy)
            elapsed = time.time() - started
            if tracker is not None:
                tracker.note_retry(is_bio_policy, is_timeout)
            if (attempt >= cfg.max_retries
                    or (elapsed + delay) > cfg.retry_budget
                    or timeouts >= MAX_TIMEOUT_RETRIES):
                raise
            print(f"Warning: [{who}] {label} failed ({message[:160]}); retrying in {delay}s "
                  f"(attempt {attempt + 1}/{cfg.max_retries}, {elapsed:.0f}s spent)",
                  flush=True)
            time.sleep(delay)


def web_search_tool(tool_type="web_search_preview", context_size="medium"):
    """The web-search tool definition both API arms pass to the Responses API."""
    return [{"type": tool_type,
             "search_context_size": context_size,
             "user_location": WEB_SEARCH_USER_LOCATION}]


def install_patches(cfg: HarnessConfig):
    """
    Replace Fleming's transport and interactivity in place. Idempotent and safe to call
    from several threads; the first call wins and later ones only refresh the config.
    """
    with _PATCH_LOCK:
        _CFG["cfg"] = cfg
        if _PATCHED["done"]:
            return
        _install_patches_locked(cfg)
        _PATCHED["done"] = True


_CFG: Dict[str, HarnessConfig] = {}


def _cfg() -> HarnessConfig:
    return _CFG["cfg"]


def _install_patches_locked(cfg: HarnessConfig):
    import main_fleming
    from utils import utils as fleming_utils
    from utils.pqa_agent import PQA_Agent

    Basic_Agent = fleming_utils.Basic_Agent

    # ---- 1. Models. `utils.utils` holds them as module globals, which the literature
    # tools read at call time; `main_fleming` imported them by value at import time, so
    # both have to be set or the planner keeps calling the app's default model.
    for module in (fleming_utils, main_fleming):
        module.MODEL = cfg.model
        module.PLANNER_MODEL = cfg.resolved_planner_model()
        module.PLANNER_REASONING = cfg.planner_reasoning

    # ---- 2. Per-agent client. Cached on the instance, keyed by the endpoint, so a long
    # sweep does not build a new HTTP client for every call.
    def _client(agent):
        cache = getattr(agent, "_eval_client", None)
        if cache is not None:
            return cache
        api_key = (os.environ.get("OPENAI_API_KEY")
                   or (agent.config or {}).get("openai_api_key"))
        if not api_key:
            raise RuntimeError(
                "No API key found. Set OPENAI_API_KEY in the environment (or in "
                "fleming/.env), or `openai_api_key` in the config YAML.")
        base_url = resolve_base_url(api_key, cfg.base_url, agent.config or {})
        agent._eval_api_key = api_key
        agent._eval_base_url = base_url
        agent._eval_client = _build_client(api_key, base_url, cfg.request_timeout)
        return agent._eval_client

    def _tracker(agent) -> Optional[UsageTracker]:
        return getattr(agent, "_eval_tracker", None)

    def _call_with_retry(agent, label: str, make_call: Callable, record: Callable):
        """One API call, under the shared retry policy. See `call_with_retry`."""
        tracker = _tracker(agent)
        return call_with_retry(
            make_call, label=label, cfg=_cfg(), tracker=tracker,
            record=(lambda t, r: record(t, r)) if record else None,
            base_url=getattr(agent, "_eval_base_url", ""),
            api_key=getattr(agent, "_eval_api_key", ""),
            who=agent.agent_name)

    # ---- 3. llm_infer. Same signature as the original so every existing call site works.
    # Two deliberate differences from the app, both documented in the module docstring:
    # the refusal loop is capped, and `reasoning_effort=None` becomes the configured
    # effort rather than "whatever the endpoint defaults to".
    def llm_infer(self, conversation, temp=0.000000001, max_tokens=1000, image=None,
                  role=None, openai_backend="openai", model=None, reasoning_effort=None):
        cfg_now = _cfg()
        model = model or cfg_now.model
        effort = reasoning_effort or cfg_now.reasoning
        client = _client(self)
        tracker = _tracker(self)
        # `max_tokens` is accepted and ignored, exactly as in the app: it never reaches the
        # API there either. Honouring it would cap reasoning+output together and silently
        # truncate summaries, which would be a behaviour change, not a fix.
        refusals = 0
        while True:
            kwargs = {"model": model, "messages": conversation}
            if effort:
                kwargs["reasoning_effort"] = effort
            label = f"llm_infer:{model}"
            response = _call_with_retry(
                self, label,
                lambda: client.chat.completions.create(**kwargs),
                lambda t, r: t.record_chat(r, self.agent_name, model, label))
            text = response.choices[0].message.content or ""
            if any(phrase in text for phrase in REFUSAL_PHRASES):
                refusals += 1
                if tracker is not None:
                    tracker.note_refusal()
                if refusals >= MAX_REFUSAL_RETRIES:
                    # The app loops here forever. In a sweep that is a hung job, so return
                    # the refusal and let the caller record the molecule as degraded.
                    print(f"Warning: [{self.agent_name}] {refusals} refusal-shaped replies "
                          f"in a row; returning the last one instead of looping.", flush=True)
                    return text
                print(f"Warning: [{self.agent_name}] refusal-shaped reply "
                      f"({refusals}/{MAX_REFUSAL_RETRIES}); retrying.", flush=True)
                continue
            return text

    # ---- 4. Literature tools. These build their own client against api.openai.com in the
    # app, so they need the same treatment, plus the option of a cheaper model.
    def _responses_web_search(self, question: str, label: str) -> str:
        cfg_now = _cfg()
        model = cfg_now.resolved_literature_model()
        client = _client(self)
        tool_types = list(WEB_SEARCH_TOOL_TYPES)
        while tool_types:
            tools = web_search_tool(tool_types[0], cfg_now.search_context_size)
            try:
                response = _call_with_retry(
                    self, label,
                    lambda: client.responses.create(model=model, input=question, tools=tools),
                    lambda t, r: t.record_responses(r, self.agent_name, model, label))
            except Exception as exc:
                # A rejected tool-type spelling is a different problem from a refusal: o4-mini
                # takes `web_search_preview`, newer models take `web_search`. Swap and retry
                # once rather than treating it as a failed search.
                if "web_search" in str(exc) and len(tool_types) > 1:
                    print(f"Warning: [{self.agent_name}] retrying {label} with tool type "
                          f"'{tool_types[1]}' ({str(exc)[:120]})", flush=True)
                    tool_types.pop(0)
                    continue
                raise
            return response.output_text or ""
        return ""

    def brief_literature_review(self, question):
        question = str(question) + " Make sure to return citations and the working link to the sources"
        try:
            result = _responses_web_search(self, question, "brief_literature_review")
        except Exception as exc:
            # Same contract as the app: return an explanatory string rather than raising,
            # so the agent can carry on with what it has.
            message = f"Brief literature review failed due to API error: {exc}"
            print(f"Literature review error: {message}", flush=True)
            return (f"Unable to complete literature review for: {question}. Error: {message}. "
                    "The agent should proceed with available information or try alternative "
                    "approaches.")
        if not result.strip():
            return (f"Literature review completed but returned empty results for query: "
                    f"{question}. This may indicate the specific query didn't match available "
                    "literature or there were API limitations.")
        return result

    def wrapped_literature_review(self, question):
        cfg_now = _cfg()
        question = str(question)
        if cfg_now.canned_lit_context == "keep":
            # Faithful to the app, including the canned paragraph (see CANNED_LIT_MARKER).
            try:
                pqa_input = self.get_general_lit_info(question)
            except Exception as exc:
                pqa_input = f"(unavailable: {exc})"
            question = (question + " Here's the input from a separate literature review, "
                        "though please be sure to conduct your own review and synthesize "
                        f"the two: {pqa_input}")
        question += " Make sure to return citations and the working link to the sources"
        try:
            return _responses_web_search(self, question, "literature_review")
        except Exception as exc:
            return (f"Unable to complete literature review for: {question}. Error: {exc}. "
                    "The agent should proceed with available information.")

    def literature_review(self, question):
        return self.wrapped_literature_review(question)

    def pqa_molecular_analysis(self, molecule, question):
        """
        PQA is a ~5-minute-per-call service and the token shipped in fleming/config.yaml has
        expired, so with `literature_backend="web"` this routes to the web search and says
        so in its own output - the agent is never told it got PQA evidence when it did not.
        """
        if _cfg().literature_backend == "pqa":
            return PQA_Agent.run_molecule(self, molecule, question)
        prompt = (f"Molecule (SMILES): {molecule}\n\nQuestion: {question}\n\n"
                  "Search the literature for evidence about this molecule and close analogs. "
                  "Note: this answer comes from a web literature search, not from the PQA "
                  "substructure service. Return citations and working links.")
        try:
            result = _responses_web_search(self, prompt, "pqa_molecular_analysis")
        except Exception as exc:
            return f"Literature analysis for {molecule} failed: {exc}"
        return ("[Note: answered by web literature search; the PQA substructure service is "
                "not available in this evaluation run.]\n\n" + result)

    # ---- 5. Non-interactivity. `ask_user` must never block, and must not be answerable
    # forever, or a subagent can spend the whole round budget asking questions.
    def ask_user(self, question):
        tracker = _tracker(self)
        if tracker is not None:
            tracker.note_ask_user()
            if tracker.ask_user_calls > MAX_ASK_USER:
                return ("No user input is available and this tool has already been called "
                        f"{MAX_ASK_USER} times. Stop asking and finish the task with the "
                        "information you have.")
        print(f"\033[1;33;40m[{self.agent_name}] ask_user (auto-answered): "
              f"{str(question)[:200]}\033[0m", flush=True)
        return ASK_USER_REPLY

    Basic_Agent.llm_infer = llm_infer
    Basic_Agent.brief_literature_review = brief_literature_review
    Basic_Agent.wrapped_literature_review = wrapped_literature_review
    Basic_Agent.literature_review = literature_review
    Basic_Agent.pqa_molecular_analysis = pqa_molecular_analysis
    Basic_Agent.ask_user = ask_user
    # Fleming.ask_user calls super(), so it picks up the patch. Subagents inherit directly.

    print(f"[harness] Fleming patched: model={cfg.model} "
          f"planner={cfg.resolved_planner_model()} "
          f"literature={cfg.resolved_literature_model()} "
          f"reasoning={cfg.reasoning}/{cfg.planner_reasoning} "
          f"literature_backend={cfg.literature_backend} "
          f"canned_lit_context={cfg.canned_lit_context}", flush=True)


# ======================================================================================
# The runner
# ======================================================================================

def _walk_agents(root, seen=None, depth=0):
    """
    Every Basic_Agent instance reachable from Fleming.

    Not just the five named subagents: `OptimizationUtils.__init__` builds its *own*
    ADMETUtils and InhibitionUtils when they are not passed in, which Fleming does not do,
    so there are nested agents whose API calls would otherwise go unpatched and uncounted.
    """
    from utils.utils import Basic_Agent
    if seen is None:
        seen = []
    if depth > 3 or any(existing is root for existing in seen):
        return seen
    seen.append(root)
    for name in dir(root):
        if name.startswith("__"):
            continue
        try:
            value = getattr(root, name)
        except Exception:
            continue
        if isinstance(value, Basic_Agent):
            _walk_agents(value, seen, depth + 1)
    return seen


class FlemingRunner:
    """
    One full Fleming instance, reusable across molecules.

    Constructing Fleming is expensive - it loads two ADMET-AI models (Fleming's own ADMET
    agent, plus the private copy inside the optimization agent) and takes about a minute -
    so a sweep builds one runner per worker thread and resets it between molecules rather
    than rebuilding.
    """

    def __init__(self, cfg: HarnessConfig = None):
        self.cfg = cfg or HarnessConfig()
        prepare_environment()
        install_patches(self.cfg)

        import main_fleming
        config_path = self.cfg.config_path or os.path.join(FLEMING_DIR, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Fleming config not found: {config_path}. Copy fleming/config.example.yaml "
                "to fleming/config.yaml, or pass --config.")
        started = time.time()
        self.fleming = main_fleming.Fleming(config_path)
        self.agents = _walk_agents(self.fleming)
        self.tracker = UsageTracker(self.cfg.prices)
        for agent in self.agents:
            agent._eval_tracker = self.tracker
        # System prompts as constructed, so `reset` can restore them exactly.
        self._base_system = {id(a): getattr(a, "system_prompt", None) for a in self.agents}
        api_key = os.environ.get("OPENAI_API_KEY") or self.fleming.config.get("openai_api_key")
        self.base_url = resolve_base_url(api_key, self.cfg.base_url, self.fleming.config)
        self.api_key = api_key
        print(f"[harness] Fleming built in {time.time() - started:.0f}s with "
              f"{len(self.agents)} agent objects; endpoint="
              f"{self.base_url or 'api.openai.com (direct)'}", flush=True)

    # ------------------------------------------------------------------ state

    def reset(self):
        """
        Clear every agent back to a fresh conversation.

        `Basic_Agent.reset()` sets `conversation = []`, dropping the system prompt, and a
        Fleming whose conversation starts empty plans with no system prompt at all. So
        restore the system message explicitly here instead of calling it.
        """
        for agent in self.agents:
            system_prompt = self._base_system.get(id(agent))
            agent.conversation = ([{"role": "system", "content": system_prompt}]
                                  if system_prompt else [])
            agent.latest_final_response = ""
            agent.tools_called_in_current_run = []
            agent._tool_outputs_for_summary = []
            agent._is_waiting_for_user_input = False
            agent._question_for_user_input = None
            agent._current_user_query = None
            agent.history = []
        fleming = self.fleming
        fleming._paused_agent_generator = None
        fleming._paused_agent_context = {}
        fleming._original_user_query = None
        fleming._active_plan_requirements = []
        fleming._completed_plan_requirements = set()
        fleming._active_plan_text = ""
        fleming._active_subagent_calls = set()
        self.tracker.reset()

    def quota(self):
        return read_quota(self.base_url, self.api_key)

    # ------------------------------------------------------------------ running

    def run(self, smiles, instruction: str, max_rounds: int = None,
            on_event: Callable[[dict], None] = None) -> dict:
        """
        Run Fleming once, start to finish, on one instruction.

        `smiles` is only used for the record - the molecule reaches Fleming inside the
        instruction text, the same way it would from a user typing in the interface.
        `on_event` is called with each transcript entry as it happens, which is how the CSV
        driver keeps a crash-safe partial record.

        Returns a dict with the final answer, the plan, the transcript, the tools and
        subagents that ran, a usage/cost snapshot, and a status.
        """
        from gradio import ChatMessage  # noqa: F401  (imported for isinstance clarity)

        self.reset()
        max_rounds = max_rounds or self.cfg.max_rounds
        transcript: List[dict] = []
        started = time.time()
        status = "complete"
        error = ""
        plan_text = ""
        molecules_shown: List[dict] = []

        def emit(kind: str, role: str, content):
            entry = {"t": round(time.time() - started, 2), "kind": kind,
                     "role": role, "content": content}
            transcript.append(entry)
            if on_event is not None:
                try:
                    on_event(entry)
                except Exception:
                    pass

        emit("instruction", "user", instruction)
        try:
            generator = self.fleming.run_gradio_chat(
                message=instruction,
                history=[],
                temperature=0.0,
                max_new_tokens=4096,
                max_token=None,
                call_agent=True,
                conversation_state=None,
                max_round=max_rounds,
                uploaded_csv_path=None,
                use_synspace=False,
            )
            for item in generator:
                # run_gradio_chat yields ('main'|'inner', ChatMessage), ('plan', text),
                # ('molecules', items), ('batch', df, summary) and ('download', path).
                panel = item[0] if isinstance(item, (tuple, list)) and item else "unknown"
                payload = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None
                if panel == "plan":
                    plan_text = str(payload or "")
                    emit("plan", "assistant", plan_text)
                elif panel == "molecules":
                    items = payload or []
                    molecules_shown.extend(
                        {"label": i.get("label"), "smiles": i.get("smiles")}
                        for i in items if isinstance(i, dict))
                    emit("molecules", "system",
                         ", ".join(str(i.get("label")) for i in items if isinstance(i, dict)))
                elif panel in ("main", "inner"):
                    emit(panel, str(getattr(payload, "role", "assistant")),
                         str(getattr(payload, "content", payload)))
                else:
                    emit(panel, "system", str(payload)[:4000])
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            emit("error", "system", traceback.format_exc()[-4000:])

        answer = str(getattr(self.fleming, "latest_final_response", "") or "")
        if status != "failed":
            if self.fleming._is_waiting_for_user_input:
                # Fleming stopped to ask a question. ask_user is auto-answered, so this can
                # only happen on a path that returns before the answer is consumed; the
                # molecule has no final answer and must be counted as an exclusion.
                status = "awaiting_user"
            elif not answer.strip():
                status = "no_answer"
            elif answer.strip().startswith("BLOCKED"):
                status = "blocked"

        # Fleming's own record only covers tools it dispatched through `run_function`. A
        # subagent call is eval()'d directly inside run_gradio_chat and never passes through
        # there, so `tools_called_in_current_run` is blank for exactly the calls that matter
        # most - a run that did all its work in two subagents looks, from here, like a run
        # that did nothing. Ask the subagents themselves instead: each keeps its own record
        # and clears it at the start of every run, so this is authoritative rather than a
        # guess parsed out of display strings.
        tools_used = list(getattr(self.fleming, "tools_called_in_current_run", []) or [])
        subagent_tools = {}
        for agent in self.agents:
            if agent is self.fleming:
                continue
            used = list(getattr(agent, "tools_called_in_current_run", []) or [])
            if used:
                subagent_tools[agent.agent_name] = used
                tools_used.extend(used)
        subagents = sorted(subagent_tools)
        usage = self.tracker.snapshot()
        return {
            "smiles": smiles,
            "instruction": instruction,
            "status": status,
            "error": error,
            "answer": answer,
            "plan": plan_text,
            "tools_used": tools_used,
            "subagents_used": subagents,
            "subagent_tools": subagent_tools,
            "molecules_shown": molecules_shown,
            "transcript": transcript,
            "usage": usage,
            "runtime_sec": round(time.time() - started, 1),
            "n_rounds_cap": max_rounds,
        }

    def ask_for_score_block(self, block_spec: str) -> Optional[str]:
        """
        One follow-up call asking Fleming, on its own conversation, for just the score
        block. Used when the final answer came back without it: the analysis is already
        paid for, and one cheap extra call is far better than discarding the molecule.
        Recorded in the usage as a separate call, and flagged in the CSV as
        `score_source=followup` so these rows are never mistaken for first-pass answers.
        """
        if not getattr(self.fleming, "conversation", None):
            return None
        conversation = list(self.fleming.conversation) + [{"role": "user", "content": block_spec}]
        try:
            return self.fleming.llm_infer(
                conversation,
                model=self.cfg.resolved_planner_model(),
                reasoning_effort=self.cfg.planner_reasoning,
            )
        except Exception as exc:
            print(f"Warning: score-block follow-up failed: {exc}", flush=True)
            return None


# ======================================================================================
# Environment
# ======================================================================================

_ENV_READY = {"done": False}


def prepare_environment():
    """
    Make `import main_fleming` work and make Fleming's relative paths resolve.

    Fleming's config points at `model_checkpoints/...` and `data/...` relative to the
    working directory, so the process has to run from the fleming/ directory. Resolve any
    user-supplied path to an absolute one *before* calling this.
    """
    with _PATCH_LOCK:
        if _ENV_READY["done"]:
            return
        if not os.path.isdir(FLEMING_DIR):
            raise FileNotFoundError(f"Fleming package directory not found: {FLEMING_DIR}")
        if FLEMING_DIR not in sys.path:
            sys.path.insert(0, FLEMING_DIR)
        os.chdir(FLEMING_DIR)
        _ENV_READY["done"] = True


if __name__ == "__main__":
    # Smoke test: build Fleming, report what got patched and what the gateway says.
    # No molecule is run and no model is called, so this costs nothing.
    runner = FlemingRunner(HarnessConfig())
    print(f"[harness] agents: {[a.agent_name for a in runner.agents]}")
    print(f"[harness] quota: {runner.quota()}")


# Older name, kept so existing imports keep working.
_resolve_base_url = resolve_base_url

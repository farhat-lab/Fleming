#!/usr/bin/env python
"""
Turning an arm's prose answer into numbers, for the TB-activity / Vero-cytotoxicity task.

Three jobs, kept apart on purpose:

  parse_answer()    pull the scores the arm stated out of its final answer.
  ranking_score()   decide the single number the 150 molecules are ranked by.
  target_label()    decide which molecules the ranking is supposed to retrieve.

Every arm - full Fleming, LLM + web search, and the chemprop baseline - writes the same
three score fields, so one parser and one metric script cover all of them.

`ranking_score` is the place to change the scoring spec. The default (`"promise"`) uses the
arm's own 0-10 PROMISE_SCORE, which is the thing the benchmark is testing: whether the
model can weigh activity against cytotoxicity itself. The other modes compute a ranking
from the two 0-1 axis scores instead, which is the fair comparison - they are what a fixed
formula would have said given the same two inputs.


"""

import re

# Models write typographic punctuation. A U+2010 HYPHEN in "literature-adjusted score"
# looks identical to "-" on screen and matches no ASCII pattern, which silently dropped two
# perfectly good scores on an earlier run. Normalize before any regex runs.
_PUNCTUATION_FIXES = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
    0x2212: "-", 0x00AD: "-", 0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2026: "...", 0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x2009: " ",
}
_PUNCTUATION_TABLE = str.maketrans(_PUNCTUATION_FIXES)


def normalize_for_parsing(text):
    return str(text or "").translate(_PUNCTUATION_TABLE)


def _label_pattern(*labels):
    """A label at the start of a line, then a colon/equals, then a number on that line."""
    alternatives = "|".join(labels)
    return re.compile(
        r"^[ \t>*#\-]*(?:\*\*)?(?:" + alternatives + r")(?:\*\*)?[ \t]*[:=][ \t]*(?:\*\*)?"
        r"([0-9]*\.?[0-9]+)",
        re.IGNORECASE | re.MULTILINE,
    )


# The activity axis. Label varies by assay: the replicating-Mtb prompt says
# TB_INHIBITION_SCORE, the non-replicating one NRP_TB_ACTIVITY_SCORE. All accepted, so one
# parser covers both datasets and older result files keep working.
TB_RE = _label_pattern(r"NRP_TB_ACTIVITY_SCORE", r"NRP TB ACTIVITY SCORE", r"NRP_TB_SCORE",
                       r"TB_ACTIVITY_SCORE", r"TB_INHIBITION_SCORE", r"TB INHIBITION SCORE",
                       r"INHIBITION_SCORE", r"MTB_INHIBITION_SCORE")
# The cytotoxicity axis. Vero for the first dataset, HepG2 for the second.
CYTOTOX_RE = _label_pattern(r"HEPG2_CYTOTOX_SCORE", r"HEPG2 CYTOTOX SCORE",
                            r"HEPG2_CYTOTOXICITY_SCORE",
                            r"VERO_CYTOTOX_SCORE", r"VERO CYTOTOX SCORE",
                            r"VERO_CYTOTOXICITY_SCORE", r"CYTOTOX_SCORE")
VERO_RE = CYTOTOX_RE  # older name, kept so nothing importing it breaks
# The 0-10 selectivity axis. The current prompt labels it SELECTIVITY_SCORE; earlier runs
# used PROMISE_SCORE. Both are accepted so result files from either generation parse, and
# the parsed value keeps the key `promise_score` so existing result CSVs, evaluate.py and
# compare_arms.py continue to line up.
PROMISE_RE = _label_pattern(r"SELECTIVITY_SCORE", r"SELECTIVITY SCORE",
                            r"PROMISE_SCORE", r"PROMISE SCORE", r"OVERALL_SCORE")
# The second endpoint, added later: 0-10, how plausible the molecule is as an antibiotic
# worth advancing (selective, drug-like, synthesizable). Absent from earlier runs, so it
# parses as None there rather than failing.
PLAUSIBILITY_RE = _label_pattern(r"ANTIBIOTIC_PLAUSIBILITY_SCORE", r"ANTIBIOTIC PLAUSIBILITY SCORE",
                                 r"PLAUSIBILITY_SCORE", r"PLAUSIBILITY SCORE")
RATIONALE_RE = re.compile(
    r"^[ \t>*#\-]*(?:\*\*)?(?:RATIONALE|OVERALL[_ ]RATIONALE)(?:\*\*)?[ \t]*[:=][ \t]*(.+)$",
    re.IGNORECASE | re.MULTILINE)

# The raw model numbers, as the two Fleming tools phrase them. Recorded on every row - they
# are how the agent's stated score can be compared against what the model it called actually
# said - and also the last-resort fallback when no block was produced. Only the agent arm
# has these; the LLM-only arm leaves them blank, which is the point of that arm.
#
# Both tools use the same shape, with the scale spelled out on every number:
#   "Inhibition model score (0-1, higher = more predicted active): 0.72;
#    Dirichlet uncertainty (0-1, higher = less confident): 0.18."
#   "hepatocyte (HepG2) cytotoxicity score (0-1, higher = more predicted cytotoxic): 0.131;
#    Dirichlet uncertainty (0-1, higher = less confident): 0.245."
# Two consequences for parsing:
#   * the parenthetical itself contains digits, so a lazy "[^0-9]{0,40}" gap reads the "0"
#     out of "(0-1" and reports a score of 0.0. The parenthetical is skipped explicitly.
#   * "Dirichlet uncertainty" appears up to four times in one answer (inhibition plus three
#     cell lines), so each uncertainty is anchored to the score it belongs to rather than
#     taken as the last match in the text.
# The OLD wording of the cytotoxicity tool ("...toxicity prediction is [0.131] with an
# uncertainty of [0.245]") is kept as a fallback so results from before that fix parse.
_PAREN = r"(?:\s*\([^)]*\))?"
_NUM = r"\[?\s*([0-9]*\.?[0-9]+)"

RAW_INHIBITION_RES = [
    re.compile(r"inhibition model score" + _PAREN + r"\s*[:=]\s*" + _NUM, re.IGNORECASE),
]
RAW_INH_UNCERTAINTY_RES = [
    re.compile(r"inhibition model score" + _PAREN + r"\s*[:=]\s*\[?\s*[0-9]*\.?[0-9]+\s*;\s*"
               r"dirichlet uncertainty" + _PAREN + r"\s*[:=]\s*" + _NUM, re.IGNORECASE),
    re.compile(r"dirichlet uncertainty" + _PAREN + r"\s*[:=]\s*" + _NUM, re.IGNORECASE),
]
# The three human proxy lines. There is no Vero model, so these are what the agent actually
# had; each is recorded separately, because agreement across all three is the only part of
# this evidence that transfers to a different tissue and species.
_SCALE = r"[^:\n]{0,70}\(\s*0\s*-\s*1[^)\n]*\)"


def _cytotox_res(line_name, *aliases):
    names = "|".join((line_name,) + aliases)
    return [
        re.compile(r"(?:" + names + r")" + _SCALE + r"\s*[:=]\s*" + _NUM, re.IGNORECASE),
    ]


RAW_HEPG2_RES = _cytotox_res(r"hepatocyte", r"hepg2") + [
    re.compile(r"hepatocyte toxicity prediction is\s*\[*\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)]
RAW_HSKMC_RES = _cytotox_res(r"skeletal muscle", r"hskmc") + [
    re.compile(r"skeletal muscle toxicity prediction\s*is\s*\[*\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)]
RAW_IMR90_RES = _cytotox_res(r"lung", r"imr90") + [
    re.compile(r"lung \(imr90\)\s*toxicity prediction is\s*\[*\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)]
RAW_HEPG2_UNC_RES = [
    re.compile(r"(?:hepatocyte|hepg2)" + _SCALE + r"\s*[:=]\s*\[?\s*[0-9]*\.?[0-9]+\s*;\s*"
               r"dirichlet uncertainty" + _PAREN + r"\s*[:=]\s*" + _NUM, re.IGNORECASE),
    re.compile(r"hepatocyte toxicity prediction is.{0,60}?uncertainty of\s*\[*\s*"
               r"([0-9]*\.?[0-9]+)", re.IGNORECASE),
]


def _last(pattern, text):
    """The last match, or None. See the module docstring for why last, not first."""
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _last_of(patterns, text):
    """The last match of the first pattern that matches at all, in priority order."""
    for pattern in patterns:
        value = _last(pattern, text)
        if value is not None:
            return value
    return None


def _in_range(value, low=0.0, high=1.0):
    """
    Keep only scores that are actually on their stated scale.

    A model that writes "85%" or "8.5/10" where a probability was asked for is not giving a
    probability, and silently treating 85 as a score would wreck every metric downstream.
    Out-of-range values are dropped and surface as a missing score, which is visible, rather
    than a wrong one.
    """
    if value is None:
        return None
    return value if low <= value <= high else None


def parse_answer(answer_text, followup_text=None):
    """
    Read the scores out of an arm's final answer.

    `followup_text` is the reply to the score-block follow-up call, used when the first
    answer had no block. Returns a dict; `score_source` says where the numbers came from:

        block      the required block, in the first answer            (best)
        followup   the required block, from the follow-up call         (fine)
        prose      raw model numbers scraped from the discussion       (weak - excluded from
                   the metrics by default)
        none       nothing parseable
    """
    primary = normalize_for_parsing(answer_text)
    result = {
        "tb_score": None,
        "vero_score": None,
        "promise_score": None,
        "plausibility_score": None,
        "rationale": "",
        "raw_inhibition": _in_range(_last_of(RAW_INHIBITION_RES, primary)),
        "inhibition_uncertainty": _in_range(_last_of(RAW_INH_UNCERTAINTY_RES, primary)),
        "raw_hepg2": _in_range(_last_of(RAW_HEPG2_RES, primary)),
        "raw_hskmc": _in_range(_last_of(RAW_HSKMC_RES, primary)),
        "raw_imr90": _in_range(_last_of(RAW_IMR90_RES, primary)),
        "hepg2_uncertainty": _in_range(_last_of(RAW_HEPG2_UNC_RES, primary)),
        "score_source": "none",
    }

    for source, text in (("block", primary),
                         ("followup", normalize_for_parsing(followup_text) if followup_text else "")):
        if not text:
            continue
        tb = _in_range(_last(TB_RE, text))
        vero = _in_range(_last(CYTOTOX_RE, text))
        promise = _in_range(_last(PROMISE_RE, text), 0.0, 10.0)
        plausibility = _in_range(_last(PLAUSIBILITY_RE, text), 0.0, 10.0)
        if any(v is not None for v in (tb, vero, promise, plausibility)):
            rationale = RATIONALE_RE.findall(text)
            result.update({
                "tb_score": tb,
                "vero_score": vero,
                "promise_score": promise,
                "plausibility_score": plausibility,
                "rationale": rationale[-1].strip() if rationale else "",
                "score_source": source,
            })
            return result

    # No block anywhere. Fall back to the raw model numbers, clearly flagged.
    if result["raw_inhibition"] is not None or result["raw_hepg2"] is not None:
        result["tb_score"] = result["raw_inhibition"]
        result["vero_score"] = result["raw_hepg2"]
        result["score_source"] = "prose"
    return result


# ======================================================================================
# The ranking score
# ======================================================================================

RANKING_MODES = ("promise", "product", "min", "weighted", "mean")


def ranking_score(tb, vero, promise=None, mode="promise", weight=0.5):
    """
    The single 0-1 number the molecules are ranked by, higher = more promising.

    `tb` is higher-is-better; `vero` is higher-is-WORSE, so it is flipped to a safety term
    (1 - vero) before being combined. Getting that direction wrong inverts the whole
    benchmark, so it happens in exactly one place.

    Modes:
        promise   the arm's own PROMISE_SCORE, rescaled from 0-10 to 0-1. The default: this
                  is what the benchmark is testing. Rescaling is monotone, so it changes no
                  ranking, only the axis the plots share. Falls back to `product` if the arm
                  gave no promise score.
        product   tb * (1 - vero). Multiplicative, so either axis failing kills the
                  candidate - the usual shape for a hard counter-screen constraint.
        min       min(tb, 1 - vero). Harsher: the score is whichever axis is weaker.
        weighted  weight * tb + (1 - weight) * (1 - vero). Additive, so a very potent
                  compound can buy its way past cytotoxicity. Only use it if that is
                  genuinely the trade-off you want.
        mean      weighted with weight=0.5, named for readability.

    Returns (score, mode_actually_used) so a fallback is never invisible.
    """
    if mode not in RANKING_MODES:
        raise ValueError(f"Unknown ranking mode {mode!r}; expected one of {RANKING_MODES}")

    if mode == "promise":
        if promise is not None:
            return round(promise / 10.0, 6), "promise"
        mode = "product"  # fall through, and say so in the returned mode

    if tb is None or vero is None:
        return None, mode
    safety = 1.0 - vero
    if mode == "product":
        return round(tb * safety, 6), mode
    if mode == "min":
        return round(min(tb, safety), 6), mode
    if mode == "mean":
        weight = 0.5
    return round(weight * tb + (1.0 - weight) * safety, 6), mode


def tiebreak_score(tb, vero):
    """
    A continuous second key for ranking.

    PROMISE_SCORE is 0-10 to one decimal, so across 150 molecules many will share a value -
    and "the top 20" is then not well defined. Ties are broken by tb * (1 - vero), the arm's
    own finer-grained judgement of the same question. `evaluate.py` also reports the
    tie-free expectation under random tie-breaking, so no conclusion rests on this choice.
    """
    if tb is None or vero is None:
        return None
    return round(tb * (1.0 - vero), 6)


def target_label(y_tb, y_vero):
    """
    The ground truth this benchmark retrieves: active against Mtb AND not Vero-cytotoxic.

    `y_tb`: 1 = TB active. `y_vero`: 1 = cytotoxic to Vero, 0 = not.

    As of the confirmed set (data/mtb_150_confirmed.csv), `y_tb` is the source assay's
    CONFIRMATORY outcome, not single-point primary-screen inhibition. The primary screen
    retracted 11 of the original 50 targets on re-test, and a fifth of the target set being
    compounds the assay itself called Inactive depressed every arm's measured enrichment.

    A missing `y_vero` is NOT missing data here. In the 150-molecule set the TB primary
    non-hits were never counter-screened, because there was no reason to counter-screen a
    compound that did not hit - so `y_vero` is blank for all 50 of them, and every one is a
    definite negative on this endpoint regardless of what a Vero assay would have shown.
    Treating those as unknown would silently drop a third of the data, and the third that
    the ranking most needs to push down.

    Returns None only when TB activity itself is unknown.
    """
    if y_tb is None or (isinstance(y_tb, float) and y_tb != y_tb):
        return None
    if int(y_tb) == 0:
        return 0
    if y_vero is None or (isinstance(y_vero, float) and y_vero != y_vero):
        # A TB hit with no counter-screen result. Does not occur in mtb_150, but if it
        # appears in a future set, it cannot be scored either way.
        return None
    return int(int(y_vero) == 0)


# The three groups in mtb_150, named the way the source file names them. Which group a
# top-k false positive comes from is the substance of the "can Fleming tell these apart"
# claim, so the group is carried through the whole pipeline rather than collapsed to 0/1.
GROUPS = ("TB confirmed / Vero nontoxic", "TB confirmed / Vero toxic", "TB primary nonhit")
# The non-replicating / HepG2 dataset uses its own group names for the same three roles:
# target, active-but-cytotoxic, inactive.
NRP_GROUPS = ("Mtb hit / HepG2 nontoxic", "Mtb hit / HepG2 toxic", "Mtb nonhit")
# The names used before the labels moved to the confirmatory outcome. Kept so result files
# from earlier runs still group correctly instead of silently reporting empty categories.
LEGACY_GROUPS = ("TB hit / Vero nontoxic", "TB hit / Vero toxic", "TB primary nonhit")


def group_of(y_tb, y_vero):
    """The selection group implied by a pair of labels, for result files lacking the column."""
    label = target_label(y_tb, y_vero)
    if label is None:
        return ""
    if label == 1:
        return GROUPS[0]
    return GROUPS[1] if int(y_tb) == 1 else GROUPS[2]


def canonical_group(name):
    """Map a legacy group name onto its current equivalent, so mixed files still line up."""
    name = str(name or "").strip()
    for legacy, current in zip(LEGACY_GROUPS, GROUPS):
        if name == legacy:
            return current
    return name


def group_role(name):
    """
    Which of the three roles a group name plays, across datasets.

    Returns 'target', 'active_toxic', 'inactive' or None. The two datasets name their
    groups differently (Vero vs HepG2, "TB confirmed" vs "Mtb hit"), and analysis code
    should key on the role rather than on either dataset's wording.
    """
    name = canonical_group(name)
    for role, vero, nrp in zip(("target", "active_toxic", "inactive"), GROUPS, NRP_GROUPS):
        if name in (vero, nrp):
            return role
    return None


if __name__ == "__main__":
    # Parser self-check against the tools' REAL output strings, not paraphrases of them.
    # Both are copied verbatim from a live call, en-dashes and all - the en-dash in "(0-1"
    # is a U+2013 that normalize_for_parsing has to fold before any pattern sees it.
    INHIBITION_TOOL = (
        "Molecule 1 (CC(=O)Nc1ccc(O)cc1) - Inhibition model score (0–1, higher = more "
        "predicted active): 0.72; Dirichlet uncertainty (0–1, higher = less confident): 0.18. "
    )
    CYTOTOX_TOOL = (
        "For molecule 1 (CC(=O)Nc1ccc(O)cc1): hepatocyte (HepG2) cytotoxicity score "
        "(0-1, higher = more predicted cytotoxic): 0.131; Dirichlet uncertainty "
        "(0-1, higher = less confident): 0.245. Skeletal muscle (HSKMC) cytotoxicity score "
        "(0-1, higher = more predicted cytotoxic): 0.052; Dirichlet uncertainty "
        "(0-1, higher = less confident): 0.104. Lung (IMR90) cytotoxicity score "
        "(0-1, higher = more predicted cytotoxic): 0.472; Dirichlet uncertainty "
        "(0-1, higher = less confident): 0.426. "
    )
    CYTOTOX_TOOL_OLD = (
        "For molecule 1 (CC(=O)Nc1ccc(O)cc1), the predicted hepatocyte toxicity prediction is "
        "[0.131] with an uncertainty of [0.245]. The predicted skeletal muscle toxicity "
        "prediction is [0.052] with an uncertainty of [0.104]. The predicted lung (imr90) "
        "toxicity prediction is [0.472] with an uncertainty of [0.426]. "
    )
    BLOCK = ("TB_INHIBITION_SCORE: 0.68\nVERO_CYTOTOX_SCORE: 0.35\nPROMISE_SCORE: 6.5\n"
             "RATIONALE: Active-looking scaffold, but a nitro group raises a real "
             "counter-screen risk.\n")
    PROSE = ("ANSWER: The scaffold looks plausible.\n\n" + INHIBITION_TOOL + "\n\n" +
             CYTOTOX_TOOL + "\n\nSeveral features inform the whole-cell prediction:\n\n"
             "1. Mycobacterial cell-wall permeability\n2. Efflux liability\n\n")

    parsed = parse_answer(PROSE + BLOCK)
    assert parsed["score_source"] == "block", parsed
    assert (parsed["tb_score"], parsed["vero_score"], parsed["promise_score"]) == (0.68, 0.35, 6.5), parsed
    assert parsed["raw_inhibition"] == 0.72, parsed
    assert (parsed["raw_hepg2"], parsed["raw_hskmc"], parsed["raw_imr90"]) == (0.131, 0.052, 0.472), parsed
    # Each uncertainty must come from its own score, not from whichever came last: with four
    # "Dirichlet uncertainty" phrases in one answer, an unanchored pattern reports the lung
    # value (0.426) for everything.
    assert parsed["inhibition_uncertainty"] == 0.18, parsed
    assert parsed["hepg2_uncertainty"] == 0.245, parsed

    # The pre-fix cytotoxicity wording still parses.
    old = parse_answer("ANSWER: x\n\n" + INHIBITION_TOOL + "\n\n" + CYTOTOX_TOOL_OLD + BLOCK)
    assert old["raw_hepg2"] == 0.131 and old["hepg2_uncertainty"] == 0.245, old
    assert old["inhibition_uncertainty"] == 0.18, old

    # The current prompt's label set, with the second endpoint.
    current = parse_answer("TB_INHIBITION_SCORE: 0.62\nVERO_CYTOTOX_SCORE: 0.31\n"
                           "SELECTIVITY_SCORE: 7.2\nANTIBIOTIC_PLAUSIBILITY_SCORE: 5.5\n"
                           "RATIONALE: x")
    assert current["promise_score"] == 7.2, current
    assert current["plausibility_score"] == 5.5, current
    assert current["score_source"] == "block", current
    # Runs from before the rename must still parse, with no plausibility score.
    legacy = parse_answer("TB_INHIBITION_SCORE: 0.62\nVERO_CYTOTOX_SCORE: 0.31\nPROMISE_SCORE: 7.2")
    assert legacy["promise_score"] == 7.2 and legacy["plausibility_score"] is None, legacy
    # ANTIBIOTIC_PLAUSIBILITY_SCORE must not be mistaken for the selectivity score.
    only_plaus = parse_answer("ANTIBIOTIC_PLAUSIBILITY_SCORE: 5.5")
    assert only_plaus["promise_score"] is None and only_plaus["plausibility_score"] == 5.5, only_plaus

    # The non-replicating / HepG2 label set, and the collision it creates: the block's own
    # HEPG2_CYTOTOX_SCORE must not be mistaken for the HepG2 MODEL output quoted above it.
    nrp = parse_answer(CYTOTOX_TOOL + "\n\nNRP_TB_ACTIVITY_SCORE: 0.62\n"
                       "HEPG2_CYTOTOX_SCORE: 0.31\nSELECTIVITY_SCORE: 7.2\n"
                       "ANTIBIOTIC_PLAUSIBILITY_SCORE: 5.5\nRATIONALE: x")
    assert nrp["tb_score"] == 0.62, nrp
    assert nrp["vero_score"] == 0.31, nrp
    assert nrp["raw_hepg2"] == 0.131, f"block label leaked into raw_hepg2: {nrp['raw_hepg2']}"
    assert nrp["hepg2_uncertainty"] == 0.245, nrp
    assert nrp["promise_score"] == 7.2 and nrp["plausibility_score"] == 5.5, nrp

    # A promise score on the wrong scale is dropped, not believed.
    assert parse_answer("PROMISE_SCORE: 65")["promise_score"] is None
    assert parse_answer("PROMISE_SCORE: 9.5")["promise_score"] == 9.5
    assert parse_answer("TB_INHIBITION_SCORE: 85")["tb_score"] is None

    # No block: fall back to prose, say so, and do NOT read the markdown "1." as a score.
    parsed = parse_answer(PROSE)
    assert parsed["score_source"] == "prose", parsed
    assert parsed["tb_score"] == 0.72 and parsed["vero_score"] == 0.131, parsed

    # Follow-up path.
    parsed = parse_answer(PROSE, "TB_INHIBITION_SCORE: 0.5\nVERO_CYTOTOX_SCORE: 0.1\nPROMISE_SCORE: 7.0")
    assert parsed["score_source"] == "followup" and parsed["promise_score"] == 7.0, parsed

    # The LLM-only arm has no model numbers at all, and that must parse cleanly.
    llm_only = parse_answer("Based on the literature this looks risky.\n\n" + BLOCK)
    assert llm_only["score_source"] == "block" and llm_only["raw_inhibition"] is None, llm_only

    assert ranking_score(0.8, 0.2, 6.5) == (0.65, "promise")
    assert ranking_score(0.8, 0.2, None) == (0.64, "product")
    assert ranking_score(0.8, 0.2, None, mode="min") == (0.8, "min")     # min(0.8, 1-0.2)
    assert ranking_score(0.8, 0.9, None, mode="min") == (0.1, "min")     # the weaker axis wins
    assert ranking_score(0.8, 0.5, None, mode="mean") == (0.65, "mean")
    assert tiebreak_score(0.8, 0.2) == 0.64

    # The label rules, including the 50 non-hits that were never counter-screened.
    assert target_label(1, 0) == 1
    assert target_label(1, 1) == 0
    assert target_label(0, float("nan")) == 0, "a TB non-hit is a negative, not unknown"
    assert target_label(0, None) == 0
    assert target_label(None, 0) is None
    assert target_label(1, None) is None
    assert group_of(1, 0) == GROUPS[0] and group_of(1, 1) == GROUPS[1]
    assert group_of(0, float("nan")) == GROUPS[2]
    print("scoring.py self-check passed")

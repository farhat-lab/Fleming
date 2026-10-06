"""Median Max Tanimoto Similarity (MMTS).

For every prospective compound, compute its maximum Tanimoto similarity against a
single pool of reference compounds (train + val + test), then report the median of
those per-compound maxima.

Usage:
    python mmts_calculation.py
    python mmts_calculation.py --n-jobs 8 --radius 2 --n-bits 2048

    # restrict to experimentally confirmed hits only
    python mmts_calculation.py --hits-only --hit-column is_hit
    python mmts_calculation.py --hits-only --hit-file hits.csv --hit-key SMILES
"""

import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# Paths relative to this script (evaluation/similarity_analyses/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROSPECTIVE_PATH = SCRIPT_DIR / "combined_prospective_smiles_for_mmts.csv"
REFERENCE_PATHS = [
    SCRIPT_DIR / "../../fleming/data/train_data.csv",
    SCRIPT_DIR / "../../fleming/data/val_data.csv",
    SCRIPT_DIR / "../../fleming/data/test_data.csv",
]
OUTPUT_PATH = SCRIPT_DIR / "prospective_max_tanimoto_vs_train_val_test.csv"

SMILES_COL = "SMILES"

# Globals populated per worker process (avoids pickling fingerprints per task).
_REF_FPS = None
_FPGEN = None


def _make_generator(radius, n_bits):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def featurize(smiles_list, radius, n_bits, desc):
    """Return (valid_smiles, fingerprints) dropping anything RDKit can't parse."""
    fpgen = _make_generator(radius, n_bits)
    kept, fps = [], []
    for smi in tqdm(smiles_list, desc=desc):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        kept.append(smi)
        fps.append(fpgen.GetFingerprint(mol))
    return kept, fps


def _init_worker(ref_fps, radius, n_bits):
    global _REF_FPS, _FPGEN
    _REF_FPS = ref_fps
    _FPGEN = _make_generator(radius, n_bits)


def _max_sim_for_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.nan
    fp = _FPGEN.GetFingerprint(mol)
    return max(BulkTanimotoSimilarity(fp, _REF_FPS))


# Values treated as "this row is a hit" for non-numeric hit columns.
TRUTHY_HIT_VALUES = {"hit", "hits", "active", "actives", "yes", "y", "true", "t", "1"}


def canonicalize(smiles):
    """Canonical SMILES, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def is_hit_series(series, threshold):
    """Boolean mask of which rows of `series` count as hits.

    Numeric columns are thresholded (>= threshold); everything else is matched
    case-insensitively against TRUTHY_HIT_VALUES.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric >= threshold
    return series.astype(str).str.strip().str.lower().isin(TRUTHY_HIT_VALUES)


def select_hits(prospective, hit_file, hit_column, hit_key, threshold):
    """Return the subset of `prospective` that counts as hits.

    Two sources are supported:
      * hit_column present in the prospective table -> filter it directly
      * hit_file -> a separate table of hits, joined on hit_key
        (optionally filtered by hit_column within that file first)
    """
    if hit_file is None and hit_column is None:
        raise SystemExit(
            "--hits-only needs a hit definition. Pass --hit-column NAME if the "
            "prospective CSV carries a hit label, or --hit-file PATH to join "
            "against a separate table of hits.\n"
            f"Columns available in {PROSPECTIVE_PATH.name}: {list(prospective.columns)}"
        )

    if hit_file is None:
        if hit_column not in prospective.columns:
            raise SystemExit(
                f"--hit-column '{hit_column}' not found in {PROSPECTIVE_PATH.name} "
                f"(columns: {list(prospective.columns)})"
            )
        mask = is_hit_series(prospective[hit_column], threshold)
        return prospective[mask].reset_index(drop=True)

    hits = pd.read_csv(hit_file)
    if hit_column is not None:
        if hit_column not in hits.columns:
            raise SystemExit(
                f"--hit-column '{hit_column}' not found in {hit_file} "
                f"(columns: {list(hits.columns)})"
            )
        hits = hits[is_hit_series(hits[hit_column], threshold)]
    if hit_key not in hits.columns:
        raise SystemExit(
            f"--hit-key '{hit_key}' not found in {hit_file} (columns: {list(hits.columns)})"
        )
    if hit_key not in prospective.columns:
        raise SystemExit(
            f"--hit-key '{hit_key}' not found in {PROSPECTIVE_PATH.name} "
            f"(columns: {list(prospective.columns)})"
        )

    if hit_key == SMILES_COL:
        # Join on canonical structure so formatting differences do not drop hits.
        hit_keys = {c for c in (canonicalize(x) for x in hits[hit_key]) if c is not None}
        mask = prospective[hit_key].map(lambda x: canonicalize(x) in hit_keys)
    else:
        hit_keys = set(hits[hit_key].astype(str))
        mask = prospective[hit_key].astype(str).isin(hit_keys)

    matched = int(mask.sum())
    print(f"Hit table {Path(hit_file).name}: {len(hits)} hits, {matched} matched into the prospective set")
    if matched == 0:
        raise SystemExit(
            f"No prospective compounds matched the hits in {hit_file} on '{hit_key}'. "
            "Check that the key column refers to the same identifier space."
        )
    return prospective[mask].reset_index(drop=True)


def load_smiles(path, column=SMILES_COL):
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"{path} has no '{column}' column (found: {list(df.columns)})")
    return df[column].dropna().astype(str).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius (default: 2)")
    parser.add_argument("--n-bits", type=int, default=2048, help="Fingerprint length (default: 2048)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker processes (default: 1)")
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Drop duplicate prospective SMILES before computing the median",
    )
    parser.add_argument(
        "--hits-only",
        action="store_true",
        help="Compute MMTS over hits only, instead of the full prospective set",
    )
    parser.add_argument(
        "--hit-column",
        default=None,
        help="Column holding the hit label. Looked up in the prospective CSV, or "
             "within --hit-file when that is given. Numeric columns use --hit-threshold; "
             "text columns match hit/active/yes/true.",
    )
    parser.add_argument(
        "--hit-file",
        type=Path,
        default=None,
        help="Separate CSV listing the hits, joined to the prospective set on --hit-key",
    )
    parser.add_argument(
        "--hit-key",
        default=SMILES_COL,
        help=f"Column used to join --hit-file to the prospective set (default: {SMILES_COL}). "
             "SMILES joins are canonicalized before matching.",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=0.5,
        help="For a numeric hit column, rows with value >= this count as hits (default: 0.5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N prospective compounds (for quick timing checks)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Per-compound CSV output path")
    args = parser.parse_args()

    # Reference pool: train + val + test together.
    ref_smiles = []
    for path in REFERENCE_PATHS:
        smiles = load_smiles(path)
        print(f"{path.name}: {len(smiles)} SMILES")
        ref_smiles.extend(smiles)
    ref_smiles = list(dict.fromkeys(ref_smiles))  # unique, order-preserving
    print(f"Reference pool (unique): {len(ref_smiles)}")

    prospective = pd.read_csv(PROSPECTIVE_PATH)
    if args.dedup:
        prospective = prospective.drop_duplicates(subset=SMILES_COL)
    prospective = prospective.dropna(subset=[SMILES_COL]).reset_index(drop=True)
    print(f"Prospective compounds: {len(prospective)}")

    population = "all prospective compounds"
    if args.hits_only:
        prospective = select_hits(
            prospective, args.hit_file, args.hit_column, args.hit_key, args.hit_threshold
        )
        population = "hits only"
        print(f"Hits retained: {len(prospective)}")
        if len(prospective) == 0:
            raise SystemExit("No hits selected; nothing to score.")

    if args.limit:
        prospective = prospective.head(args.limit)

    _, ref_fps = featurize(ref_smiles, args.radius, args.n_bits, "Featurizing reference pool")
    print(f"Reference fingerprints: {len(ref_fps)}")

    query_smiles = prospective[SMILES_COL].astype(str).tolist()
    if args.n_jobs > 1:
        with Pool(
            processes=args.n_jobs,
            initializer=_init_worker,
            initargs=(ref_fps, args.radius, args.n_bits),
        ) as pool:
            max_sims = list(
                tqdm(
                    pool.imap(_max_sim_for_smiles, query_smiles, chunksize=32),
                    total=len(query_smiles),
                    desc="Max Tanimoto",
                )
            )
    else:
        _init_worker(ref_fps, args.radius, args.n_bits)
        max_sims = [_max_sim_for_smiles(s) for s in tqdm(query_smiles, desc="Max Tanimoto")]

    prospective["max_tanimoto_vs_train_val_test"] = max_sims

    valid = prospective["max_tanimoto_vs_train_val_test"].dropna()
    n_invalid = len(prospective) - len(valid)

    print(f"\n=== Max Tanimoto similarity ({population} vs. train+val+test pool) ===")
    print(f"Compounds scored : {len(valid)}" + (f"  ({n_invalid} unparseable, skipped)" if n_invalid else ""))
    print(f"MEDIAN max Tanimoto similarity (MMTS): {valid.median():.4f}")
    print(f"Mean   : {valid.mean():.4f}")
    print(f"Std    : {valid.std():.4f}")
    print(f"Min    : {valid.min():.4f}")
    print(f"25th   : {valid.quantile(0.25):.4f}")
    print(f"75th   : {valid.quantile(0.75):.4f}")
    print(f"Max    : {valid.max():.4f}")
    print(f"Fraction with max sim < 0.4 : {(valid < 0.4).mean():.4f}")

    output = args.output
    if args.hits_only and output == OUTPUT_PATH:
        output = output.with_name(output.stem + "_hits_only" + output.suffix)
    prospective.to_csv(output, index=False)
    print(f"\nPer-compound results written to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Build data/molecules_350.csv from PubChem.



    python build_dataset.py                     # fetch, build, verify
    python build_dataset.py --out ../../data/molecules_350.csv

Needs network access and rdkit. Responses are cached in cache/ so a second run is offline.

Inputs, both alongside this file:
    selection.csv    SID, CID and design group for the 350 compounds
    arm_scores.csv   the four arm scores per SID

Assays:
    AID 488890   M. tuberculosis H37Rv, non-replicating persistence (activity)
    AID 489018   HepG2 cytotoxicity counter-screen (safety)
"""

import argparse
import io
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "cache")
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
AID_TB, AID_HEPG2 = 488890, 489018
CHUNK = 100                      # SIDs/CIDs per request; keeps the URL comfortably short


def _get(url, tries=4):
    """GET with linear backoff. PubChem returns 503 under load and 500 when a service
    is briefly down; both clear on retry, so a transient failure should not kill a build."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read().decode()
        except Exception as exc:                                   # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"   retry {attempt + 1}/{tries - 1} after {exc}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        print(f"[cache] {name}")
        return pd.read_csv(path, low_memory=False)
    frame = build()
    frame.to_csv(path, index=False)
    print(f"[fetch] {name}  ({len(frame)} rows)")
    return frame


def fetch_assay(aid, sids, name):
    """Rows of one assay for the given SIDs only - the full AID 488890 table is 324,545
    rows, and we need 350 of them."""
    def build():
        parts = []
        for i in range(0, len(sids), CHUNK):
            block = ",".join(str(s) for s in sids[i:i + CHUNK])
            text = _get(f"{PUG}/assay/aid/{aid}/CSV?sid={block}")
            frame = pd.read_csv(io.StringIO(text), low_memory=False)
            # PubChem prefixes two metadata rows (RESULT_TYPE, RESULT_DESCR) that carry
            # no SID; they would otherwise become NaN-keyed rows in the merge.
            parts.append(frame[pd.to_numeric(frame.PUBCHEM_SID, errors="coerce").notna()])
        return pd.concat(parts, ignore_index=True)
    return _cached(name, build)


def fetch_smiles(cids):
    """Structures by CID. Both assay exports carry an empty SMILES column, so the
    structures have to come from the compound record."""
    def build():
        parts = []
        for i in range(0, len(cids), CHUNK):
            block = ",".join(str(c) for c in cids[i:i + CHUNK])
            text = _get(f"{PUG}/compound/cid/{block}/property/SMILES/CSV")
            parts.append(pd.read_csv(io.StringIO(text)))
        return pd.concat(parts, ignore_index=True)
    return _cached("smiles.csv", build)


def rdkit_properties(smiles):
    """QED, synthetic accessibility and PAINS+BRENK alert counts."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, RDConfig
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDLogger.DisableLog("rdApp.*")
    # SAscore ships with RDKit as a contrib script rather than an importable module.
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    params = FilterCatalogParams()
    for catalog in (FilterCatalogParams.FilterCatalogs.PAINS,
                    FilterCatalogParams.FilterCatalogs.BRENK):
        params.AddCatalog(catalog)
    alerts = FilterCatalog(params)

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"RDKit could not parse {smi!r}")
        rows.append(dict(QED=QED.qed(mol),
                         SAscore=sascorer.calculateScore(mol),
                         n_alerts=len(alerts.GetMatches(mol))))
    return pd.DataFrame(rows, index=pd.Index(smiles, name="SMILES")).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "..", "data",
                                                  "molecules_350.csv"))
    ap.add_argument("--verify", default=None,
                    help="compare the result against an existing dataset")
    args = ap.parse_args()

    sel = pd.read_csv(os.path.join(_HERE, "selection.csv"))
    arms = pd.read_csv(os.path.join(_HERE, "arm_scores.csv"))
    sids, cids = sel.SID.tolist(), sel.CID.tolist()
    print(f"selection: {len(sel)} compounds\n")

    tb = fetch_assay(AID_TB, sids, "aid_488890.csv")
    hep = fetch_assay(AID_HEPG2, sids, "aid_489018.csv")
    smi = fetch_smiles(cids)

    num = lambda frame, col: pd.to_numeric(frame[col], errors="coerce")
    tb = pd.DataFrame({
        "SID": num(tb, "PUBCHEM_SID").astype("int64"),
        "tb_outcome": tb.PUBCHEM_ACTIVITY_OUTCOME,
        "pct_inhibition_25uM": num(tb, "Percent Inhibition @ 25 uM"),
        "IC50": num(tb, "IC50")})
    hep = pd.DataFrame({
        "SID": num(hep, "PUBCHEM_SID").astype("int64"),
        "hepg2_outcome": hep.PUBCHEM_ACTIVITY_OUTCOME,
        "min_pct_viability": num(hep, "Min % Viability")})
    smi = smi.rename(columns={"CID": "CID", "SMILES": "SMILES"})

    df = (sel.merge(tb, on="SID", validate="1:1")
             .merge(hep, on="SID", validate="1:1")
             .merge(smi, on="CID", validate="1:1")
             .merge(arms, on="SID", validate="1:1"))
    if len(df) != len(sel):
        raise SystemExit(f"[error] merged to {len(df)} rows, expected {len(sel)}")

    df["Y_TB_confirmed_active"] = (df.tb_outcome == "Active").astype(int)
    df["Y_HepG2_cytotoxic"] = (df.hepg2_outcome == "Active").astype(int)
    # The selectivity endpoint: how much bacterial killing per unit of hepatocyte killing,
    # on the raw percentages. No curve fit, no imputation.
    df["activity_window"] = df.pct_inhibition_25uM - (100.0 - df.min_pct_viability)

    props = rdkit_properties(df.SMILES.tolist())
    df = df.merge(props, on="SMILES", validate="1:1")
    # Negated so that, as on every other axis, higher means a better molecule.
    df["neg_SAscore"] = -df.SAscore
    df["neg_n_alerts"] = -df.n_alerts

    # The design groups are claims about the two assays; check they hold rather than
    # trusting the selection file.
    expect_tb = df.group.str.startswith("Mtb hit").astype(int)
    expect_hep = df.group.str.endswith("toxic").astype(int) - \
        df.group.str.endswith("nontoxic").astype(int)
    if not (df.Y_TB_confirmed_active == expect_tb).all():
        raise SystemExit("[error] confirmed-active labels disagree with the design groups")
    if not (df.Y_HepG2_cytotoxic == expect_hep).all():
        raise SystemExit("[error] cytotoxicity labels disagree with the design groups")

    columns = ["SMILES", "SID", "CID", "group",
               "Y_TB_confirmed_active", "Y_HepG2_cytotoxic",
               "pct_inhibition_25uM", "min_pct_viability", "activity_window",
               "QED", "neg_SAscore", "neg_n_alerts",
               "fleming_plausibility", "llm_plausibility",
               "dmpnn_pTB", "dmpnn_pTB_x_1minus_pHepG2"]
    out = df[columns]
    if out.isna().any().any():
        raise SystemExit(f"[error] missing values:\n{out.isna().sum()[out.isna().sum() > 0]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[ok] {args.out}  ({len(out)} rows x {len(out.columns)} columns)")
    print(out.groupby("group").size().to_string())

    if args.verify:
        ref = pd.read_csv(args.verify)
        merged = out.merge(ref, on="SID", suffixes=("_new", "_ref"))
        numeric = [c for c in columns if out[c].dtype.kind == "f"]
        worst = max(float((merged[c + "_new"] - merged[c + "_ref"]).abs().max())
                    for c in numeric)
        same_smiles = (merged.SMILES_new == merged.SMILES_ref).all()
        print(f"\nverify against {args.verify}: {len(merged)} rows matched, "
              f"max numeric difference {worst:.2e}, SMILES identical: {same_smiles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

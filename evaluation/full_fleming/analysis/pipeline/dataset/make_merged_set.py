#!/usr/bin/env python
"""

"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["SMILES", "Y_TB", "Y_HepG2", "selection_group", "SID", "CID"]
READOUTS = ["Primary Screen Percent Inhibition @ 25 uM Rep 1", "Percent Inhibition @ 25 uM",
            "IC50", "Maximum Inhibition", "% Cell Viability @ 20 uM", "Min % Viability",
            "CC50", "CC50 Modifier"]

# The only replicated measurement anywhere in this assay series. Rep 1 is universal;
# Rep 2 and Rep 3 exist for ~3% of the library, a QC subset that is mostly inactive
# compounds (mean inhibition 8.8%), not the actives. Averaged here for the record, but
# note the averaged value is not comparable to Rep 1 across molecules: the library was
# triaged on Rep 1, so a molecule that entered the confirmatory screen had a high Rep 1
# by selection, and its later replicates regress downward.
PRIMARY_REPS = ["Primary Screen Percent Inhibition @ 25 uM Rep 1",
                "Primary Screen Percent Inhibition @ 25 uM Rep 2",
                "Primary Screen Percent Inhibition Rep 3"]


def attach_primary_replicates(merged):
    """Mean of whatever primary replicates exist, plus how many there were."""
    bulk = pd.read_csv(os.path.join(_HERE, "hung_pubchem_cache/AID_488890_bulk.csv"),
                       low_memory=False)
    for col in PRIMARY_REPS:
        bulk[col] = pd.to_numeric(bulk[col], errors="coerce")
    reps = pd.DataFrame({
        "SID": bulk["PUBCHEM_SID"],
        "primary_inhib_mean": bulk[PRIMARY_REPS].mean(axis=1),
        "primary_n_rep": bulk[PRIMARY_REPS].notna().sum(axis=1)})
    return merged.merge(reps, on="SID", how="left")


def main():
    old = pd.read_csv(os.path.join(_HERE, "data/nrp_300_continuous.csv"))
    new = pd.read_csv(os.path.join(_HERE, "data/nonconfirmers_150_input.csv"))
    pool = pd.read_csv(os.path.join(_HERE, "data/secondary_pool.csv"))

    new = new.merge(pool[["SID"] + [c for c in READOUTS if c in pool.columns]],
                    on="SID", how="left")
    for col in READOUTS:
        for frame in (old, new):
            if col not in frame.columns:
                frame[col] = np.nan

    merged = pd.concat([old[KEY + READOUTS], new[KEY + READOUTS]], ignore_index=True)
    merged["target"] = ((merged.Y_TB == 1) & (merged.Y_HepG2 == 0)).astype(int)
    merged["log_IC50"] = np.log10(pd.to_numeric(merged.IC50, errors="coerce"))
    merged["activity_window"] = (
        pd.to_numeric(merged["Percent Inhibition @ 25 uM"], errors="coerce")
        - (100.0 - pd.to_numeric(merged["Min % Viability"], errors="coerce")))
    # Primary-side window, defined for all 450: the only endpoint that spans the whole
    # set, at the cost of imputing viability for molecules nobody counter-screened.
    viab = pd.to_numeric(merged["Min % Viability"], errors="coerce").fillna(100.0)
    merged["activity_window_primary"] = (
        pd.to_numeric(merged["Primary Screen Percent Inhibition @ 25 uM Rep 1"],
                      errors="coerce") - (100.0 - viab))

    merged = attach_primary_replicates(merged)
    assert merged.SMILES.nunique() == len(merged) == 450, "duplicate or missing structures"
    path = os.path.join(_HERE, "data/merged_450_continuous.csv")
    merged.to_csv(path, index=False)
    print(f"wrote {path}   n = {len(merged)}")
    print(merged.groupby("selection_group").size().to_string())
    print(f"\n  reached the secondary screen: {int(merged.IC50.notna().sum())}")
    print(f"  TARGET prevalence:            {100 * merged.target.mean():.1f}%")
    print("\n  primary replicates per molecule:")
    print(merged.primary_n_rep.value_counts().sort_index().to_string())

    # The *_dl.csv pair is the one that carries SAscore; the older per-set files predate
    # it being computed here.
    props = pd.concat([pd.read_csv(os.path.join(_HERE, p)) for p in
                       ("data/nrp_hepg2_300_dl.csv",
                        "data/nonconfirmers_150_input_dl.csv")], ignore_index=True)
    props = props.drop_duplicates("SMILES")
    ppath = os.path.join(_HERE, "data/merged_450_druglikeness.csv")
    props.to_csv(ppath, index=False)
    print(f"\nwrote {ppath}   n = {len(props)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

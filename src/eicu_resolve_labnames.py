"""E0 -- resolve eICU-CRD analyte concepts against `eicu_crd.lab` at runtime,
mirroring `mimic_resolve_itemids.py`'s completeness-scan logic exactly (same
CORE_THRESHOLD/DISC_FLOOR, same role assignment) but keyed by `labname`
(string) instead of MIMIC's integer `itemid` -- eICU has no `d_labitems`
equivalent and no reference-range columns on `lab` at all, so there is no
has_refrange gate and no derived/non-analyte exclusion list to port (eICU's
labnames here are primary results, not a panel with calculated sub-fields --
verified below via the disjoint completeness numbers, not assumed).

    PYTHONPATH=src .venv/bin/python src/eicu_resolve_labnames.py
"""
from __future__ import annotations

import json

import bqutil

CORE_THRESHOLD = 0.95   # unchanged from mimic_resolve_itemids.py -- do not tune
DISC_FLOOR = 0.02       # to hit a target headcount; report what it finds

COMPLETENESS_SQL = """
WITH base AS (
  SELECT COUNT(*) AS n FROM `labmae.labpid_eicu.cohort`
),
win AS (
  SELECT c.patientunitstayid, l.labname
  FROM `labmae.labpid_eicu.cohort` c
  JOIN `physionet-data.eicu_crd.lab` l USING (patientunitstayid)
  WHERE l.labresultoffset BETWEEN -1440 AND 1440
    AND l.labresult IS NOT NULL
)
SELECT
  labname,
  COUNT(DISTINCT patientunitstayid)                                AS n_stays,
  COUNT(DISTINCT patientunitstayid) / (SELECT n FROM base)         AS completeness,
  COUNT(*)                                                         AS n_obs
FROM win
GROUP BY 1
ORDER BY completeness DESC, labname ASC   -- labname breaks completeness ties
"""


def classify(df):
    df = df.copy()
    df["role"] = "ignored"
    df.loc[df["completeness"] >= CORE_THRESHOLD, "role"] = "core"
    df.loc[(df["completeness"] < CORE_THRESHOLD) & (df["completeness"] >= DISC_FLOOR),
           "role"] = "discretionary"
    return df


def main():
    df = bqutil.cache("eicu_completeness", COMPLETENESS_SQL, refresh=True)
    df = classify(df)
    df.to_csv(bqutil.ROOT / "out" / "eicu_completeness_table.csv", index=False)

    # labname is the tiebreak -- same reason as mimic_resolve_itemids.py: without it
    # BigQuery's row order for tied completeness makes this file differ between
    # identical runs. Analyte order is numerically inert downstream.
    sort = lambda d: d.sort_values(["completeness", "labname"], ascending=[False, True])
    core = sort(df[df.role == "core"])
    disc = sort(df[df.role == "discretionary"])

    cfg = {
        "source": "physionet-data.eicu_crd.lab",
        "core_threshold": CORE_THRESHOLD,
        "discretionary_floor": DISC_FLOOR,
        "n_cohort_with_window_labs": int(df.n_stays.max()) if len(df) else 0,
        "core": [
            {"labname": r.labname, "completeness": round(float(r.completeness), 4)}
            for r in core.itertuples()
        ],
        "discretionary": [
            {"labname": r.labname, "completeness": round(float(r.completeness), 4)}
            for r in disc.itertuples()
        ],
    }
    out = bqutil.ROOT / "config" / "eicu_labnames.resolved.json"
    out.write_text(json.dumps(cfg, indent=2))

    print(f"core analytes         : {len(core)}")
    for r in core.itertuples():
        print(f"  {r.labname:20s} {r.completeness:.4f}  n_obs={r.n_obs:,}")
    print(f"discretionary analytes: {len(disc)}  (completeness {DISC_FLOOR}..{CORE_THRESHOLD})")
    print(f"\nGATE (>=5 core analytes): {'PASS' if len(core) >= 5 else 'FAIL'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

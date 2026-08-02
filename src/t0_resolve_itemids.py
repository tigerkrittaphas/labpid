"""T0 — resolve analyte concepts against d_labitems at runtime and split them
into the core (value-channel) and discretionary (structure-channel) sets.

Nothing is hardcoded except the *exclusion rules*, which are stated as label
patterns and justified inline. Run counts come from the in-cohort window so a
concept that has drifted out of a release fails loudly rather than silently.
"""
from __future__ import annotations

import json
import re

import bqutil

CORE_THRESHOLD = 0.95      # T2: value channel = near-complete analytes
DISC_FLOOR = 0.02          # below this a mask column is ~constant zero, no signal

# --- exclusion rules -------------------------------------------------------
# 1. Arithmetic functions of other analytes carry zero independent information.
#    MCV/MCH/MCHC/RDW are functions of Hgb/Hct/RBC; Anion Gap = Na - Cl - HCO3;
#    eAG is a function of HbA1c; the calculated cholesterol/CO2/Hct entries and
#    CK-MB index are likewise closed-form.
DERIVED = {
    "MCV", "MCH", "MCHC", "RDW", "RDW-SD", "Anion Gap", "eAG",
    "Cholesterol, LDL, Calculated", "Cholesterol Ratio (Total/HDL)",
    "Calculated Total CO2", "Hematocrit, Calculated", "CK-MB Index",
}
# 2. Not analytes: ventilator settings and specimen-quality indices recorded on
#    the same rows as blood gases / chemistry panels.
NON_ANALYTE = {"PEEP", "Tidal Volume", "Temperature", "Oxygen", "H", "I", "L",
               "Ventilator", "Intubated", "Ventilation Rate"}

COMPLETENESS_SQL = """
WITH base AS (
  SELECT COUNT(DISTINCT subject_id) AS n FROM `{ds}.labs_48h` WHERE phase = 'W'
)
SELECT
  d.itemid, d.label, d.fluid, d.category,
  COUNT(DISTINCT l.subject_id)                                     AS n_subj,
  COUNT(DISTINCT l.subject_id) / (SELECT n FROM base)              AS completeness,
  COUNT(*)                                                         AS n_obs,
  COUNTIF(l.ref_range_lower IS NOT NULL
          AND l.ref_range_upper IS NOT NULL) / COUNT(*)            AS frac_refrange,
  COUNT(DISTINCT CONCAT(CAST(l.ref_range_lower AS STRING), '|',
                        CAST(l.ref_range_upper AS STRING)))        AS n_distinct_ranges
FROM `{ds}.labs_48h` l
JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d USING (itemid)
WHERE l.phase = 'W' AND l.valuenum IS NOT NULL
GROUP BY 1, 2, 3, 4
ORDER BY completeness DESC
"""


def classify(df):
    lab = df["label"].fillna("")
    df = df.copy()
    df["excluded"] = ""
    df.loc[lab.isin(DERIVED), "excluded"] = "derived"
    df.loc[lab.isin(NON_ANALYTE), "excluded"] = "non_analyte"
    df.loc[df["fluid"] != "Blood", "excluded"] = df.loc[df["fluid"] != "Blood", "excluded"].replace("", "non_blood")
    # An analyte with no reference range cannot yield A_j; it may still act as
    # structure, so it is only barred from the value channel / target set.
    df["has_refrange"] = df["frac_refrange"] >= 0.90

    keep = df["excluded"] == ""
    df["role"] = "ignored"
    df.loc[keep & (df["completeness"] >= CORE_THRESHOLD) & df["has_refrange"], "role"] = "core"
    df.loc[keep & (df["completeness"] < CORE_THRESHOLD)
           & (df["completeness"] >= DISC_FLOOR), "role"] = "discretionary"
    return df


def main():
    df = bqutil.cache("completeness", COMPLETENESS_SQL, refresh=True)
    df = classify(df)
    df.to_csv(bqutil.ROOT / "out" / "completeness_table.csv", index=False)

    core = df[df.role == "core"].sort_values("completeness", ascending=False)
    disc = df[df.role == "discretionary"].sort_values("completeness", ascending=False)

    # Duplicate-concept check: the same normalized label resolving to >1 itemid.
    norm = df.assign(key=df.label.str.lower().str.strip())
    dupes = (norm[norm.role != "ignored"].groupby("key")["itemid"]
             .agg(list).loc[lambda s: s.str.len() > 1].to_dict())

    cfg = {
        "source": "physionet-data.mimiciv_3_1_hosp.d_labitems",
        "core_threshold": CORE_THRESHOLD,
        "discretionary_floor": DISC_FLOOR,
        "n_cohort_with_window_labs": int(df.n_subj.max()),
        "core": [
            {"itemid": int(r.itemid), "label": r.label, "completeness": round(float(r.completeness), 4)}
            for r in core.itertuples()
        ],
        "discretionary": [
            {"itemid": int(r.itemid), "label": r.label, "completeness": round(float(r.completeness), 4)}
            for r in disc.itertuples()
        ],
        "excluded": {
            reason: sorted({r.label for r in df[df.excluded == reason].itertuples()
                            if r.completeness >= DISC_FLOOR})
            for reason in ("derived", "non_analyte", "non_blood")
        },
        "duplicate_label_itemids": {k: [int(v) for v in vs] for k, vs in dupes.items()},
    }
    out = bqutil.ROOT / "config" / "itemids.resolved.json"
    out.write_text(json.dumps(cfg, indent=2))

    print(f"core analytes         : {len(core)}")
    for r in core.itertuples():
        print(f"  {r.itemid:6d}  {r.label:28s} {r.completeness:.4f}  ranges={r.n_distinct_ranges}")
    print(f"discretionary analytes: {len(disc)}  (completeness {DISC_FLOOR}..{CORE_THRESHOLD})")
    print(f"excluded derived      : {sorted(cfg['excluded']['derived'])}")
    print(f"duplicate labels      : {cfg['duplicate_label_itemids']}")
    print(f"\nGATE T2 (>=5 core analytes): {'PASS' if len(core) >= 5 else 'FAIL'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

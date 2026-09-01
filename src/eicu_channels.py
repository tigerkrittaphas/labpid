"""E3 -- eICU-CRD paired value/time channels: first & last draw per analyte,
selected by time alone (sql/eicu_02_paired_channels.sql). Mirrors
`mimic_channels.py`'s Channels/complete_case/impute/retarget logic
exactly, adapted for eICU's schema:

  - keyed by `labname` (string) instead of MIMIC's integer `itemid`
  - group/primary key is `patientunitstayid` (not `subject_id`)
  - all tables/caches live under the `eicu_` prefix / `labmae.labpid_eicu`
    dataset, fully separate from the MIMIC-critical `labmae.labpid.cohort`

A parallel module rather than a parameterized shared one: mimic's `_pull()`
hardcodes MIMIC column names and reads `config/itemids.resolved.json` at
import time, so making one module serve both datasets would mean threading
5-6 dataset-specific parameters through every call for what is, for now, a
single replication dataset. Not worth it while there's only one.

    PYTHONPATH=src .venv/bin/python src/eicu_channels.py
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import bqutil

ROOT = bqutil.ROOT
CFG = json.loads((ROOT / "config" / "eicu_labnames.resolved.json").read_text())
CORE = {d["labname"]: d["labname"] for d in CFG["core"]}
DISC = {d["labname"]: d["labname"] for d in CFG["discretionary"]}

TIME_SENTINEL = -999.0  # real offsets fall in [-24, 24]; -999 unambiguously means "never ordered"
VALUE_SUFFIXES = ("value_first", "value_last")
TIME_SUFFIXES = ("time_first", "time_last")


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _name_map() -> dict[str, str]:
    labels = {**CORE, **DISC}
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[slug(lab)] = counts.get(slug(lab), 0) + 1
    return {name: (slug(name) if counts[slug(name)] == 1 else f"{slug(name)}_{i}")
            for i, name in enumerate(labels)}


NAME = _name_map()


def _pull(refresh=False):
    long = bqutil.cache("eicu_paired_first_last",
                         "SELECT * FROM `labmae.labpid_eicu.paired_first_last`", refresh)
    cohort = bqutil.cache("eicu_cohort",
                           "SELECT patientunitstayid, hospitalid, age, gender, "
                           "ethnicity, hospital_expire_flag FROM `labmae.labpid_eicu.cohort`",
                           refresh)
    return long, cohort


def demographic_targets(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """patientunitstayid-indexed frame with gender_female, age_ge65,
    race_binary (NaN = excluded/unknown) -- eICU analog of
    mimic_channels.demographic_targets(). `race_binary` is named to
    match the MIMIC target set for pipeline/figure parity even though it's
    sourced from eICU's `ethnicity` field (config/eicu_ethnicity_bucket_map.json),
    not a field literally called `race` -- eICU has no separate race/
    ethnicity distinction the way MIMIC does. gender_female/age_ge65 are not
    null controls (real physiological links to lab values expected); the
    ethnicity contrast (Caucasian vs African American, the two largest
    eICU ethnicity buckets after excluding Other/Unknown/blank) is the
    actual bias probe, mirroring MIMIC's WHITE vs BLACK/AFRICAN AMERICAN
    contrast."""
    _, cohort = _pull(refresh)

    df = cohort.copy()
    # gender has a small "Unknown"/"Other"/blank tail (58 of 137,773 rows) --
    # exclude rather than silently code as not-female.
    df["gender_female"] = np.select(
        [df["gender"] == "Female", df["gender"] == "Male"], [1.0, 0.0], default=np.nan)
    df["age_ge65"] = (df["age"] >= 65).astype(float)

    race_cfg = json.loads((ROOT / "config" / "eicu_ethnicity_bucket_map.json").read_text())
    bucket = df["ethnicity"].map(race_cfg["race_map"])
    a, b = race_cfg["contrast"]
    df["race_binary"] = np.select([bucket == a, bucket == b], [0.0, 1.0], default=np.nan)

    info = {
        "race_contrast": race_cfg["contrast"],
        "race_bucket_sizes": race_cfg["bucket_sizes"],
        "race_excluded_counts": df.loc[df["race_binary"].isna(), "ethnicity"]
                                    .value_counts().to_dict(),
    }
    out = df.set_index("patientunitstayid")[["gender_female", "age_ge65", "race_binary"]]
    return out, info


def build_wide(labname_scope: set[str], refresh: bool = False):
    """Pivot the long paired table to one row per patientunitstayid, 4
    columns per analyte in scope. Missing entries are NaN -- this function
    does not fill or drop anything, that choice belongs to the arms below."""
    long, cohort = _pull(refresh)
    long = long[long.labname.isin(labname_scope)].copy()

    wide = cohort[["patientunitstayid"]].copy()
    v_cols, s_cols = [], []
    for labname in sorted(labname_scope):
        name = NAME[labname]
        sub = long[long.labname == labname].set_index("patientunitstayid")
        for suf in VALUE_SUFFIXES + TIME_SUFFIXES:
            col = f"{name}_{suf}"
            wide[col] = (sub[suf].reindex(wide.patientunitstayid).to_numpy()
                         if len(sub) else np.nan)
            (v_cols if suf.startswith("value") else s_cols).append(col)
    return wide, v_cols, s_cols


def audit(v_cols: list[str], s_cols: list[str]) -> None:
    assert all(c.endswith(VALUE_SUFFIXES) for c in v_cols), "non-value column in V"
    assert all(c.endswith(TIME_SUFFIXES) for c in s_cols), "non-time column in S"
    assert not (set(v_cols) & set(s_cols)), "V/S overlap"


@dataclass
class Channels:
    Xv: np.ndarray
    Xs: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    v_names: list[str]
    s_names: list[str]
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Channels(n={len(self.y):,}  V={self.Xv.shape[1]}d  "
                f"S={self.Xs.shape[1]}d  arm={self.meta.get('arm')}  "
                f"prevalence={self.y.mean():.4f})")


def complete_case(labname_scope: set[str] | None = None,
                   target: str = "hospital_expire_flag",
                   refresh: bool = False) -> Channels:
    labname_scope = labname_scope or set(CORE)
    wide, v_cols, s_cols = build_wide(labname_scope, refresh)
    audit(v_cols, s_cols)
    _, cohort = _pull(refresh)

    keep = wide[v_cols + s_cols].notna().all(axis=1)
    w = wide[keep].merge(cohort[["patientunitstayid", target]], on="patientunitstayid")

    Xv = w[v_cols].to_numpy(dtype=float)
    Xs = w[s_cols].to_numpy(dtype=float)
    y = w[target].astype(int).to_numpy()
    meta = {"arm": "complete_case", "labname_scope": sorted(labname_scope),
            "target": target, "n": int(len(y)),
            "n_dropped": int((~keep).sum()),
            "drop_rate": float((~keep).mean()), "prevalence": float(y.mean())}
    return Channels(Xv, Xs, y, w.patientunitstayid.to_numpy(), v_cols, s_cols, meta)


def impute(labname_scope: set[str] | None = None,
           target: str = "hospital_expire_flag",
           method: str = "mean",
           refresh: bool = False) -> Channels:
    labname_scope = labname_scope or set(CORE)
    wide, v_cols, s_cols = build_wide(labname_scope, refresh)
    audit(v_cols, s_cols)
    _, cohort = _pull(refresh)

    w = wide.merge(cohort[["patientunitstayid", target]], on="patientunitstayid")
    Xv_raw = w[v_cols].to_numpy(dtype=float)
    Xs_raw = w[s_cols].to_numpy(dtype=float)

    Xs = np.where(np.isfinite(Xs_raw), Xs_raw, TIME_SENTINEL)

    if method == "mean":
        mu = np.nanmean(Xv_raw, axis=0)
        Xv = np.where(np.isfinite(Xv_raw), Xv_raw, mu)
    elif method == "mice":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        Xv = IterativeImputer(random_state=0).fit_transform(Xv_raw)
    else:
        raise ValueError(f"unknown imputation method {method!r}")

    y = w[target].astype(int).to_numpy()
    meta = {"arm": f"impute_{method}", "labname_scope": sorted(labname_scope),
            "target": target, "n": int(len(y)), "time_sentinel": TIME_SENTINEL,
            "v_missing_rate": float(np.mean(~np.isfinite(Xv_raw))),
            "prevalence": float(y.mean())}
    return Channels(Xv, Xs, y, w.patientunitstayid.to_numpy(), v_cols, s_cols, meta)


def retarget(ch: Channels, y_new: pd.Series, target_name: str) -> Channels:
    """Swap an already-built arm's target without rebuilding Xv/Xs -- kept
    for parity with mimic_channels.py's retarget(), not used by this
    pass's mortality-only scope."""
    assert len(np.unique(ch.groups)) == len(ch.groups), "ch.groups must be unique patientunitstayids"
    y_aligned = y_new.reindex(ch.groups)
    keep = y_aligned.notna().to_numpy()
    meta = {**ch.meta, "target": target_name,
            "base_arm": ch.meta.get("arm"), "base_arm_n": ch.meta.get("n"),
            "n": int(keep.sum()), "n_dropped_for_target": int((~keep).sum()),
            "drop_rate_for_target": float((~keep).mean()),
            "prevalence": float(y_aligned[keep].mean())}
    return Channels(ch.Xv[keep], ch.Xs[keep], y_aligned[keep].to_numpy(dtype=int),
                     ch.groups[keep], ch.v_names, ch.s_names, meta)


if __name__ == "__main__":
    cc = complete_case()
    print("complete-case:", cc, "  dropped:", cc.meta["n_dropped"],
          f"({cc.meta['drop_rate']:.1%})")

    im_mean = impute(method="mean")
    print("impute (mean): ", im_mean)

    im_mice = impute(method="mice")
    print("impute (mice): ", im_mice)

    print(f"\nV[:4] {cc.v_names[:4]}")
    print(f"S[:4] {cc.s_names[:4]}")

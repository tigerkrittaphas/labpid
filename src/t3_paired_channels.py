"""T3 -- paired value/time channels: first & last draw per analyte, selected
by time alone, never by value (sql/04_paired_channels.sql). V and S are
separable by construction, so the audit is a one-line suffix check instead of
a maintained column list, and nothing needs to be redacted.

Two arms for missingness, built from the same wide table:

  complete_case(...)  Drop patients missing any selected analyte. No fill, no
                       sentinel -- the honest baseline, at the cost of N.

  impute(...)          Keep every patient. Missing time_first/time_last get an
                       out-of-range sentinel (TIME_SENTINEL), which makes
                       "never ordered" a real value living in S, rather than a
                       derived mask bolted on afterward. Missing
                       value_first/value_last are imputed by a model that
                       reads ONLY other V columns -- conditioning on anything
                       in S here would smuggle structure back into value
                       through the imputation step itself, which is exactly
                       the leak this whole redesign exists to avoid.

Default itemid scope is the core analyte set: complete-case over the full
~84-analyte discretionary set will shred the sample size, so CORE is the
primary arm and DISC is a stress test, not swapped by default.

    PYTHONPATH=src .venv/bin/python src/t3_paired_channels.py
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import bqutil

ROOT = bqutil.ROOT
CFG = json.loads((ROOT / "config" / "itemids.resolved.json").read_text())
CORE = {int(d["itemid"]): d["label"] for d in CFG["core"]}
DISC = {int(d["itemid"]): d["label"] for d in CFG["discretionary"]}

# Concept-level aliases (e.g. CBC Hgb vs blood-gas Hgb): kept for future
# per-analyte-target tracks that need T6-style leakage banning. The mortality
# target used by default below is not analyte-derived, so nothing bans on it.
CONCEPT_ALIASES: dict[int, list[int]] = {}
for _grp in CFG["duplicate_label_itemids"].values():
    for _i in _grp:
        CONCEPT_ALIASES[int(_i)] = [int(x) for x in _grp]

TIME_SENTINEL = -999.0  # real offsets fall in [-24, 24); -999 unambiguously means "never ordered"
VALUE_SUFFIXES = ("value_first", "value_last")
TIME_SUFFIXES = ("time_first", "time_last")


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _name_map() -> dict[int, str]:
    labels = {**CORE, **DISC}
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[slug(lab)] = counts.get(slug(lab), 0) + 1
    return {i: (slug(l) if counts[slug(l)] == 1 else f"{slug(l)}_{i}")
            for i, l in labels.items()}


NAME = _name_map()


def _pull(refresh=False):
    long = bqutil.cache("paired_first_last",
                         "SELECT * FROM `{ds}.paired_first_last`", refresh)
    cohort = bqutil.cache("cohort", "SELECT subject_id, hadm_id, stay_id, anchor_age, "
                                     "gender, icu_los_days, hospital_expire_flag "
                                     "FROM `{ds}.cohort`", refresh)
    return long, cohort


def build_wide(itemid_scope: set[int], refresh: bool = False):
    """Pivot the long paired table to one row per subject_id, 4 columns per
    analyte in scope. Missing entries are NaN -- this function does not fill
    or drop anything, that choice belongs to the arms below."""
    long, cohort = _pull(refresh)
    long = long[long.itemid.isin(itemid_scope)].copy()

    wide = cohort[["subject_id"]].copy()
    v_cols, s_cols = [], []
    for itemid in sorted(itemid_scope):
        name = NAME[itemid]
        sub = long[long.itemid == itemid].set_index("subject_id")
        for suf in VALUE_SUFFIXES + TIME_SUFFIXES:
            col = f"{name}_{suf}"
            wide[col] = sub[suf].reindex(wide.subject_id).to_numpy() if len(sub) else np.nan
            (v_cols if suf.startswith("value") else s_cols).append(col)
    return wide, v_cols, s_cols


def audit(v_cols: list[str], s_cols: list[str]) -> None:
    """The rule is checkable by column name alone -- no maintained provenance
    dict to fall out of sync with the SQL."""
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


def complete_case(itemid_scope: set[int] | None = None,
                   target: str = "hospital_expire_flag",
                   refresh: bool = False) -> Channels:
    itemid_scope = itemid_scope or set(CORE)
    wide, v_cols, s_cols = build_wide(itemid_scope, refresh)
    audit(v_cols, s_cols)
    _, cohort = _pull(refresh)

    keep = wide[v_cols + s_cols].notna().all(axis=1)
    w = wide[keep].merge(cohort[["subject_id", target]], on="subject_id")

    Xv = w[v_cols].to_numpy(dtype=float)
    Xs = w[s_cols].to_numpy(dtype=float)
    y = w[target].astype(int).to_numpy()
    meta = {"arm": "complete_case", "itemid_scope": sorted(itemid_scope),
            "target": target, "n": int(len(y)),
            "n_dropped": int((~keep).sum()),
            "drop_rate": float((~keep).mean()), "prevalence": float(y.mean())}
    return Channels(Xv, Xs, y, w.subject_id.to_numpy(), v_cols, s_cols, meta)


def impute(itemid_scope: set[int] | None = None,
           target: str = "hospital_expire_flag",
           method: str = "mean",
           refresh: bool = False) -> Channels:
    itemid_scope = itemid_scope or set(CORE)
    wide, v_cols, s_cols = build_wide(itemid_scope, refresh)
    audit(v_cols, s_cols)
    _, cohort = _pull(refresh)

    w = wide.merge(cohort[["subject_id", target]], on="subject_id")
    Xv_raw = w[v_cols].to_numpy(dtype=float)
    Xs_raw = w[s_cols].to_numpy(dtype=float)

    # S: missing time IS data -- "never ordered" -- not something to hide.
    Xs = np.where(np.isfinite(Xs_raw), Xs_raw, TIME_SENTINEL)

    # V: imputed reading only other V columns, never S. Conditioning the
    # imputer on order-log content (e.g. "how many labs were drawn") would
    # leak structure back into value through the imputation model itself.
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
    meta = {"arm": f"impute_{method}", "itemid_scope": sorted(itemid_scope),
            "target": target, "n": int(len(y)), "time_sentinel": TIME_SENTINEL,
            "v_missing_rate": float(np.mean(~np.isfinite(Xv_raw))),
            "prevalence": float(y.mean())}
    return Channels(Xv, Xs, y, w.subject_id.to_numpy(), v_cols, s_cols, meta)


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

"""T3-T6 — pull the long-format aggregates, pivot to admission-level matrices,
and persist. Leakage removal (T6) is applied per target at analysis time by
`assemble()`, which is the only function allowed to hand out feature blocks.

Channel discipline:
  V  = core-analyte values only (first/last/min/max/mean/slope). No counts,
       no mask, no timing.
  S  = ordering process only. Volume, timing, time-of-day, STAT fraction,
       discretionary counts+masks, and per-core-analyte counts. No values.

Per-core-analyte counts are a documented extension of the plan's T4 table: they
are counts, not values, so they are structural by definition, and the plan
already admits the aggregate versions (n_results, n_distinct_analytes).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import bqutil

ROOT = bqutil.ROOT
CFG = json.loads((ROOT / "config" / "itemids.resolved.json").read_text())
CORE = {int(d["itemid"]): d["label"] for d in CFG["core"]}
DISC = {int(d["itemid"]): d["label"] for d in CFG["discretionary"]}

VALUE_STATS = ["v_first", "v_last", "v_min", "v_max", "v_mean", "v_slope"]


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


# Column names are keyed by itemid, not by label: the same analyte measured on a
# different instrument resolves to a second itemid with an identical label (CBC
# vs blood-gas Hgb, chemistry vs blood-gas glucose), so labels are not unique.
# Colliding slugs get an itemid suffix; leakage removal then works on itemids.
def _name_map() -> dict[int, str]:
    labels = {**CORE, **DISC}
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[slug(lab)] = counts.get(slug(lab), 0) + 1
    return {i: (slug(l) if counts[slug(l)] == 1 else f"{slug(l)}_{i}")
            for i, l in labels.items()}


NAME = _name_map()

# Concept-level leakage: those same duplicate-label pairs measure one concept.
# Removing only the target itemid would leave its twin behind.
CONCEPT_ALIASES: dict[int, list[int]] = {}
for _grp in CFG["duplicate_label_itemids"].values():
    for _i in _grp:
        CONCEPT_ALIASES[int(_i)] = [int(x) for x in _grp]


def _pivot(df, index, columns, values, prefix):
    w = df.pivot(index=index, columns=columns, values=values)
    w.columns = [f"{prefix}{c}" for c in w.columns]
    return w.reset_index()


def build():
    out = ROOT / "out"

    cohort = bqutil.cache("cohort", "SELECT subject_id, hadm_id, stay_id, anchor_age, gender, "
                                    "icu_los_days, hospital_expire_flag FROM `{ds}.cohort`")
    val = bqutil.cache("value_long", "SELECT * FROM `{ds}.value_long`")
    sg = bqutil.cache("struct_global", "SELECT * FROM `{ds}.struct_global`")
    sd = bqutil.cache("struct_disc_long", "SELECT * FROM `{ds}.struct_disc_long`")
    tg = bqutil.cache("targets_long", "SELECT subject_id, itemid, valuenum, abnormal, "
                                      "hours_from_t0 FROM `{ds}.targets_long`")

    # ---- value channel: core analytes only, wide ---------------------------
    v = val[val.itemid.isin(CORE)].copy()
    v["name"] = v.itemid.map(NAME)
    blocks = []
    for stat in VALUE_STATS:
        blocks.append(_pivot(v[["subject_id", "name", stat]], "subject_id", "name", stat,
                             f"{stat[2:]}_").set_index("subject_id"))
    V = pd.concat(blocks, axis=1).reset_index()
    # slope is 0, not missing, when a single observation was taken
    for c in [c for c in V.columns if c.startswith("slope_")]:
        base = c.split("_", 1)[1]
        V[c] = V[c].where(~(V[c].isna() & V[f"mean_{base}"].notna()), 0.0)

    # ---- structural channel ------------------------------------------------
    # counts for every analyte (core + discretionary); masks for discretionary
    sd_keep = sd[sd.itemid.isin(set(CORE) | set(DISC))].copy()
    sd_keep["name"] = sd_keep.itemid.map(NAME)
    Cnt = _pivot(sd_keep[["subject_id", "name", "n_obs"]], "subject_id", "name", "n_obs", "cnt_")
    Cnt = Cnt.fillna(0.0)
    # .to_numpy() is load-bearing: these Series carry Cnt's RangeIndex, so
    # handing them to a DataFrame keyed by subject_id would align on the wrong
    # index and produce an all-NaN (i.e. silently empty) mask block.
    mask_cols = {f"msk_{NAME[i]}": (Cnt[f"cnt_{NAME[i]}"] > 0).astype(float).to_numpy()
                 for i in DISC if f"cnt_{NAME[i]}" in Cnt}
    S = pd.concat([sg.set_index("subject_id"),
                   Cnt.set_index("subject_id"),
                   pd.DataFrame(mask_cols, index=Cnt.subject_id).rename_axis("subject_id")],
                  axis=1).reset_index()

    # ---- targets -----------------------------------------------------------
    t = tg[tg.abnormal.notna()].copy()
    T = _pivot(t[["subject_id", "itemid", "abnormal"]], "subject_id", "itemid", "abnormal", "tgt_")

    for name, df in [("V", V), ("S", S), ("T", T)]:
        df.to_parquet(out / f"channel_{name}.parquet", index=False)
    cohort.to_parquet(out / "cohort_local.parquet", index=False)

    print(f"V: {V.shape}  S: {S.shape}  T: {T.shape}  cohort: {cohort.shape}")
    return V, S, T, cohort


def load():
    out = ROOT / "out"
    return (pd.read_parquet(out / "channel_V.parquet"),
            pd.read_parquet(out / "channel_S.parquet"),
            pd.read_parquet(out / "channel_T.parquet"),
            pd.read_parquet(out / "cohort_local.parquet"))


def assemble(target_itemid: int, V=None, S=None, T=None, cohort=None, impute_all=False):
    """Return (Xv, Xs, y, groups, meta) with T6 leakage removal applied.

    impute_all=False -> value channel is core analytes only (the T2 design).
    impute_all=True  -> value channel additionally carries every discretionary
                        analyte with a population fill, which is the T13
                        encoding-artifact arm.
    """
    if V is None:
        V, S, T, cohort = load()
    tcol = f"tgt_{target_itemid}"
    if tcol not in T.columns:
        raise KeyError(f"no target column for itemid {target_itemid}")

    banned = set(CONCEPT_ALIASES.get(target_itemid, [])) | {target_itemid}
    banned_names = {NAME[i] for i in banned if i in NAME}
    # every prefix a channel column can carry for an analyte
    banned_cols = {f"{p}_{n}" for n in banned_names
                   for p in ("first", "last", "min", "max", "mean", "slope", "cnt", "msk")}

    def drop(df):
        return df[[c for c in df.columns if c not in banned_cols]]

    Vd, Sd = drop(V), drop(S).copy()

    # the target's own results still sit inside the aggregate volume counters
    own = np.zeros(len(Sd))
    for n in banned_names:
        if f"cnt_{n}" in S.columns:
            own = own + S[f"cnt_{n}"].fillna(0.0).to_numpy()
    Sd["n_results_total"] = Sd["n_results_total"] - own
    Sd["n_distinct_analytes"] = Sd["n_distinct_analytes"] - (own > 0).astype(float)

    df = (cohort[["subject_id"]]
          .merge(Vd, on="subject_id").merge(Sd, on="subject_id")
          .merge(T[["subject_id", tcol]].dropna(), on="subject_id"))

    vcols = [c for c in Vd.columns if c != "subject_id"]
    scols = [c for c in Sd.columns if c != "subject_id"]

    if impute_all:
        raw_disc = pd.read_parquet(ROOT / "out" / "value_long.parquet")
        raw_disc = raw_disc[raw_disc.itemid.isin(DISC)].copy()
        raw_disc["name"] = raw_disc.itemid.map(NAME)
        extra = _pivot(raw_disc[["subject_id", "name", "v_last"]], "subject_id",
                       "name", "v_last", "dlast_")
        extra = extra.drop(columns=[f"dlast_{n}" for n in banned_names
                                    if f"dlast_{n}" in extra.columns])
        df = df.merge(extra, on="subject_id", how="left")
        vcols = vcols + [c for c in extra.columns if c != "subject_id"]

    y = df[tcol].astype(int).to_numpy()
    groups = df["subject_id"].to_numpy()
    Xv = df[vcols].astype(float)
    Xs = df[scols].astype(float)
    # population-constant fill (T3): small by construction on the core block
    Xv = Xv.fillna(Xv.median()).fillna(0.0).to_numpy()
    Xs = Xs.fillna(Xs.median()).fillna(0.0).to_numpy()

    # T6 assertion: no surviving column may reference the target concept
    for c in vcols + scols:
        stem = c.split("_", 1)[1] if "_" in c else c
        assert stem not in banned_names, f"leak: column {c} references target concept"

    meta = {"target_itemid": int(target_itemid),
            "target_label": {**CORE, **DISC}.get(target_itemid, "?"),
            "n": int(len(y)), "n_patients": int(len(set(groups))),
            "prevalence": float(y.mean()),
            "dim_V": len(vcols), "dim_S": len(scols),
            "banned_itemids": sorted(int(b) for b in banned),
            "banned_names": sorted(banned_names),
            "impute_all": impute_all}
    return Xv, Xs, y, groups, meta, (vcols, scols)


if __name__ == "__main__":
    build()

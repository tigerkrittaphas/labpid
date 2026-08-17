"""Track A (PRIMARY) — discrete BROJA decomposition per target analyte.

A1 builds the two 3-level summaries deliberately (a clinically meaningful value
and an ordering-pattern variable), A2 estimates the atoms, A3 reports cell
occupancy and Miller-Madow-corrected node MIs, A4 runs the permutation null and
A5 sweeps the binning grid.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bqutil
import pid_broja as pb
import t3_build_channels as t3

ROOT = bqutil.ROOT

# A1 — Ṽ is the most recent in-window value of the analyte most physiologically
# related to the target, binned by that analyte's own canonical reference range.
PARTNER = {
    51265: 51222,   # Platelet Count  -> Hemoglobin
    51301: 51222,   # WBC             -> Hemoglobin
    50912: 51006,   # Creatinine      -> Urea Nitrogen
    51222: 51221,   # Hemoglobin      -> Hematocrit
    51006: 50912,   # Urea Nitrogen   -> Creatinine
    50882: 50902,   # Bicarbonate     -> Chloride
}
PARTNER_ALT = {
    51265: 51301, 51301: 51265, 50912: 50971,
    51222: 51279, 51006: 50983, 50882: 50931,
}
TARGETS = list(PARTNER)


def load_ref_canon() -> pd.DataFrame:
    return bqutil.cache("ref_canon", "SELECT itemid, gender, canon_lower, canon_upper "
                                     "FROM `{ds}.ref_canon`")


def v_tilde(df, partner_itemid, ref, scheme="refrange3"):
    """Bin the partner analyte's last in-window value."""
    col = f"last_{t3.NAME[partner_itemid]}"
    x = df[col].astype(float).to_numpy()
    if scheme.startswith("refrange"):
        r = ref[ref.itemid == partner_itemid].set_index("gender")
        lo = df.gender.map(r.canon_lower).astype(float).to_numpy()
        hi = df.gender.map(r.canon_upper).astype(float).to_numpy()
        if scheme == "refrange3":
            return np.where(x < lo, 0, np.where(x > hi, 2, 1))
        return ((x < lo) | (x > hi)).astype(int)          # refrange2
    k = {"tertile3": 3, "quartile4": 4, "median2": 2}[scheme]
    qs = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(qs, x)


def s_tilde(df, scheme="amlab3"):
    """Bin the ordering-pattern variable."""
    if scheme == "amlab3":
        f = df.frac_amlab.astype(float).to_numpy()
        return np.where(f <= 0.2, 0, np.where(f >= 0.8, 2, 1))
    if scheme == "amlab2":
        return (df.frac_amlab.astype(float).to_numpy() > 0.2).astype(int)
    if scheme == "offhours3":
        f = df.frac_offhours.astype(float).to_numpy()
        return np.where(f <= 0.2, 0, np.where(f >= 0.8, 2, 1))
    src = {"nspec3": "n_specimens", "gapcv3": "gap_cv", "stat3": "frac_stat",
           "discmask3": "_n_disc_mask", "discmask4": "_n_disc_mask"}[scheme]
    k = 4 if scheme.endswith("4") else 3
    x = df[src].astype(float).fillna(0.0).to_numpy()
    qs = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
    return np.searchsorted(qs, x)


def collapse_rare(levels, min_share=0.02):
    """A6 remedy — merge any level holding less than `min_share` of the mass into
    its nearest surviving neighbour. Reference-range trichotomies go degenerate
    in an ICU cohort (almost nobody has *high* haemoglobin), which starves cells
    and breaks the occupancy gate; collapsing is what A6 prescribes."""
    levels = np.asarray(levels)
    uniq = np.unique(levels)
    shares = {u: float((levels == u).mean()) for u in uniq}
    keep = [u for u in uniq if shares[u] >= min_share]
    # never collapse to a single level: a constant summary is not a binning
    # scheme, it carries no information and would enter the A5 grid as a
    # spurious "synergy = 0" cell.
    if len(keep) < min(2, len(uniq)):
        keep = sorted(sorted(uniq, key=lambda u: -shares[u])[:min(2, len(uniq))])
    remap = {u: min(keep, key=lambda k: abs(int(k) - int(u))) for u in uniq}
    out = np.array([remap[u] for u in levels])
    _, inv = np.unique(out, return_inverse=True)
    return inv, {int(k): int(v) for k, v in remap.items()}


def collapse_for_occupancy(v_raw, s_raw, y, min_cell=20, shares=(0.02, 0.05, 0.10, 0.20)):
    """A6 remedy, driven by the gate it exists to satisfy.

    Collapsing on marginal share alone is not enough: two merely uncommon levels
    can still intersect in a starved cell (creatinine's BUN-low x routine-only
    cell holds 8 admissions at n=56k). Escalate the merge threshold until the
    occupancy gate passes, and report the threshold that was needed."""
    best = None
    for sh in shares:
        v, vm = collapse_rare(v_raw, sh)
        s, sm = collapse_rare(s_raw, sh)
        tab, _ = pb.joint_table(v, s, y)
        best = (v, s, {"min_share": sh, "min_cell": float(tab.min()),
                       "v_map": vm, "s_map": sm,
                       "levels": [int(tab.shape[0]), int(tab.shape[1])]})
        if tab.min() >= min_cell:
            return best
        if tab.shape[0] <= 2 and tab.shape[1] <= 2:
            break
    return best


V_SCHEMES = ["refrange3", "refrange2", "tertile3", "quartile4"]
S_SCHEMES = ["amlab3", "amlab2", "offhours3", "nspec3", "gapcv3", "discmask3", "discmask4", "stat3"]


def frame_for(target, V, S, T, cohort, ref):
    """One admission-level frame carrying everything the summaries need."""
    tcol = f"tgt_{target}"
    disc_mask_cols = [c for c in S.columns if c.startswith("msk_")]
    s = S.copy()
    s["_n_disc_mask"] = s[disc_mask_cols].sum(axis=1)
    df = (cohort[["subject_id", "gender"]]
          .merge(V, on="subject_id")
          .merge(s, on="subject_id")
          .merge(T[["subject_id", tcol]].dropna(), on="subject_id"))
    df = df[df[f"last_{t3.NAME[PARTNER[target]]}"].notna()]
    return df, df[tcol].astype(int).to_numpy()


def run_target(target, V, S, T, cohort, ref, n_perm=1000, n_jobs=16):
    df, y = frame_for(target, V, S, T, cohort, ref)
    label = t3.CORE.get(target, str(target))
    res = {"target_itemid": int(target), "target_label": label,
           "partner_itemid": int(PARTNER[target]),
           "partner_label": t3.CORE.get(PARTNER[target], "?"),
           "n": int(len(y)), "prevalence": float(y.mean())}

    def cell(axis, vs, ss, sub, vv, sv, partner_tag="primary"):
        yy = sub[f"tgt_{target}"].astype(int).to_numpy()
        d = pb.decompose(vv, sv, yy)
        nl, n_used = pb.permutation_null_budgeted(vv, sv, yy, n_perm=n_perm,
                                                  n_jobs=n_jobs, budget_s=300.0)
        syn = d["atoms_nats"]["syn"]
        return {"axis": axis, "v_scheme": vs, "s_scheme": ss, "partner": partner_tag,
                "n": d["n"], "levels": d["levels"], "converged": d["converged"],
                "syn_nats": syn, "syn_pct": d["atoms_pct_of_joint"]["syn"],
                "syn_debiased": syn - float(nl["syn"].mean()),
                "syn_p": float(((nl["syn"] >= syn).sum() + 1) / (n_used + 1)),
                "n_perm_used": n_used,
                "min_cell": d["min_cell_count"],
                "I_joint": d["mi_plugin_nats"]["I_joint"],
                "atoms_pct": d["atoms_pct_of_joint"],
                "s_shares": d["s_level_shares"], "v_shares": d["v_level_shares"]}, d, nl

    # --- A2/A3 primary -----------------------------------------------------
    v_raw = v_tilde(df, PARTNER[target], ref, "refrange3")
    s_raw = s_tilde(df, "amlab3")
    raw = pb.decompose(v_raw, s_raw, y)
    v, s, cinfo = collapse_for_occupancy(v_raw, s_raw, y)
    primary = pb.decompose(v, s, y)
    primary.update({"v_scheme": "refrange3+collapse", "s_scheme": "amlab3+collapse",
                    "collapse": cinfo,
                    "uncollapsed_min_cell": raw["min_cell_count"],
                    "uncollapsed_levels": raw["levels"],
                    "uncollapsed_atoms_pct": raw["atoms_pct_of_joint"]})
    res["primary"] = primary

    # --- A4 permutation null ----------------------------------------------
    null, n_used = pb.permutation_null_budgeted(v, s, y, n_perm=n_perm, n_jobs=n_jobs,
                                                budget_s=600.0)
    res["permutation"] = pb.summarize_with_null(primary, null)
    res["n_perm_primary"] = n_used

    # --- A5 binning sensitivity, varied one axis at a time -----------------
    grid = []
    for vs in V_SCHEMES:
        for partner, tag in ((PARTNER[target], "primary"), (PARTNER_ALT[target], "alt")):
            if tag == "alt" and vs != "refrange3":
                continue
            sub = df[df[f"last_{t3.NAME[partner]}"].notna()]
            yy = sub[f"tgt_{target}"].astype(int).to_numpy()
            vv, sv, _ = collapse_for_occupancy(v_tilde(sub, partner, ref, vs),
                                               s_tilde(sub, "amlab3"), yy)
            grid.append(cell("V", vs, "amlab3", sub, vv, sv, tag)[0])
    for ss in S_SCHEMES:
        vv, sv, _ = collapse_for_occupancy(v_raw, s_tilde(df, ss), y)
        grid.append(cell("S", "refrange3", ss, df, vv, sv)[0])
    res["a5_grid"] = grid

    # --- A6 gates ----------------------------------------------------------
    # A cell whose summary ended up with one level is not a binning scheme; it
    # is excluded from the stability verdict and reported separately.
    for g in grid:
        g["degenerate"] = bool(g["levels"]["V"] < 2 or g["levels"]["S"] < 2)
    live = [g for g in grid if not g["degenerate"]]
    syn_pcts = [g["syn_pct"] for g in live]
    res["gates"] = {
        "cell_occupancy_ok": bool(primary["min_cell_count"] >= 20),
        "min_cell_count": primary["min_cell_count"],
        "cell_occupancy_ok_uncollapsed": bool(raw["min_cell_count"] >= 20),
        "min_cell_count_uncollapsed": raw["min_cell_count"],
        "s_nondegenerate_ok": bool(max(primary["s_level_shares"]) <= 0.80),
        "s_max_share": float(max(primary["s_level_shares"])),
        "permutation_reported": True,
        "binning_sign_consistent": bool(all(p > 0 for p in syn_pcts)
                                        or all(p <= 0 for p in syn_pcts)),
        "syn_pct_range": [float(np.min(syn_pcts)), float(np.max(syn_pcts))],
        "grid_all_significant": bool(all(g["syn_p"] < 0.05 for g in live)),
        "grid_cell_occupancy_ok": bool(all(g["min_cell"] >= 20 for g in live)),
        "collapse_min_share_used": cinfo["min_share"],
        "n_grid_cells": len(grid),
        "n_grid_degenerate": len(grid) - len(live),
        "solver_converged": bool(primary["converged"]
                                 and all(g["converged"] for g in grid)),
    }
    return res


def main(n_perm=1000, n_jobs=24):
    pb._pool(n_jobs)                 # fork workers before the frames are loaded
    V, S, T, cohort = t3.load()
    ref = load_ref_canon()
    out = {}
    for tgt in TARGETS:
        print(f"--- {t3.CORE[tgt]} ({tgt})", flush=True)
        r = run_target(tgt, V, S, T, cohort, ref, n_perm=n_perm, n_jobs=n_jobs)
        a = r["primary"]["atoms_pct_of_joint"]
        p = r["permutation"]["syn"]
        print(f"    n={r['n']} prev={r['prevalence']:.3f} I_joint={r['primary']['mi_plugin_nats']['I_joint']:.5f}")
        print(f"    red={a['red']:.1f}%  u_val={a['u_val']:.1f}%  u_str={a['u_str']:.1f}%  "
              f"SYN={a['syn']:.1f}%  (p={p['p_value']:.4f}, debiased={p['debiased']:.6f} nats)")
        print(f"    gates: {r['gates']}")
        out[str(tgt)] = r
    path = ROOT / "results" / "track_a.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

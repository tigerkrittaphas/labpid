"""Track B (SECONDARY / robustness) — T7-T13.

Its job is narrow: show that Track A's result is not an artifact of discarding
resolution. Every number here is model-relative and is reported with the probe
family named.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import bqutil
import pid_vinfo as pv
import t3_build_channels as t3
import vinfo

ROOT = bqutil.ROOT
TARGETS = [51265, 51301, 50912, 51222, 51006, 50882]


def _atoms_from_nodes(iv, is_, ij, ps_v, ps_s, ps_j, y):
    """Both redundancy measures, plus a per-sample synergy series for the CI."""
    y = np.asarray(y)
    red_mmi = min(iv, is_)
    ps_red_mmi = ps_v if iv <= is_ else ps_s          # per-sample view of the min

    spec_v = pv.specific_v_information(ps_v, y)
    spec_s = pv.specific_v_information(ps_s, y)
    py = {int(c): float((y == c).mean()) for c in np.unique(y)}
    red_imin = sum(py[c] * min(spec_v[c], spec_s[c]) for c in py)
    # per-sample I_min: within each class, the channel achieving the minimum
    ps_red_imin = np.where(
        np.isin(y, [c for c in py if spec_v[c] <= spec_s[c]]), ps_v, ps_s)

    out = {}
    for name, red, ps_red in (("mmi", red_mmi, ps_red_mmi), ("imin", red_imin, ps_red_imin)):
        out[name] = {
            "red": red, "u_val": iv - red, "u_str": is_ - red,
            "syn": ij - iv - is_ + red,
            "_ps_syn": ps_j - ps_v - ps_s + ps_red,
            "_ps_ustr": ps_s - ps_red,
        }
    return out


def sweep(Xv, Xs, y, groups, n_splits=5, floor_seeds=(0, 1), verbose=True):
    """T7 + T9 + T10 in one pass: every capacity, every channel, with floors."""
    Xj = np.hstack([Xv, Xs])
    blocks = {"val": Xv, "str": Xs, "joint": Xj}
    h_marg = vinfo.h_marginal_binary(y, groups, n_splits=n_splits)

    rows = []
    for hidden, wd in vinfo.PROBE_GRID:
        t0 = time.time()
        node, ps, floor = {}, {}, {}
        for bname, X in blocks.items():
            node[bname], ps[bname] = vinfo.v_information(
                X, y, groups, hidden, wd, n_splits, h_marg=h_marg)
            floor[bname], _ = vinfo.noise_floor(
                X, y, groups, hidden, wd, n_splits, seeds=floor_seeds)
        atoms = _atoms_from_nodes(node["val"], node["str"], node["joint"],
                                  ps["val"], ps["str"], ps["joint"], y)
        syn_thr = pv.synergy_threshold(floor)
        row = {
            "hidden": list(hidden), "wd": wd,
            "I_V": node, "floor": floor,
            "net_joint": node["joint"] - floor["joint"],
            "synergy_threshold": syn_thr,
            "atoms": {k: {kk: vv for kk, vv in a.items() if not kk.startswith("_")}
                      for k, a in atoms.items()},
            "syn_above_floor": {k: bool(a["syn"] > syn_thr) for k, a in atoms.items()},
            "monotone_joint_ok": bool(node["joint"] >= max(node["val"], node["str"]) - 1e-12),
            "channel_above_floor": {k: bool(node[k] > floor[k]) for k in blocks},
            "secs": round(time.time() - t0, 1),
        }
        rows.append(row)
        if verbose:
            a = atoms["mmi"]
            print(f"  {str(hidden):16s} a={wd:<7.0e} "
                  f"I_V(V)={node['val']:+.4f} I_V(S)={node['str']:+.4f} "
                  f"I_V(J)={node['joint']:+.4f} floorJ={floor['joint']:.4f} "
                  f"SYN={a['syn']:+.4f} thr={syn_thr:.4f} "
                  f"mono={'ok' if row['monotone_joint_ok'] else 'FAIL'} ({row['secs']}s)",
                  flush=True)
    best = max(rows, key=lambda r: r["net_joint"])
    return best, rows, h_marg


def run_target(target, n_splits=5, n_boot=2000, do_t13=True):
    V, S, T, C = t3.load()
    Xv, Xs, y, groups, meta, _ = t3.assemble(target, V, S, T, C)
    print(f"--- {meta['target_label']} ({target})  n={meta['n']} "
          f"prev={meta['prevalence']:.3f} dimV={meta['dim_V']} dimS={meta['dim_S']}", flush=True)

    best, rows, h_marg = sweep(Xv, Xs, y, groups, n_splits=n_splits)
    hidden, wd = tuple(best["hidden"]), best["wd"]
    print(f"  T7 winner: hidden={hidden} alpha={wd}", flush=True)

    # T8 — the analytic marginal is the reported H_V(Y); the ablated probe is a
    # convergence diagnostic only.
    h_analytic = float(h_marg.mean())
    zero = np.zeros((len(y), 4))
    h_ablated = float(vinfo.crossfit_nll(zero, y, groups, hidden, wd, n_splits).mean())

    # recompute at the winner to get per-sample series for the CIs
    Xj = np.hstack([Xv, Xs])
    iv, ps_v = vinfo.v_information(Xv, y, groups, hidden, wd, n_splits, h_marg=h_marg)
    is_, ps_s = vinfo.v_information(Xs, y, groups, hidden, wd, n_splits, h_marg=h_marg)
    ij, ps_j = vinfo.v_information(Xj, y, groups, hidden, wd, n_splits, h_marg=h_marg)
    atoms = _atoms_from_nodes(iv, is_, ij, ps_v, ps_s, ps_j, y)

    ci = {
        "I_V_val": vinfo.cluster_bootstrap_ci(ps_v, groups, n_boot),
        "I_V_str": vinfo.cluster_bootstrap_ci(ps_s, groups, n_boot),
        "I_V_joint": vinfo.cluster_bootstrap_ci(ps_j, groups, n_boot),
        "syn_mmi": vinfo.cluster_bootstrap_ci(atoms["mmi"]["_ps_syn"], groups, n_boot),
        "syn_imin": vinfo.cluster_bootstrap_ci(atoms["imin"]["_ps_syn"], groups, n_boot),
        "u_str_mmi": vinfo.cluster_bootstrap_ci(atoms["mmi"]["_ps_ustr"], groups, n_boot),
    }

    res = {
        "meta": meta,
        "H_V_Y_analytic": h_analytic,
        "H_V_Y_ablated_probe": h_ablated,
        "t8_marginal_agreement": abs(h_analytic - h_ablated) / h_analytic,
        "probe_selected": {"hidden": list(hidden), "wd": wd,
                           "net_joint": best["net_joint"]},
        "probe_grid": rows,
        "nodes": {"I_V_val": iv, "I_V_str": is_, "I_V_joint": ij},
        "floors": best["floor"],
        "synergy_threshold": best["synergy_threshold"],
        "atoms": {k: {kk: vv for kk, vv in a.items() if not kk.startswith("_")}
                  for k, a in atoms.items()},
        "atoms_pct_of_joint": {k: {kk: (100.0 * vv / ij if ij else np.nan)
                                   for kk, vv in a.items() if not kk.startswith("_")}
                               for k, a in atoms.items()},
        "ci95": ci,
        "gates": {
            "monotone_joint_ok": bool(ij >= max(iv, is_) - 1e-12),
            "channel_above_floor": {"val": bool(iv > best["floor"]["val"]),
                                    "str": bool(is_ > best["floor"]["str"]),
                                    "joint": bool(ij > best["floor"]["joint"])},
            "syn_above_threshold_mmi": bool(atoms["mmi"]["syn"] > best["synergy_threshold"]),
            "syn_above_threshold_imin": bool(atoms["imin"]["syn"] > best["synergy_threshold"]),
            "nonnegative_atoms_mmi": bool(all(v >= -best["synergy_threshold"]
                                              for k, v in atoms["mmi"].items()
                                              if not k.startswith("_"))),
            "t8_agreement_ok": bool(abs(h_analytic - h_ablated) / h_analytic < 0.05),
        },
    }

    # --- T13 encoding-artifact bound ---------------------------------------
    if do_t13:
        Xv2, Xs2, y2, g2, meta2, _ = t3.assemble(target, V, S, T, C, impute_all=True)
        Xj2 = np.hstack([Xv2, Xs2])
        hm2 = vinfo.h_marginal_binary(y2, g2, n_splits=n_splits)
        iv2, _ = vinfo.v_information(Xv2, y2, g2, hidden, wd, n_splits, h_marg=hm2)
        is2, _ = vinfo.v_information(Xs2, y2, g2, hidden, wd, n_splits, h_marg=hm2)
        ij2, _ = vinfo.v_information(Xj2, y2, g2, hidden, wd, n_splits, h_marg=hm2)
        syn2 = ij2 - iv2 - is2 + min(iv2, is2)
        res["t13"] = {
            "dim_V_imputed": meta2["dim_V"],
            "syn_imputed_mmi": syn2, "syn_core_mmi": atoms["mmi"]["syn"],
            "artifact_upper_bound": syn2 - atoms["mmi"]["syn"],
            "nodes_imputed": {"I_V_val": iv2, "I_V_str": is2, "I_V_joint": ij2},
        }
        print(f"  T13 artifact bound: {res['t13']['artifact_upper_bound']:+.5f} nats", flush=True)

    a = atoms["mmi"]
    print(f"  H_V(Y)={h_analytic:.4f} (ablated {h_ablated:.4f})  "
          f"I_V: V={iv:.4f} S={is_:.4f} J={ij:.4f}", flush=True)
    print(f"  MMI atoms: red={a['red']:.4f} u_val={a['u_val']:.4f} "
          f"u_str={a['u_str']:.4f} SYN={a['syn']:.4f} (thr {best['synergy_threshold']:.4f}) "
          f"CI{ci['syn_mmi']}", flush=True)
    print(f"  gates: {res['gates']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*", default=TARGETS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="track_b.json")
    args = ap.parse_args()

    out = {}
    path = ROOT / "results" / args.out
    path.parent.mkdir(exist_ok=True)
    for tgt in args.targets:
        out[str(tgt)] = run_target(tgt, n_splits=args.folds)
        path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

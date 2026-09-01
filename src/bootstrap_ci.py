"""Nested bootstrap confidence intervals for the PID atoms.

Every `±` reported in this project before now was `np.std` over a handful of
k-means seeds -- a measure of *discretisation instability*, computed with the
patient sample held completely fixed. It says nothing about whether a result
would replicate in another sample of ICU patients, and reviewers will read it
as a standard error, which it is not.

This module produces the actual inferential object. For each replicate:

    resample patients with replacement  ->  REFIT k-means on the resample
    ->  decompose  ->  collect atoms

then take percentile intervals across replicates. Refitting the codebook inside
the replicate is the point: it nests discretisation uncertainty inside sampling
uncertainty, so one interval covers both sources. Holding the codebook fixed
would condition away exactly the uncertainty that turned out to dominate this
project (seed spread reached +-13 percentage points at the original k=7 -- see
`results/k_resolution_headline_impact.json`).

Each row of the channel matrices is one patient (the cohorts are first-stay-only),
so an ordinary bootstrap over rows *is* a patient-level cluster bootstrap. The
`groups` array is checked for uniqueness rather than assumed.

**What this does not fix.** Plug-in mutual information is upward biased, and the
resamples inherit that bias. These intervals quantify *precision*, not
*accuracy* -- they will not touch the 2-10x underestimate against the
Gaussian-copula estimator (`results/pid_gcmi_paired_channels.json`). Use them
for "how reproducible", never for "how correct".

    PYTHONPATH=src .venv/bin/python src/bootstrap_ci.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np

import bqutil
import pid_broja as pb
import quantize as qz

ROOT = bqutil.ROOT
ATOMS = ("red", "u_val", "u_str", "syn")
MEASURES = ("broja", "ccs")

_G: dict = {}   # per-worker globals, populated by _init via fork (never pickled)


def _init(Xv, Xs, y, sentinel, k):
    _G.update(Xv=Xv, Xs=Xs, y=y, sentinel=sentinel, k=k)


def _one(rep: int):
    """One bootstrap replicate: resample patients, refit the codebook, decompose.

    The replicate index doubles as the k-means seed, so no two replicates share
    a clustering -- that is what nests discretisation variance inside the
    interval rather than beside it.
    """
    Xv, Xs, y, sentinel, k = (_G["Xv"], _G["Xs"], _G["y"], _G["sentinel"], _G["k"])
    n = len(y)
    rng = np.random.default_rng(100_000 + rep)
    idx = rng.integers(0, n, n)
    yb = y[idx]
    if len(np.unique(yb)) < 2:                    # degenerate draw, drop it
        return None

    v, _ = qz.kmeans_levels(qz.standardize_for_kmeans(Xv[idx]), k, seed=rep)
    s, _ = qz.kmeans_levels(qz.standardize_for_kmeans(Xs[idx], sentinel=sentinel), k, seed=rep)

    out = {}
    for measure in MEASURES:
        d = pb.decompose(v, s, yb, measure=measure)
        total = d["mi_plugin_nats"]["I_joint"]
        if not d["converged"] or total <= 0:
            out[measure] = None
            continue
        out[measure] = {"I_joint": total,
                        **{a: 100.0 * d["atoms_nats"][a] / total for a in ATOMS}}
    return out


def bootstrap(Xv, Xs, y, groups, sentinel, k=13, n_boot=1000, n_jobs=16):
    """Percentile CIs for every atom, both measures. Returns a summary dict."""
    assert len(np.unique(groups)) == len(groups), (
        "rows must be one-per-patient for the ordinary bootstrap to be a "
        "patient-level cluster bootstrap; got repeated group ids")

    t0 = time.time()
    ctx = mp.get_context("fork")          # fork so the big arrays are shared, not pickled
    with ctx.Pool(n_jobs, initializer=_init, initargs=(Xv, Xs, y, sentinel, k)) as pool:
        reps = pool.map(_one, range(n_boot), chunksize=8)

    summary = {"k": k, "n": int(len(y)), "n_boot_requested": n_boot,
               "runtime_s": time.time() - t0, "measures": {}}
    for measure in MEASURES:
        vals = [r[measure] for r in reps if r is not None and r.get(measure) is not None]
        summary["measures"][measure] = {
            "n_boot_used": len(vals),
            "n_failed": n_boot - len(vals),
            **{a: {"median": float(np.median([v[a] for v in vals])),
                   "ci95": [float(np.percentile([v[a] for v in vals], 2.5)),
                            float(np.percentile([v[a] for v in vals], 97.5))],
                   "boot_sd": float(np.std([v[a] for v in vals]))}
               for a in ATOMS},
            "I_joint": {"median": float(np.median([v["I_joint"] for v in vals])),
                        "ci95": [float(np.percentile([v["I_joint"] for v in vals], 2.5)),
                                 float(np.percentile([v["I_joint"] for v in vals], 97.5))]},
        }
    return summary


def main(n_boot: int = 1000, k: int = 13, n_jobs: int = 16):
    import mimic_channels as mimic
    import eicu_channels as eicu

    MG = 50960
    P12E = {"bicarbonate", "chloride", "creatinine", "glucose", "Hct", "Hgb",
            "platelets x 1000", "potassium", "RBC", "sodium", "BUN", "WBC x 1000"}
    TARGETS = ["mortality", "gender_female", "age_ge65", "race_binary"]

    dbs = [("MIMIC-IV", mimic, mimic.complete_case(itemid_scope=set(mimic.CORE) - {MG})),
           ("eICU-CRD", eicu, eicu.complete_case(labname_scope=P12E))]

    results, t0 = {}, time.time()
    for db, mod, cc in dbs:
        demo, _ = mod.demographic_targets()
        for target in TARGETS:
            ch = cc if target == "mortality" else mod.retarget(cc, demo[target], target)
            r = bootstrap(ch.Xv, ch.Xs, ch.y, ch.groups, mod.TIME_SENTINEL,
                          k=k, n_boot=n_boot, n_jobs=n_jobs)
            results[f"{db}/{target}"] = r
            b = r["measures"]["broja"]
            print(f"[{db}] {target:14s} n={r['n']:7,d}  "
                  f"syn {b['syn']['median']:5.1f} [{b['syn']['ci95'][0]:5.1f},{b['syn']['ci95'][1]:5.1f}]  "
                  f"u_str {b['u_str']['median']:5.2f} [{b['u_str']['ci95'][0]:5.2f},{b['u_str']['ci95'][1]:5.2f}]  "
                  f"({r['n_boot_used'] if 'n_boot_used' in r else b['n_boot_used']} ok, "
                  f"{b['n_failed']} failed, {r['runtime_s']:.0f}s)", flush=True)

    payload = {
        "design": ("Patient-level bootstrap with the k-means codebook REFIT inside each replicate, "
                   "so discretisation uncertainty is nested inside sampling uncertainty and one "
                   "interval covers both. Percentile CIs."),
        "supersedes": ("the `_sd` fields in results/headline_matched_panel12.json and "
                       "results/headline_broja_k13.json, which are SD across 5 k-means seeds with the "
                       "patient sample held fixed -- a discretisation-stability measure, not a CI"),
        "caveat": ("Plug-in MI is upward biased and the resamples inherit that bias. These intervals "
                   "quantify precision, not accuracy; they do not address the 2-10x underestimate "
                   "against the Gaussian-copula estimator."),
        "panel": "matched 12 analytes, 24-d channels on both databases",
        "k": k, "n_boot": n_boot, "total_runtime_s": time.time() - t0,
        "cells": results,
    }
    path = ROOT / "results" / "bootstrap_ci_matched_panel12.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {path}  ({payload['total_runtime_s']/60:.1f} min)")


if __name__ == "__main__":
    main()

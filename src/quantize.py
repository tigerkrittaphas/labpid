"""Vector quantization utilities for turning continuous channels into the
small finite alphabet BROJA-2PID needs.

Two pieces, factored out because they're generic -- they operate on already-
built level/feature arrays and don't know anything about how V/S were
constructed, so they're reusable across channel definitions:

  kmeans_levels(X, k)                    one discrete level per row, fit on
                                          the whole (possibly multi-dim) block.
  collapse_for_occupancy(v, s, y, ...)   BROJA's exponential-cone solver is
                                          unreliable on sparse joint cells;
                                          this merges rare levels by marginal
                                          share until the occupancy gate holds,
                                          escalating only as far as needed.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import pid_broja as pb


def kmeans_levels(X, k, seed=0):
    """One discrete level per row via MiniBatchKMeans. Levels are relabelled
    by cluster size (largest = 0) so level indices are stable across k and
    across seeds -- otherwise "level 0" would mean a different cluster every
    run and nothing downstream (occupancy shares, plots) would line up."""
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=4096, max_iter=300)
    lab = km.fit_predict(X)
    order = np.argsort(-np.bincount(lab, minlength=k))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[lab], km.inertia_


def collapse_rare(levels, min_share=0.02):
    """Merge any level holding less than `min_share` of the mass into its
    nearest surviving neighbour. Never collapse to a single level: a constant
    summary carries no information and would enter a sweep as a spurious
    "synergy = 0" cell rather than a real degenerate case."""
    levels = np.asarray(levels)
    uniq = np.unique(levels)
    shares = {u: float((levels == u).mean()) for u in uniq}
    keep = [u for u in uniq if shares[u] >= min_share]
    if len(keep) < min(2, len(uniq)):
        keep = sorted(sorted(uniq, key=lambda u: -shares[u])[:min(2, len(uniq))])
    remap = {u: min(keep, key=lambda k: abs(int(k) - int(u))) for u in uniq}
    out = np.array([remap[u] for u in levels])
    _, inv = np.unique(out, return_inverse=True)
    return inv, {int(k): int(v) for k, v in remap.items()}


def collapse_for_occupancy(v_raw, s_raw, y, min_cell=20, shares=(0.02, 0.05, 0.10, 0.20)):
    """Collapsing on marginal share alone isn't enough: two merely-uncommon
    levels can still intersect in a starved joint cell. Escalate the merge
    threshold until the occupancy gate passes, and report the threshold that
    was needed -- a heavier collapse is itself a finding, not just a fix."""
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

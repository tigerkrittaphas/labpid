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
from threadpoolctl import threadpool_limits

import pid_broja as pb


def standardize_for_kmeans(X, sentinel=None):
    """Prepare a feature block for kmeans_levels(). K-means minimizes
    Euclidean distance, so unscaled columns are not equally weighted: a
    column with 100x the raw numeric range of another dominates every
    cluster boundary regardless of its actual signal content. Z-scoring
    each column fixes that for an ordinary continuous block.

    A single scalar `sentinel` (e.g. mimic_channels.TIME_SENTINEL) is a
    harder case than an ordinary outlier: it's a real missingness code, not
    noise, but feeding it straight into a Euclidean-distance clustering
    inflates that column's std by 3-10x (see WORKNOTES.md 2026-08-19) and
    the clustering ends up mostly resolving *which columns are missing*
    rather than the real within-range signal. If `sentinel` is given, every
    column's sentinel entries are replaced by that column's real-value mean
    (so they stop acting as multi-hundred-sigma outliers) before z-scoring,
    and one binary "was this value real" column per input column is
    appended -- so missingness is still visible to k-means, just as its own
    bounded feature instead of as an unbounded distance blowout.
    """
    X = np.asarray(X, dtype=np.float64)
    if sentinel is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return (X - mu) / sd

    is_real = X != sentinel
    real_or_nan = np.where(is_real, X, np.nan)
    mu = np.nanmean(real_or_nan, axis=0)
    sd = np.nanstd(real_or_nan, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)              # column with zero real entries
    sd = np.where(np.isnan(sd) | (sd <= 0), 1.0, sd)
    filled = np.where(is_real, X, mu)
    z = (filled - mu) / sd
    return np.hstack([z, is_real.astype(np.float64)])


def kmeans_levels(X, k, seed=0):
    """One discrete level per row via MiniBatchKMeans. Levels are relabelled
    by cluster size (largest = 0) so level indices are stable across k and
    across seeds -- otherwise "level 0" would mean a different cluster every
    run and nothing downstream (occupancy shares, plots) would line up."""
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=4096, max_iter=300)
    # Pin to one thread. MiniBatchKMeans does not merely fail to scale here, it
    # collapses: on this 64,623 x 48 structure block at k=13, measured
    #
    #     1 thread   0.75 s        4 threads  91.3 s        32 threads  >10 min
    #
    # a ~120x SLOWDOWN, because the batch is only 4,096 rows and the per-batch
    # OpenMP fork/join overhead swamps the work -- n_init=10 x max_iter=300 means
    # up to 3,000 of those parallel regions per call. Uncapped, this is what hung
    # notebooks/03_matched_panel12 for 2h38m and 58 CPU-hours in what should be a
    # ~2 min cell, and what put bootstrap_ci (1,000 refits) 2x over its runtime.
    # Verified label-for-label identical to the unpinned result, so this is a
    # pure speed fix -- it changes no number this project reports.
    with threadpool_limits(limits=1):
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

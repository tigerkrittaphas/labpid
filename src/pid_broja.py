"""Track A — discrete Shannon PID via BROJA-2PID on binned channels.

This is the primary track: exact given the discretization, non-negative atoms,
monotone lattice, DPI intact. The discretization *is* the assumption, so A5
sweeps it.
"""
from __future__ import annotations

import numpy as np
import dit
from dit.pid import PID_BROJA

LOG2 = np.log(2.0)


def joint_table(v, s, y):
    """Counts over (Ṽ, S̃, Y) with the observed level sets."""
    v, s, y = np.asarray(v), np.asarray(s), np.asarray(y)
    lv, ls, ly = np.unique(v), np.unique(s), np.unique(y)
    tab = np.zeros((len(lv), len(ls), len(ly)), dtype=np.float64)
    iv = np.searchsorted(lv, v)
    is_ = np.searchsorted(ls, s)
    iy = np.searchsorted(ly, y)
    np.add.at(tab, (iv, is_, iy), 1.0)
    return tab, (lv, ls, ly)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mi_plugin(tab_2d, n):
    p = tab_2d / n
    return _entropy(p.sum(1)) + _entropy(p.sum(0)) - _entropy(p)


def _mi_mm(tab_2d, n):
    """Miller-Madow: Ĥ_MM = Ĥ_plug + (m̂-1)/(2N) applied to each entropy term."""
    mi = _mi_plugin(tab_2d, n)
    mx = (tab_2d.sum(1) > 0).sum()
    my = (tab_2d.sum(0) > 0).sum()
    mxy = (tab_2d > 0).sum()
    return mi + (mx - 1) / (2 * n) + (my - 1) / (2 * n) - (mxy - 1) / (2 * n)


# The exponential-cone program (Makkeh et al. 2018) is the reference BROJA-2PID
# solver. dit's SLSQP fallback silently returns *infeasible* points on some
# tables here -- negative atoms, which BROJA guarantees cannot happen -- and is
# also ~5x slower. Never let the solver fall back silently.
BROJA_METHOD = "cone"
NEG_TOL = 1e-9


def broja_atoms(tab, method=BROJA_METHOD):
    """Four BROJA atoms in nats from a (|Ṽ|, |S̃|, |Y|) count table."""
    n = tab.sum()
    pmf = {}
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            for k in range(tab.shape[2]):
                if tab[i, j, k] > 0:
                    pmf[(i, j, k)] = tab[i, j, k] / n
    d = dit.Distribution(pmf)
    d.set_rv_names("VSY")
    pid = PID_BROJA(d, [["V"], ["S"]], ["Y"], method=method)
    # dit keys lattice nodes by rv name (sorted) and reports bits
    get = lambda k: float(pid.get_pi(k)) * LOG2
    return {
        "red":   get((("S",), ("V",))),
        "u_val": get((("V",),)),
        "u_str": get((("S",),)),
        "syn":   get((("S", "V"),)),
    }


SOLVER_CASCADE = ("cone", "admui", "scipy")


def broja_atoms_checked(tab, methods=SOLVER_CASCADE):
    """`broja_atoms` plus the non-negativity check that detects a solver that
    terminated at an infeasible point.

    A negative atom is proof of failure, not a finding: BROJA atoms are
    non-negative by construction. On failure, fall through to the next solver
    and keep the best (largest minimum atom) attempt, so a single awkward table
    cannot silently poison a sensitivity cell."""
    best, best_method = None, None
    for m in methods:
        try:
            a = broja_atoms(tab, m)
        except Exception:
            continue
        if best is None or min(a.values()) > min(best.values()):
            best, best_method = a, m
        if min(a.values()) >= -NEG_TOL:
            break
    if best is None:
        raise RuntimeError("every BROJA solver failed on this table")
    best = dict(best)
    best["_converged"] = bool(min(v for k, v in best.items()
                                  if not k.startswith("_")) >= -NEG_TOL)
    best["_solver"] = best_method
    return best


def decompose(v, s, y, bias_correct=True):
    """Atoms plus the node MIs, cell occupancy, and the A3 diagnostics."""
    tab, levels = joint_table(v, s, y)
    n = float(tab.sum())
    atoms = broja_atoms_checked(tab)
    converged = atoms.pop("_converged")
    solver_used = atoms.pop("_solver")

    vy = tab.sum(axis=1)                      # (|Ṽ|, |Y|)
    sy = tab.sum(axis=0)                      # (|S̃|, |Y|)
    jy = tab.reshape(-1, tab.shape[2])        # ((|Ṽ|·|S̃|), |Y|)

    mi = {
        "I_val":   _mi_plugin(vy, n),
        "I_str":   _mi_plugin(sy, n),
        "I_joint": _mi_plugin(jy, n),
    }
    mi_mm = {
        "I_val":   _mi_mm(vy, n),
        "I_str":   _mi_mm(sy, n),
        "I_joint": _mi_mm(jy, n),
    }
    total = mi["I_joint"]
    out = {
        "n": int(n),
        "solver": solver_used,
        "converged": converged,
        "levels": {"V": len(levels[0]), "S": len(levels[1]), "Y": len(levels[2])},
        "atoms_nats": atoms,
        "atoms_pct_of_joint": {k: (100.0 * v_ / total if total > 0 else np.nan)
                               for k, v_ in atoms.items()},
        "mi_plugin_nats": mi,
        "mi_miller_madow_nats": mi_mm,
        "min_cell_count": float(tab.min()),
        "n_cells": int(tab.size),
        "n_empty_cells": int((tab == 0).sum()),
        "cell_counts": tab.astype(int).tolist(),
        "s_level_shares": (tab.sum(axis=(0, 2)) / n).tolist(),
        "v_level_shares": (tab.sum(axis=(1, 2)) / n).tolist(),
    }
    return out


_POOL = None


def _pool(n_jobs):
    """One persistent pool, forked before the large frames are loaded."""
    global _POOL
    if _POOL is None:
        from multiprocessing import get_context
        _POOL = get_context("fork").Pool(n_jobs)
    return _POOL


def _perm_batch(args):
    """Shuffling Y against (Ṽ, S̃) holds both the (v,s) cell counts and the Y
    marginal fixed, so the permutation distribution of the table is exactly
    multivariate hypergeometric over the cells. Sampling it directly means a
    worker receives 9 numbers instead of three 56k-element vectors."""
    cell_counts, n_pos, shape, seed, k = args
    rng = np.random.default_rng(seed)
    cell_counts = np.asarray(cell_counts, dtype=np.int64)
    out = []
    draws = rng.multivariate_hypergeometric(cell_counts, n_pos, size=k)
    for d in draws:
        tab = np.stack([cell_counts - d, d], axis=-1).reshape(*shape, 2).astype(float)
        # same estimator as the observed value, or the p-value compares a
        # cascade-derived statistic against a cone-only null
        a = broja_atoms_checked(tab)
        out.append({k: v for k, v in a.items() if not k.startswith("_")})
    return out


def permutation_null(v, s, y, n_perm=1000, seed=0, n_jobs=16, batch=25):
    """A4 — shuffle Y against (Ṽ, S̃) and recompute every atom. A four-term
    difference reliably produces nonzero values under the null, so an atom
    without this number means nothing."""
    tab, _ = joint_table(v, s, y)
    shape = tab.shape[:2]
    cell_counts = tab.sum(axis=2).astype(np.int64).ravel()
    n_pos = int(tab[..., 1].sum()) if tab.shape[2] > 1 else 0

    n_batches = int(np.ceil(n_perm / batch))
    args = [(cell_counts, n_pos, shape, seed * 100_000 + i, batch) for i in range(n_batches)]
    res = [r for chunk in _pool(n_jobs).map(_perm_batch, args) for r in chunk][:n_perm]
    keys = ["red", "u_val", "u_str", "syn"]
    return {k: np.array([r[k] for r in res]) for k in keys}


def permutation_null_budgeted(v, s, y, n_perm=1000, seed=0, n_jobs=16, budget_s=90.0):
    """As `permutation_null`, but probe the cost first and shrink n_perm to fit a
    wall-clock budget. BROJA's optimiser is far slower on some tables than
    others (a 4-level structural summary can cost 100x a 3-level one), so a
    fixed count would either stall the sweep or under-resource the primary.
    The realised count is returned so it can be reported per cell."""
    import time
    t0 = time.time()
    probe = permutation_null(v, s, y, n_perm=n_jobs, seed=seed + 7717, n_jobs=n_jobs, batch=1)
    per = max((time.time() - t0) / n_jobs, 1e-6)
    remaining = max(0.0, budget_s - (time.time() - t0))
    n_more = int(np.clip(remaining / per, 0, n_perm - n_jobs))
    if n_more <= 0:
        return probe, len(probe["syn"])
    rest = permutation_null(v, s, y, n_perm=n_more, seed=seed, n_jobs=n_jobs,
                            batch=max(1, min(25, n_more // n_jobs or 1)))
    out = {k: np.concatenate([probe[k], rest[k]]) for k in probe}
    return out, len(out["syn"])


def summarize_with_null(obs, null):
    out = {}
    for k, arr in null.items():
        o = obs["atoms_nats"][k]
        out[k] = {
            "observed": o,
            "null_mean": float(arr.mean()),
            "null_p975": float(np.quantile(arr, 0.975)),
            "debiased": float(o - arr.mean()),
            "p_value": float(((arr >= o).sum() + 1) / (len(arr) + 1)),
        }
    return out

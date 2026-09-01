"""Gaussian-copula PID (Ince et al. 2017) on the current paired V/S channels.

Ported from `archive/v1_engineered_channels/src/gcmi_pid.py`, which ran the same
estimator against the retired redaction-rule channels. The estimator core
(`copnorm`, `ent_g`, `mi_gd`, `shuffle_floor`, `cluster_bootstrap`, `decompose`)
is carried over unchanged -- only the channel plumbing is new.

**Why this estimator, now.** The BROJA/CCS pipeline reaches a discrete table via
k-means, and k turns out to be doing far more work than intended: the value
channel has no cluster structure to find (its silhouette is indistinguishable
from a Gaussian null of identical covariance), the atoms do not converge until
k>=10, and `collapse_for_occupancy` destabilises them further -- see WORKNOTES.md
2026-08-22 and `results/k_resolution_headline_impact.json`. GCMI has **no
tunable knobs at all**: no k, no seed, no probe capacity, no occupancy gate. Each
marginal is rank-transformed to a standard normal (the copula step, invariant to
any monotone reparameterisation of a feature) and MI is then closed-form
Gaussian with an analytic bias correction. That makes it the cleanest available
check on whether the discretised result is real or an artifact of one
quantisation family.

Three consequences that set how to read the output:

  * GCMI is a **lower bound** on the true MI. It sees only the Gaussian part of
    the copula, so monotone dependence is captured and non-monotone structure is
    not. It is biased toward the null: synergy that survives here is hard to
    dismiss, absence of synergy here is weak evidence of absence.

  * Redundancy is **MMI** (min of the two node MIs), matching `pid_vinfo` so the
    atoms are directly comparable. For Gaussian variables MMI coincides with
    BROJA, I_min and CCS (Barrett 2015, PRE 91:052802) -- which is what makes
    this a bridge between the two tracks, but also means it **cannot**
    distinguish redundancy definitions the way the discrete pipeline can. One
    answer, not four.

  * The joint channel here is 52-dimensional (26 value + 26 structure) against
    the 205 dimensions this code was originally written for. That is a large
    improvement in `n >> d`, but NOT enough to rescue the entropy-difference
    form: measured on MIMIC-IV mortality it still returns I(joint)=0.506 nats
    against H(Y)=0.343, which is impossible. `mi_gd_entropy` therefore remains a
    documented failure rather than a cross-check, and `mi_gd` -- which never
    forms a d-dimensional entropy -- remains the only usable estimator here.

    PYTHONPATH=src .venv/bin/python src/gcmi_pid.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.special import ndtri, psi
from scipy.stats import rankdata

import bqutil

ROOT = bqutil.ROOT
RIDGE = 1e-9   # ent_g only; mi_gd uses its own, much larger, ridge
EPS = 1e-12


def copnorm(X: np.ndarray) -> np.ndarray:
    """Rank-transform each column to a standard normal. X is (n, d)."""
    n = X.shape[0]
    r = np.apply_along_axis(rankdata, 0, X)      # average ranks handle ties
    return ndtri(r / (n + 1.0))


def ent_g(X: np.ndarray, biascorrect: bool = True) -> float:
    """Entropy in nats of an (n, d) Gaussian, with the Ince analytic correction."""
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / (n - 1)
    C.flat[:: d + 1] += RIDGE * np.trace(C) / d
    chC = np.linalg.cholesky(C)
    H = np.log(np.diagonal(chC)).sum() + 0.5 * d * np.log(2 * np.pi * np.e)
    if biascorrect:
        # the log-determinant of a sample covariance is biased; this is the exact
        # expectation correction for a Gaussian, not an asymptotic one
        psiterms = psi((n - np.arange(1, d + 1)) / 2.0) / 2.0
        dterm = (np.log(2.0) - np.log(n - 1.0)) / 2.0
        H = H - d * dterm - psiterms.sum()
    return float(H)


def mi_gd_entropy(Xc: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) as H(X) - E_y H(X|y).

    At the 205 dimensions this code was written for, the two entropies were each
    ~45 nats and their difference had to be <= H(Y) ~ 0.34, so a 1% relative
    error in either swamped the effect and this returned MI above H(Y).
    Dropping to 52 dimensions does not fix it: on MIMIC-IV mortality it returns
    I(joint)=0.506 against H(Y)=0.343, still impossible. Kept only to document
    that, and to make the check cheap to repeat on any new channel. Use `mi_gd`.
    """
    n = len(y)
    return ent_g(Xc) - sum((y == c).sum() / n * ent_g(Xc[y == c]) for c in np.unique(y))


def mi_gd(Xc: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> float:
    """I(X;Y) in nats for copula-normalised X and BINARY y, via the discriminant.

    Under a shared-covariance Gaussian model the posterior P(y|x) depends on x
    only through the scalar score w'x with w = S^-1 (mu1 - mu0). So the whole
    decomposition collapses to a one-dimensional integral and no d-dimensional
    entropy is ever formed -- which is what removes the cancellation.

    The score is Gaussian within each class with common variance D2 and a mean
    separation also equal to D2 (the Mahalanobis distance), so a single number
    determines the answer. Shrinkage handles any rank deficiency.
    """
    y = np.asarray(y)
    cls = np.unique(y)
    assert len(cls) == 2, "discriminant form is for binary targets"
    n = len(y)
    n0, n1 = int((y == cls[0]).sum()), int((y == cls[1]).sum())
    d = Xc.shape[1]

    m0, m1 = Xc[y == cls[0]].mean(0), Xc[y == cls[1]].mean(0)
    Z = np.vstack([Xc[y == cls[0]] - m0, Xc[y == cls[1]] - m1])
    S = (Z.T @ Z) / (n - 2)
    S.flat[:: d + 1] += ridge * np.trace(S) / d       # ridge, not a token epsilon

    delta = m1 - m0
    w = np.linalg.solve(S, delta)
    D2 = float(delta @ w)

    # Lachenbruch bias correction: the plug-in Mahalanobis distance is inflated
    # by roughly d*(1/n0 + 1/n1) even when the true separation is zero.
    D2 = max(((n - d - 3) / (n - 2)) * D2 - d * (1.0 / n0 + 1.0 / n1), 0.0)

    # I(X;Y) = H(Y) - E_s H(Y|s), by Gauss-Hermite over the score distribution
    p1 = n1 / n
    prior = np.log(p1 / (1 - p1))
    Hy = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    if D2 <= 0:
        return 0.0
    sd = np.sqrt(D2)
    nodes, wts = np.polynomial.hermite_e.hermegauss(129)
    wts = wts / wts.sum()
    Hcond = 0.0
    for c, pc, mu in ((1, p1, D2 / 2), (0, 1 - p1, -D2 / 2)):
        lo = prior + (nodes * sd + mu)            # log-odds at each quadrature node
        q = 1.0 / (1.0 + np.exp(-np.clip(lo, -50, 50)))
        h = -(q * np.log(np.clip(q, EPS, 1)) + (1 - q) * np.log(np.clip(1 - q, EPS, 1)))
        Hcond += pc * float(wts @ h)
    return float(Hy - Hcond)


def shuffle_floor(Xc, y, n_shuf=5, seed=0) -> float:
    """Same construction as vinfo.noise_floor: permute labels, take max |MI|."""
    rng = np.random.default_rng(1000 + seed)
    return float(np.max([abs(mi_gd(Xc, rng.permutation(y))) for _ in range(n_shuf)]))


def cluster_bootstrap(Xcv, Xcs, y, groups, n_boot=400, seed=0):
    """Resample patients with replacement; recompute the whole decomposition."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    bounds = np.r_[starts, len(gs)]
    n_g = len(starts)

    Xcj = np.hstack([Xcv, Xcs])
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idx = order[np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in pick])] \
            if (bounds[1:] - bounds[:-1]).max() > 1 else order[pick]
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        iv, is_, ij = mi_gd(Xcv[idx], yb), mi_gd(Xcs[idx], yb), mi_gd(Xcj[idx], yb)
        out.append(ij - max(iv, is_))
    return np.asarray(out)


def decompose(Xv, Xs, y, groups, n_boot=400, cross_check=False):
    Xcv, Xcs = copnorm(Xv), copnorm(Xs)
    Xcj = np.hstack([Xcv, Xcs])

    iv, is_, ij = mi_gd(Xcv, y), mi_gd(Xcs, y), mi_gd(Xcj, y)
    red = min(iv, is_)
    atoms = {"red": red, "u_val": iv - red, "u_str": is_ - red,
             "syn": ij - iv - is_ + red}

    floors = {"val": shuffle_floor(Xcv, y), "str": shuffle_floor(Xcs, y),
              "joint": shuffle_floor(Xcj, y)}
    thr = floors["val"] + floors["str"] + floors["joint"]

    boot = cluster_bootstrap(Xcv, Xcs, y, groups, n_boot=n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    p = float(np.mean(y))
    Hy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    out = {
        "n": int(len(y)), "prevalence": p, "H_Y": Hy,
        "dim_V": int(Xv.shape[1]), "dim_S": int(Xs.shape[1]),
        "nodes": {"I_val": iv, "I_str": is_, "I_joint": ij},
        "I_joint_pct_of_H": 100.0 * ij / Hy,
        "atoms_mmi": atoms,
        "atoms_pct_of_joint": {k: 100 * v / ij for k, v in atoms.items()} if ij > 0 else {},
        "floors": floors, "synergy_threshold": thr,
        "syn_ci95": [float(lo), float(hi)],
        "clears_threshold": bool(lo > thr),
        "monotone_joint_ok": bool(ij >= max(iv, is_) - 1e-12),
        # MMI sets redundancy = min(I_val, I_str), so whichever node is smaller
        # gets a unique atom of exactly 0. That zero is definitional, not a
        # finding -- unlike BROJA's, which can be non-zero for either node.
        "mmi_forced_zero_atom": "u_str" if is_ <= iv else "u_val",
        "n_boot_used": int(len(boot)),
    }
    if cross_check:
        # KNOWN to violate I <= H(Y) at these dimensions; off by default, see
        # mi_gd_entropy's docstring. Retained so the failure stays reproducible.
        out["nodes_entropy_form_KNOWN_BROKEN"] = {
            "I_val": mi_gd_entropy(Xcv, y), "I_str": mi_gd_entropy(Xcs, y),
            "I_joint": mi_gd_entropy(Xcj, y)}
    return out


def _targets(mod, cc):
    """(name, Channels) for mortality plus the three demographic probes."""
    demo, _ = mod.demographic_targets()
    yield "mortality", cc
    for t in ("gender_female", "age_ge65", "race_binary"):
        yield t, mod.retarget(cc, demo[t], t)


def main(n_boot: int = 400):
    import mimic_channels as mimic
    import eicu_channels as eicu

    t0 = time.time()
    results = {}
    for db, mod in (("MIMIC-IV", mimic), ("eICU-CRD", eicu)):
        cc = mod.complete_case()
        print(f"\n=== {db} === {cc}", flush=True)
        for name, ch in _targets(mod, cc):
            t = time.time()
            r = decompose(ch.Xv, ch.Xs, ch.y, ch.groups, n_boot=n_boot)
            n_, a = r["nodes"], r["atoms_mmi"]
            print(f"  {name:14s} n={r['n']:7,d}  I(V)={n_['I_val']:.5f} I(S)={n_['I_str']:.5f} "
                  f"I(joint)={n_['I_joint']:.5f} ({r['I_joint_pct_of_H']:.3f}% of H)", flush=True)
            print(f"  {'':14s} red={a['red']:.5f} u_val={a['u_val']:.5f} "
                  f"u_str={a['u_str']:.5f} syn={a['syn']:.5f}", flush=True)
            if r["atoms_pct_of_joint"]:
                pc = r["atoms_pct_of_joint"]
                print(f"  {'':14s} %: red={pc['red']:.1f} u_val={pc['u_val']:.1f} "
                      f"u_str={pc['u_str']:.1f} syn={pc['syn']:.1f}", flush=True)
            print(f"  {'':14s} syn threshold={r['synergy_threshold']:.5f} "
                  f"CI95=[{r['syn_ci95'][0]:.5f},{r['syn_ci95'][1]:.5f}] "
                  f"clears={r['clears_threshold']}  monotone={r['monotone_joint_ok']}", flush=True)
            print(f"  {'':14s} ({time.time()-t:.0f}s)", flush=True)
            results[f"{db}/{name}"] = r

    payload = {
        "design": ("Gaussian-copula MI (Ince 2017) on the paired first/last V/S channels, "
                   "MMI redundancy. No k, no seed, no probe capacity, no occupancy gate."),
        "caveats": ["GCMI is a lower bound on true MI (Gaussian copula only) -- biased toward the null.",
                    "MMI redundancy coincides with BROJA/I_min/CCS for Gaussians (Barrett 2015), "
                    "so this cannot distinguish redundancy definitions."],
        "n_boot": n_boot, "runtime_s": time.time() - t0, "results": results,
    }
    path = ROOT / "results" / "pid_gcmi_paired_channels.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {path}  ({payload['runtime_s']/60:.1f} min)")


if __name__ == "__main__":
    main()

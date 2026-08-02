"""Track B — PID over two channels under V-information.

V-information supplies the three node values but not a redundancy function, and
PID needs one to close the system (two constraints, four unknowns). Both the
standard choices are computed and reported; their spread is the
redundancy-choice uncertainty.

V-PID loses non-negativity and monotonicity, so `monotone_joint_ok` must be
checked before any atom is read.
"""
from __future__ import annotations

import numpy as np

import vinfo


def specific_v_information(per_sample, y):
    """DeWeese-Meister specific information I_V(X -> Y=y), estimated as the
    mean per-sample log-likelihood ratio within each class."""
    y = np.asarray(y)
    return {int(c): float(per_sample[y == c].mean()) for c in np.unique(y)}


def pid_two_channels(Xv, Xs, y, groups, hidden=(128, 64), wd=1e-3, n_splits=5,
                     seed=0, h_marg=None):
    """Node values plus the four atoms under both redundancy measures."""
    y = np.asarray(y)
    if h_marg is None:
        h_marg = vinfo.h_marginal_binary(y, groups, n_splits=n_splits, seed=seed)

    Xj = np.hstack([np.asarray(Xv), np.asarray(Xs)])
    iv, ps_v = vinfo.v_information(Xv, y, groups, hidden, wd, n_splits, seed, h_marg)
    is_, ps_s = vinfo.v_information(Xs, y, groups, hidden, wd, n_splits, seed, h_marg)
    ij, ps_j = vinfo.v_information(Xj, y, groups, hidden, wd, n_splits, seed, h_marg)

    # --- MMI redundancy: crude, robust, exact for Gaussians -----------------
    red_mmi = min(iv, is_)

    # --- I_min (Williams & Beer) -------------------------------------------
    spec_v = specific_v_information(ps_v, y)
    spec_s = specific_v_information(ps_s, y)
    py = {int(c): float((y == c).mean()) for c in np.unique(y)}
    red_imin = sum(py[c] * min(spec_v[c], spec_s[c]) for c in py)

    atoms = {}
    for name, red in (("mmi", red_mmi), ("imin", red_imin)):
        atoms[name] = {
            "red": red,
            "u_val": iv - red,
            "u_str": is_ - red,
            "syn": ij - iv - is_ + red,
        }

    return {
        "I_V_val": iv, "I_V_str": is_, "I_V_joint": ij,
        "H_V_Y": float(h_marg.mean()),
        "atoms": atoms,
        "specific": {"val": spec_v, "str": spec_s, "p_y": py},
        "monotone_joint_ok": bool(ij >= max(iv, is_) - 1e-12),
        "per_sample": {"val": ps_v, "str": ps_s, "joint": ps_j, "marg": h_marg},
    }


def synergy_threshold(floors: dict) -> float:
    """Synergy is a four-term difference, so its floor is about the sum of the
    per-node floors."""
    return float(floors["val"] + floors["str"] + floors["joint"])

"""Validation suite for the estimators, run against known-answer constructions.

These are correctness checks on the machinery, not on MIMIC. Run before trusting
any number the pipeline produces:

    PYTHONPATH=src .venv/bin/python src/test_vinfo.py
"""
from __future__ import annotations

import numpy as np

import pid_broja as pb
import pid_vinfo as pv
import vinfo

LN2 = np.log(2.0)
PASS, FAIL = "PASS", "**FAIL**"
results = []


def check(name, ok, detail):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL:>8s}] {name:44s} {detail}")
    return ok


# ---------------------------------------------------------------- discrete PID
def test_broja_xor():
    """XOR: all information is synergistic. Red = U_v = U_s = 0, Syn = ln 2."""
    v = np.array([0, 0, 1, 1] * 5000)
    s = np.array([0, 1, 0, 1] * 5000)
    y = v ^ s
    a = pb.decompose(v, s, y)["atoms_nats"]
    check("BROJA XOR: synergy = ln2", abs(a["syn"] - LN2) < 1e-3, f"syn={a['syn']:.6f}")
    check("BROJA XOR: other atoms = 0",
          max(abs(a["red"]), abs(a["u_val"]), abs(a["u_str"])) < 1e-3,
          f"red={a['red']:.2e} u_val={a['u_val']:.2e} u_str={a['u_str']:.2e}")


def test_broja_copy():
    """Y = V: all information is unique to V."""
    rng = np.random.default_rng(0)
    v = rng.integers(0, 2, 20000)
    s = rng.integers(0, 2, 20000)
    a = pb.decompose(v, s, v)["atoms_nats"]
    check("BROJA copy: u_val = ln2", abs(a["u_val"] - LN2) < 1e-2, f"u_val={a['u_val']:.6f}")
    check("BROJA copy: syn & u_str = 0", abs(a["syn"]) < 1e-2 and abs(a["u_str"]) < 1e-2,
          f"syn={a['syn']:.2e} u_str={a['u_str']:.2e}")


def test_broja_redundant():
    """Y = V = S: the two channels are perfectly redundant."""
    rng = np.random.default_rng(1)
    v = rng.integers(0, 2, 20000)
    a = pb.decompose(v, v.copy(), v)["atoms_nats"]
    check("BROJA redundant: red = ln2", abs(a["red"] - LN2) < 1e-2, f"red={a['red']:.6f}")
    check("BROJA redundant: unique & syn = 0",
          max(abs(a["u_val"]), abs(a["u_str"]), abs(a["syn"])) < 1e-2,
          f"u_val={a['u_val']:.2e} syn={a['syn']:.2e}")


def test_broja_nonnegative_and_sums():
    """Atoms are non-negative and sum to the joint MI on random tables."""
    rng = np.random.default_rng(2)
    worst_neg, worst_sum = 0.0, 0.0
    for _ in range(25):
        v = rng.integers(0, 3, 4000)
        s = rng.integers(0, 3, 4000)
        p = 0.2 + 0.2 * (v == 1) + 0.2 * ((v == 2) & (s == 0))
        y = (rng.random(4000) < p).astype(int)
        d = pb.decompose(v, s, y)
        a = d["atoms_nats"]
        worst_neg = min(worst_neg, min(a.values()))
        worst_sum = max(worst_sum, abs(sum(a.values()) - d["mi_plugin_nats"]["I_joint"]))
    check("BROJA non-negativity (25 random tables)", worst_neg >= -pb.NEG_TOL,
          f"worst atom = {worst_neg:.2e}")
    check("BROJA atoms sum to I_joint", worst_sum < 1e-8, f"worst abs deviation = {worst_sum:.2e}")


def test_permutation_null_calibration():
    """Under independence the permutation p-value is uniform, so it exceeds 0.05
    about 95% of the time."""
    rng = np.random.default_rng(3)
    ps = []
    for _ in range(20):
        v = rng.integers(0, 3, 3000)
        s = rng.integers(0, 3, 3000)
        y = rng.integers(0, 2, 3000)
        d = pb.decompose(v, s, y)
        nl = pb.permutation_null(v, s, y, n_perm=200, n_jobs=8)
        ps.append(((nl["syn"] >= d["atoms_nats"]["syn"]).sum() + 1) / 201)
    frac = float(np.mean(np.array(ps) < 0.05))
    check("permutation null calibrated under H0", frac <= 0.20,
          f"{frac:.0%} of 20 independent draws had p<0.05 (expect ~5%)")


# --------------------------------------------------------------- V-information
def test_h_marginal():
    """The analytic cross-fitted base rate matches the plug-in entropy."""
    rng = np.random.default_rng(4)
    for p in (0.02, 0.2, 0.5):
        y = (rng.random(20000) < p).astype(int)
        g = np.arange(len(y))
        h = vinfo.h_marginal_binary(y, g).mean()
        pe = y.mean()
        true = -(pe * np.log(pe) + (1 - pe) * np.log(1 - pe))
        check(f"H_V(Y) analytic at p={p}", abs(h - true) < 5e-3,
              f"{h:.5f} vs {true:.5f}")


def test_vinfo_independent():
    """Pure noise carries no information: I_V sits at its own shuffle floor."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6000, 12))
    y = (rng.random(6000) < 0.4).astype(int)
    g = np.arange(len(y))
    iv, _ = vinfo.v_information(X, y, g, (32,), 1e-1)
    fl, _ = vinfo.noise_floor(X, y, g, (32,), 1e-1, seeds=(0, 1))
    check("I_V(noise) below its shuffle floor", iv <= fl + 1e-3,
          f"I_V={iv:+.5f} floor={fl:.5f}")


def test_vinfo_recovers_signal():
    """A planted logistic signal is recovered near its Bayes-optimal value."""
    rng = np.random.default_rng(6)
    n = 20000
    X = rng.normal(size=(n, 5))
    logit = 1.4 * X[:, 0] - 0.9 * X[:, 1]
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    g = np.arange(n)
    h_bayes = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    h_marg = vinfo.h_marginal_binary(y, g).mean()
    true_iv = h_marg - h_bayes
    iv, _ = vinfo.v_information(X, y, g, (32,), 1e-2)
    check("I_V recovers a planted logistic signal", abs(iv - true_iv) < 0.02,
          f"I_V={iv:.5f} vs Bayes {true_iv:.5f}")


def test_vpid_xor_continuous():
    """Continuous XOR: the joint is fully informative, each channel alone is not,
    so V-PID should report synergy near the whole of I_V(joint)."""
    rng = np.random.default_rng(7)
    n = 24000
    v = rng.integers(0, 2, n)
    s = rng.integers(0, 2, n)
    y = v ^ s
    Xv = v[:, None] + 0.05 * rng.normal(size=(n, 1))
    Xs = s[:, None] + 0.05 * rng.normal(size=(n, 1))
    g = np.arange(n)
    out = pv.pid_two_channels(Xv, Xs, y, g, (64, 32), 1e-3)
    a = out["atoms"]["mmi"]
    check("V-PID XOR: monotonicity gate holds", out["monotone_joint_ok"],
          f"joint={out['I_V_joint']:.4f} val={out['I_V_val']:.4f} str={out['I_V_str']:.4f}")
    check("V-PID XOR: synergy is most of the joint",
          a["syn"] > 0.8 * out["I_V_joint"],
          f"syn={a['syn']:.4f} of joint {out['I_V_joint']:.4f}")


def test_bootstrap_ci():
    """The cluster bootstrap brackets the mean and narrows with n."""
    rng = np.random.default_rng(8)
    x = rng.normal(0.05, 1.0, 8000)
    g = np.repeat(np.arange(4000), 2)
    lo, hi = vinfo.cluster_bootstrap_ci(x, g, n_boot=800)
    check("cluster bootstrap brackets the mean", lo < x.mean() < hi,
          f"[{lo:.4f}, {hi:.4f}] around {x.mean():.4f}")


def main():
    print("\n=== discrete PID (Track A machinery) ===")
    test_broja_xor()
    test_broja_copy()
    test_broja_redundant()
    test_broja_nonnegative_and_sums()
    test_permutation_null_calibration()
    print("\n=== V-information (Track B machinery) ===")
    test_h_marginal()
    test_vinfo_independent()
    test_vinfo_recovers_signal()
    test_vpid_xor_continuous()
    test_bootstrap_ci()

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    lines = ["# Estimator validation suite\n",
             "| check | status | detail |", "|---|---|---|"]
    lines += [f"| {n} | {s} | {d} |" for n, s, d in results]
    from pathlib import Path
    Path(__file__).resolve().parent.parent.joinpath(
        "results", "VALIDATION.md").write_text("\n".join(lines) + "\n")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)

"""Validation suite for t3_paired_channels.retarget() -- the helper that lets
demographic (or any other) targets reuse an already-built arm's Xv/Xs without
rebuilding the wide table. Needs the local BigQuery cache (out/*.parquet);
does not touch demographic data, so it doesn't depend on config/race_bucket_map.json
or a live cohort_demo pull.

    PYTHONPATH=src .venv/bin/python src/test_t3_paired_channels.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import t3_paired_channels as t3

PASS, FAIL = "PASS", "**FAIL**"
results = []


def check(name, ok, detail):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL:>8s}] {name:44s} {detail}")
    return ok


def test_retarget_noop():
    """Feeding an arm's own target back through retarget() must be a true
    no-op: same n, zero drops, byte-identical Xv/y/groups."""
    cc = t3.complete_case()
    y_series = pd.Series(cc.y.astype(float), index=cc.groups)
    rt = t3.retarget(cc, y_series, "hospital_expire_flag")

    check("no-op: n unchanged", rt.meta["n"] == cc.meta["n"],
          f"rt.n={rt.meta['n']} cc.n={cc.meta['n']}")
    check("no-op: zero drops", rt.meta["n_dropped_for_target"] == 0,
          f"n_dropped_for_target={rt.meta['n_dropped_for_target']}")
    check("no-op: Xv identical", np.array_equal(rt.Xv, cc.Xv), "")
    check("no-op: Xs identical", np.array_equal(rt.Xs, cc.Xs), "")
    check("no-op: y identical", np.array_equal(rt.y, cc.y), "")
    check("no-op: groups identical", np.array_equal(rt.groups, cc.groups), "")
    return cc


def test_retarget_with_nulls(cc):
    """A target with real nulls: retained rows must be exactly the rows
    where the synthetic target is non-null, Xv/Xs bit-identical on those
    rows (recomputed independently of retarget()'s own internals), and
    dropped patients must not leak into the retained set."""
    rng = np.random.default_rng(0)
    y_raw = rng.integers(0, 2, len(cc.groups)).astype(float)
    drop_mask = rng.random(len(cc.groups)) < 0.15   # ~15% synthetic nulls
    y_raw[drop_mask] = np.nan
    y_series = pd.Series(y_raw, index=cc.groups)

    rt = t3.retarget(cc, y_series, "synthetic_target")

    keep_expected = ~drop_mask
    check("nulls: retained count matches expected keep-mask",
          rt.meta["n"] == int(keep_expected.sum()),
          f"rt.n={rt.meta['n']} expected={int(keep_expected.sum())}")
    check("nulls: Xv bit-identical on retained rows",
          np.array_equal(rt.Xv, cc.Xv[keep_expected]), "")
    check("nulls: Xs bit-identical on retained rows",
          np.array_equal(rt.Xs, cc.Xs[keep_expected]), "")
    check("nulls: groups bit-identical on retained rows",
          np.array_equal(rt.groups, cc.groups[keep_expected]), "")
    check("nulls: y matches the non-null synthetic target",
          np.array_equal(rt.y, y_raw[keep_expected].astype(int)), "")

    dropped_groups = set(cc.groups[drop_mask])
    check("nulls: no leakage -- dropped patients absent from retained set",
          set(rt.groups).isdisjoint(dropped_groups), "")
    check("nulls: no leakage -- retained set is exactly the kept patients",
          set(rt.groups) == set(cc.groups[keep_expected]), "")
    check("nulls: base_arm provenance preserved",
          rt.meta["base_arm"] == cc.meta["arm"] and rt.meta["base_arm_n"] == cc.meta["n"],
          f"base_arm={rt.meta['base_arm']} base_arm_n={rt.meta['base_arm_n']}")


def main():
    print("\n=== t3_paired_channels.retarget() ===")
    cc = test_retarget_noop()
    test_retarget_with_nulls(cc)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)

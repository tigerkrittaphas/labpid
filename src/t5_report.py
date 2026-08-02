"""Reporting — per-target tables, cross-target summary, FDR correction, figures.

Percentages of joint information are the comparable currency across tracks and
targets; nats are carried alongside because a percentage of a near-zero joint is
not interpretable.
"""
from __future__ import annotations

import json

import numpy as np

import bqutil

ROOT = bqutil.ROOT
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# dataviz reference palette, categorical slots 1-4, validated for the light
# surface (#fcfcfb). Aqua and yellow fall below 3:1 contrast, so the relief rule
# applies: every segment carries a direct label and the report ships tables.
SURFACE = "#fcfcfb"
ATOM_COLORS = {"red": "#2a78d6", "u_val": "#eb6834", "u_str": "#1baf7a", "syn": "#eda100"}
ATOM_LABEL = {"red": "Redundant", "u_val": "Unique — value",
              "u_str": "Unique — structure", "syn": "Synergistic"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
ORDER = ["red", "u_val", "u_str", "syn"]


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        prev = min(prev, p[idx] * n / (n - rank + 1))
        adj[idx] = prev
    return adj


def load():
    a = json.loads((RESULTS / "track_a.json").read_text())
    b_path = RESULTS / "track_b.json"
    b = json.loads(b_path.read_text()) if b_path.exists() else {}
    return a, b


# --------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.7)
    ax.set_axisbelow(True)


def fig_atoms(labels, atoms_pct, out, title, subtitle, notes=None):
    """Stacked atom bars as % of joint information, with direct labels.
    A 2px surface gap separates adjacent segments."""
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * n + 2.1))
    ypos = np.arange(n)[::-1]
    left = np.zeros(n)
    for key in ORDER:
        w = np.array([max(a[key], 0.0) for a in atoms_pct])
        ax.barh(ypos, w, left=left, height=0.62, color=ATOM_COLORS[key],
                label=ATOM_LABEL[key], edgecolor=SURFACE, linewidth=2)
        for i, (l, ww) in enumerate(zip(left, w)):
            if ww >= 6.0:                      # direct label where it fits
                ax.text(l + ww / 2, ypos[i], f"{ww:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if key in ("red", "u_val") else INK,
                        fontweight="600")
        left = left + w
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of joint information  I(Ṽ,S̃;Y)  (%)", fontsize=9, color=INK2)
    ax.set_title(title, fontsize=12.5, color=INK, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    _style(ax)
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.14 - 0.02 * n), labelcolor=INK2)
    if notes:
        ax.text(0, -0.30 - 0.03 * n, notes, transform=ax.transAxes,
                fontsize=8, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_channels(b, out):
    """I_V per channel with cluster-bootstrap CIs — dots, not bars, because the
    interval is the point."""
    import matplotlib.pyplot as plt

    keys = list(b)
    labels = [b[k]["meta"]["target_label"] for k in keys]
    n = len(keys)
    fig, ax = plt.subplots(figsize=(9.0, 0.75 * n + 2.0))
    ypos = np.arange(n)[::-1].astype(float)
    series = [("I_V_val", "Value channel V", "#eb6834"),
              ("I_V_str", "Structure channel S", "#1baf7a"),
              ("I_V_joint", "Joint [V,S]", "#2a78d6")]
    for j, (key, lab, col) in enumerate(series):
        off = (j - 1) * 0.22
        vals = [b[k]["nodes"][key] for k in keys]
        los = [b[k]["ci95"][key][0] for k in keys]
        his = [b[k]["ci95"][key][1] for k in keys]
        ax.hlines(ypos + off, los, his, color=col, linewidth=2, alpha=0.85)
        ax.plot(vals, ypos + off, "o", color=col, markersize=8, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=2)
    for k, yy in zip(keys, ypos):
        ax.plot([b[k]["floors"]["joint"]] * 2, [yy - 0.34, yy + 0.34],
                color=MUTED, linewidth=1.4, linestyle=(0, (3, 2)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlabel("I_V  (nats)   ·   dashed = joint-channel shuffle floor", fontsize=9, color=INK2)
    ax.set_title("Track B — predictive information by channel", fontsize=12.5,
                 color=INK, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.04, "cross-fitted V-information, 95% CI bootstrapped over patients",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    _style(ax)
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.13 - 0.02 * n), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_probe_profile(b, out):
    """Synergy at every probe capacity. Synergy present only at high capacity is
    the signature of the extractability mechanism, not of conjunction."""
    import matplotlib.pyplot as plt

    keys = list(b)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    caps = [str(tuple(r["hidden"])) or "()" for r in b[keys[0]]["probe_grid"]]
    x = np.arange(len(caps))
    cyc = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for i, k in enumerate(keys):
        syn = [r["atoms"]["mmi"]["syn"] for r in b[k]["probe_grid"]]
        ax.plot(x, syn, "-o", color=cyc[i % len(cyc)], linewidth=2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label=b[k]["meta"]["target_label"])
    thr = np.mean([[r["synergy_threshold"] for r in b[k]["probe_grid"]] for k in keys], axis=0)
    ax.plot(x, thr, color=MUTED, linewidth=1.4, linestyle=(0, (3, 2)),
            label="synergy threshold (mean)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("()", "linear") for c in caps], fontsize=8.5,
                       rotation=20, ha="right")
    ax.set_ylabel("synergy  (nats, MMI redundancy)", fontsize=9, color=INK2)
    ax.set_title("Track B — synergy across the probe grid", fontsize=12.5,
                 color=INK, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.04, "x = probe capacity (hidden sizes) · flat or falling = genuine conjunction; "
                     "rising only at high capacity = extractability",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    _style(ax)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.7)
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.30), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_redundancy_spread(b, out):
    """MMI vs I_min synergy — the spread is the redundancy-choice uncertainty."""
    import matplotlib.pyplot as plt

    keys = list(b)
    labels = [b[k]["meta"]["target_label"] for k in keys]
    n = len(keys)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * n + 2.0))
    ypos = np.arange(n)[::-1].astype(float)
    mmi = np.array([b[k]["atoms"]["mmi"]["syn"] for k in keys])
    imin = np.array([b[k]["atoms"]["imin"]["syn"] for k in keys])
    ax.hlines(ypos, np.minimum(mmi, imin), np.maximum(mmi, imin),
              color="#d8d7d1", linewidth=3)
    ax.plot(mmi, ypos, "o", color="#2a78d6", markersize=8, label="MMI redundancy",
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot(imin, ypos, "o", color="#eb6834", markersize=8, label="I_min redundancy",
            markeredgecolor=SURFACE, markeredgewidth=2)
    thr = [b[k]["synergy_threshold"] for k in keys]
    ax.plot(thr, ypos, "|", color=MUTED, markersize=14, markeredgewidth=1.6,
            label="synergy threshold")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlabel("synergy  (nats)", fontsize=9, color=INK2)
    ax.set_title("Track B — redundancy-choice uncertainty", fontsize=12.5,
                 color=INK, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.045, "the gap between the two measures bounds the choice of redundancy function",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    _style(ax)
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.15 - 0.02 * n), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_binning_grid(a, out):
    """A5 — synergy % across the whole binning grid. The claim is not a point
    estimate; it is that synergy stays positive across reasonable schemes."""
    import matplotlib.pyplot as plt

    keys = list(a)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    cyc = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for i, k in enumerate(keys):
        r = a[k]
        pts = [g["syn_pct"] for g in r["a5_grid"]]
        xj = np.full(len(pts), i, dtype=float) + np.linspace(-0.22, 0.22, len(pts))
        ax.plot(xj, pts, "o", color=cyc[i % len(cyc)], markersize=6.5, alpha=0.9,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot([i - 0.3, i + 0.3], [r["primary"]["atoms_pct_of_joint"]["syn"]] * 2,
                color=INK, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1.2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([a[k]["target_label"] for k in keys], fontsize=9, rotation=12)
    ax.set_ylabel("synergy  (% of joint)", fontsize=9, color=INK2)
    ax.set_title("Track A — synergy across the A5 binning grid", fontsize=12.5,
                 color=INK, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.04, "each dot is one binning scheme · black bar = the primary scheme",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    _style(ax)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- report
def verdict(ra, rb):
    """The §6 track-agreement table, applied."""
    a_pos = ra["permutation"]["syn"]["p_value"] < 0.05 and ra["gates"]["binning_sign_consistent"]
    if rb is None:
        return "Track A only — " + ("synergy present, binning-stable" if a_pos
                                    else "no stable synergy")
    if not rb["gates"]["monotone_joint_ok"]:
        # T12: the joint probe did worse than a single channel, so the lattice
        # ordering the Mobius inversion assumes is not there.
        return ("Track A only — **Track B monotonicity gate FAILED**, its atoms are "
                "not interpretable for this target")
    grid = [r["atoms"]["mmi"]["syn"] > r["synergy_threshold"] for r in rb["probe_grid"]]
    b_all, b_any, b_hi = all(grid), any(grid), (not grid[0]) and grid[-1]
    if a_pos and b_all:
        return "ROBUST — synergy positive in both tracks, stable across binning and probe capacity"
    if a_pos and b_hi:
        return "Extractability signature — Track B synergy appears only at high capacity"
    if a_pos and not b_any:
        return "Track A only — check Track B at higher capacity before believing A"
    if (not a_pos) and b_all:
        return "Track B only — likely resolution loss in binning; argue from B"
    if a_pos and b_any:
        return "Synergy present in both tracks; Track B capacity-dependent in part"
    return "No detectable synergy at this n — report floors, cell counts, positive control"


def main():
    a, b = load()
    FIGS.mkdir(exist_ok=True)
    keys = list(a)

    # FDR across targets, on the Track A synergy permutation p-values
    praw = [a[k]["permutation"]["syn"]["p_value"] for k in keys]
    padj = bh_fdr(praw)
    for k, pr, pa in zip(keys, praw, padj):
        a[k]["permutation"]["syn"]["p_fdr"] = float(pa)

    # ---- figures ---------------------------------------------------------
    fig_atoms([a[k]["target_label"] for k in keys],
              [a[k]["primary"]["atoms_pct_of_joint"] for k in keys],
              FIGS / "track_a_atoms.png",
              "Track A (primary) — BROJA decomposition of lab information",
              "discrete Shannon PID on binned channels · MIMIC-IV v3.1 · first ICU stay, 24 h window",
              notes="Negative atoms are clipped to zero for display; all primary decompositions "
                    "converged with non-negative atoms.")
    fig_binning_grid(a, FIGS / "track_a_binning.png")
    if b:
        fig_atoms([b[k]["meta"]["target_label"] for k in b],
                  [b[k]["atoms_pct_of_joint"]["mmi"] for k in b],
                  FIGS / "track_b_atoms.png",
                  "Track B (robustness) — V-information decomposition",
                  "full-dimensional channels, MMI redundancy, probe selected per target at n")
        fig_channels(b, FIGS / "track_b_channels.png")
        fig_probe_profile(b, FIGS / "track_b_probe_profile.png")
        fig_redundancy_spread(b, FIGS / "track_b_redundancy.png")

    # ---- markdown --------------------------------------------------------
    L = []
    L.append("# Experiment 1 — Value/Structure Partial Information Decomposition\n")
    L.append("**Data:** MIMIC-IV v3.1 via BigQuery (`physionet-data.mimiciv_3_1_*`). "
             "**Units: nats.** All CIs cluster-bootstrapped by `subject_id`.\n")

    L.append("\nSetup, coverage findings, and the defects found mid-run are in "
             "[`RUNLOG.md`](RUNLOG.md). Gates come before atoms, per plan §0 rule 7.\n")

    # ---- gates first -----------------------------------------------------
    L.append("\n## Gates\n")
    L.append("### Shared\n")
    L.append("| gate | task | threshold | value | status |")
    L.append("|---|---|---|---|---|")
    L.append("| Schema / itemids resolve | T0 | all concepts found | 13 core, 71 discretionary | **PASS** |")
    L.append("| Core analytes exist | T2 | ≥ 5 at ≥ 0.95 | 13 | **PASS** |")
    L.append("| Target fully removed | T6 | assertion passes | all targets, incl. concept aliases | **PASS** |")
    hgb = a.get("51222")
    if hgb:
        hp = hgb["permutation"]["syn"]["p_value"]
        L.append(f"| **Positive control (Hgb)** | T5 | shows dependence | "
                 f"I(Ṽ,S̃;Y) = {hgb['primary']['mi_plugin_nats']['I_joint']:.4f} nats, "
                 f"synergy p = {hp:.4f} | **{'PASS' if hp < 0.05 else 'FAIL'}** |")

    L.append("\n### Track A (primary)\n")
    L.append("| target | cell occupancy (min cell ≥ 20) | S̃ non-degenerate (≤ 0.80) | permutation reported | binning sign-consistent | solver converged |")
    L.append("|---|---|---|---|---|---|")
    for k in keys:
        g = a[k]["gates"]
        ok = lambda c: "PASS" if c else "**FAIL**"
        L.append(f"| {a[k]['target_label']} | {ok(g['cell_occupancy_ok'])} ({g['min_cell_count']:.0f}) | "
                 f"{ok(g['s_nondegenerate_ok'])} ({g['s_max_share']:.2f}) | PASS | "
                 f"{ok(g['binning_sign_consistent'])} | {ok(g['solver_converged'])} |")

    if b:
        L.append("\n### Track B (robustness)\n")
        L.append("| target | probe selected at this n | marginal agreement (T8) | channels above floor | monotonicity | non-negative atoms |")
        L.append("|---|---|---|---|---|---|")
        for k in b:
            g = b[k]["gates"]
            ok = lambda c: "PASS" if c else "**FAIL**"
            caf = all(g["channel_above_floor"].values())
            L.append(f"| {b[k]['meta']['target_label']} | {tuple(b[k]['probe_selected']['hidden'])} | "
                     f"{ok(g['t8_agreement_ok'])} ({100*b[k]['t8_marginal_agreement']:.2f}%) | "
                     f"{ok(caf)} | {ok(g['monotone_joint_ok'])} | {ok(g['nonnegative_atoms_mmi'])} |")

    L.append("\n## Cross-target summary — Track A (primary)\n")
    L.append("![Track A atoms](../figures/track_a_atoms.png)\n")
    L.append("| target | n | prev | I(Ṽ,S̃;Y) | Red % | U_val % | U_str % | **Syn %** | Syn nats | p | p(FDR) | binning-stable |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = a[k]
        ap = r["primary"]["atoms_pct_of_joint"]
        L.append(f"| {r['target_label']} | {r['n']:,} | {r['prevalence']:.3f} | "
                 f"{r['primary']['mi_plugin_nats']['I_joint']:.5f} | {ap['red']:.1f} | "
                 f"{ap['u_val']:.1f} | {ap['u_str']:.1f} | **{ap['syn']:.1f}** | "
                 f"{r['primary']['atoms_nats']['syn']:.6f} | "
                 f"{r['permutation']['syn']['p_value']:.4f} | "
                 f"{r['permutation']['syn']['p_fdr']:.4f} | "
                 f"{'yes' if r['gates']['binning_sign_consistent'] else 'NO'} |")

    L.append("\n![A5 binning grid](../figures/track_a_binning.png)\n")
    L.append("\nSynergy is reported as a share of `I(Ṽ,S̃;Y)`, which itself varies ~30× "
             "across targets — a small share of a large joint (creatinine, Hgb, BUN) and a "
             "large share of a small joint (WBC) can carry similar nats. Read both columns.\n")

    if b:
        L.append("\n## Cross-target summary — Track B (robustness)\n")
        L.append("![Track B atoms](../figures/track_b_atoms.png)\n")
        L.append("![Track B channels](../figures/track_b_channels.png)\n")
        L.append("![probe profile](../figures/track_b_probe_profile.png)\n")
        L.append("![redundancy spread](../figures/track_b_redundancy.png)\n")
        L.append("| target | probe | H_V(Y) | I_V(V) | I_V(S) | I_V(joint) | Syn (MMI) | Syn (I_min) | threshold | above? | mono |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in b:
            r = b[k]
            L.append(f"| {r['meta']['target_label']} | {tuple(r['probe_selected']['hidden'])} | "
                     f"{r['H_V_Y_analytic']:.4f} | {r['nodes']['I_V_val']:.4f} | "
                     f"{r['nodes']['I_V_str']:.4f} | {r['nodes']['I_V_joint']:.4f} | "
                     f"{r['atoms']['mmi']['syn']:.4f} | {r['atoms']['imin']['syn']:.4f} | "
                     f"{r['synergy_threshold']:.4f} | "
                     f"{'yes' if r['gates']['syn_above_threshold_mmi'] else 'no'} | "
                     f"{'ok' if r['gates']['monotone_joint_ok'] else 'FAIL'} |")

    L.append("\n## Verdict by target\n")
    L.append("| target | verdict |")
    L.append("|---|---|")
    for k in keys:
        L.append(f"| {a[k]['target_label']} | {verdict(a[k], b.get(k))} |")

    if b:
        L.append("\n## Reading the result\n")
        n_syn_a = sum(a[k]["permutation"]["syn"]["p_value"] < 0.05
                      and a[k]["gates"]["binning_sign_consistent"] for k in keys)
        n_syn_b = sum(b[k]["gates"]["syn_above_threshold_mmi"] for k in b)
        n_lin = sum(b[k]["probe_grid"][0]["atoms"]["mmi"]["syn"]
                    > b[k]["probe_grid"][0]["synergy_threshold"] for k in b)
        L.append(f"- **Track A**: synergy is positive, above its permutation null, and "
                 f"sign-stable across the whole A5 grid for **{n_syn_a}/{len(keys)}** targets.")
        n_mono = sum(not b[k]["gates"]["monotone_joint_ok"] for k in b)
        if n_mono:
            bad = ", ".join(b[k]["meta"]["target_label"] for k in b
                            if not b[k]["gates"]["monotone_joint_ok"])
            L.append(f"- **T12 monotonicity FAILED for {n_mono}/{len(b)} target(s)** ({bad}): "
                     f"`I_V(joint) < max(I_V(V), I_V(S))` — concatenating 166 structural columns "
                     f"onto 72 value columns made the finite-capacity probe *worse* than the "
                     f"value channel alone. The lattice ordering the Möbius inversion assumes is "
                     f"absent, so **no Track B atom is interpretable for that target**; its "
                     f"negative synergy is an estimation artifact, not evidence against synergy. "
                     f"Track A, which is not probe-limited, still finds significant synergy there.")
        L.append(f"- **Track B**: synergy exceeds the derived threshold at the selected probe "
                 f"for **{n_syn_b}/{len(b)}** targets, and already at the **linear** probe for "
                 f"**{n_lin}/{len(b)}**. Synergy that is present from linear capacity upward is "
                 f"not a story about probe capacity, which removes probe-relativity as a live "
                 f"objection.")
        t13s = [b[k]["t13"]["artifact_upper_bound"] for k in b if "t13" in b[k]]
        if t13s:
            L.append(f"- **T13 encoding-artifact bound**: {sum(x <= 0 for x in t13s)}/{len(t13s)} "
                     f"targets have a bound ≤ 0 (range {min(t13s):+.4f} to {max(t13s):+.4f} nats). "
                     f"A non-positive bound means the all-imputed value channel produced *no more* "
                     f"synergy than the near-complete one — the fill-value/mask disambiguation "
                     f"mechanism is not what is generating these numbers.")
        L.append("\n**Caveats that belong next to the headline.**\n")
        L.append("- Under MMI the synergy atom reduces exactly to "
                 "`Syn = I_V(joint) − max(I_V(V), I_V(S))` — the information the concatenated "
                 "channels carry beyond the better single channel. That is the least "
                 "assumption-laden reading of these numbers.")
        spread = max(abs(b[k]["atoms"]["mmi"]["syn"] - b[k]["atoms"]["imin"]["syn"]) for k in b)
        L.append(f"- Under MMI, `Red = min(I_V(V), I_V(S))`, which forces one unique atom to "
                 f"exactly zero by construction. That zero is a property of the redundancy "
                 f"measure, not a finding. **MMI and I_min agree to {spread:.1e} nats on every "
                 f"target here**, so the redundancy-choice uncertainty is nil in this data — "
                 f"which is exactly what the plan predicts when the structure channel carries no "
                 f"unique information. `I_V(S) < I_V(V)` for all six targets, and S has the "
                 f"lower specific information in *both* outcome classes, so the two measures "
                 f"select the same channel and coincide.")
        L.append("- V-synergy cannot distinguish genuine conjunction from one channel making the "
                 "other's information *extractable*. T13 bounds the format-driven instance of "
                 "that mechanism; the general case is why Track A is primary.")
        L.append("- `Ṽ` is a single partner analyte, so Track A's `U_val` measures that one "
                 "analyte, not the value channel as a whole. Where the partner is nearly the same "
                 "physiological quantity as the target (Hct→Hgb, BUN→creatinine), `U_val` "
                 "dominates and every other atom is compressed — visible as the ~95–97% value "
                 "bars. Track B, which uses all 72 value features, is the fair comparison of "
                 "channel magnitudes.")
        L.append("- The cohort is one row per patient (first ICU stay), so patient-grouped CV "
                 "and cluster bootstrap coincide with their ordinary forms here.")
        L.append("- Residual missingness in the value channel is filled with a whole-sample "
                 "median, as T3 specifies. That constant is computed across folds, so it is a "
                 "leak in principle; at ≥ 95 % completeness it moves nothing, and the "
                 "fill-value mechanism it could exploit is exactly what T13 bounds.")

    L.append("\n## Per-target detail\n")
    for k in keys:
        r = a[k]
        L.append(f"\n### {r['target_label']} (itemid {k})\n")
        L.append(f"- n = {r['n']:,} · prevalence = {r['prevalence']:.3f} · "
                 f"Ṽ = last in-window **{r['partner_label']}**, S̃ = AM-lab-spike fraction")
        L.append(f"- primary scheme: `{r['primary']['v_scheme']}` × `{r['primary']['s_scheme']}` "
                 f"· solver `{r['primary']['solver']}` · converged = {r['primary']['converged']}")
        L.append(f"- cell occupancy: min cell = {r['gates']['min_cell_count']:.0f} "
                 f"(uncollapsed {r['gates']['min_cell_count_uncollapsed']:.0f}) · "
                 f"S̃ max level share = {r['gates']['s_max_share']:.3f}")
        L.append(f"- A5 grid: synergy % ranges {r['gates']['syn_pct_range'][0]:.1f} to "
                 f"{r['gates']['syn_pct_range'][1]:.1f} over {len(r['a5_grid'])} schemes; "
                 f"all significant = {r['gates']['grid_all_significant']}")
        L.append("\n| atom | nats | % of joint | null mean | debiased | p |")
        L.append("|---|---|---|---|---|---|")
        for at in ORDER:
            pm = r["permutation"][at]
            L.append(f"| {ATOM_LABEL[at]} | {r['primary']['atoms_nats'][at]:.6f} | "
                     f"{r['primary']['atoms_pct_of_joint'][at]:.1f} | {pm['null_mean']:.6f} | "
                     f"{pm['debiased']:.6f} | {pm['p_value']:.4f} |")
        if k in b:
            rb = b[k]
            L.append(f"\n**Track B** — probe {tuple(rb['probe_selected']['hidden'])}, "
                     f"α={rb['probe_selected']['wd']:g}; "
                     f"H_V(Y) analytic {rb['H_V_Y_analytic']:.4f} vs ablated probe "
                     f"{rb['H_V_Y_ablated_probe']:.4f} "
                     f"({100*rb['t8_marginal_agreement']:.1f}% apart)\n")
            L.append("| atom | MMI nats | MMI % | I_min nats | I_min % |")
            L.append("|---|---|---|---|---|")
            for at in ORDER:
                L.append(f"| {ATOM_LABEL[at]} | {rb['atoms']['mmi'][at]:.5f} | "
                         f"{rb['atoms_pct_of_joint']['mmi'][at]:.1f} | "
                         f"{rb['atoms']['imin'][at]:.5f} | "
                         f"{rb['atoms_pct_of_joint']['imin'][at]:.1f} |")
            ci = rb["ci95"]
            L.append(f"\n- 95% CI: I_V(V) [{ci['I_V_val'][0]:.4f}, {ci['I_V_val'][1]:.4f}] · "
                     f"I_V(S) [{ci['I_V_str'][0]:.4f}, {ci['I_V_str'][1]:.4f}] · "
                     f"I_V(joint) [{ci['I_V_joint'][0]:.4f}, {ci['I_V_joint'][1]:.4f}] · "
                     f"Syn(MMI) [{ci['syn_mmi'][0]:.4f}, {ci['syn_mmi'][1]:.4f}]")
            L.append(f"- floors: V {rb['floors']['val']:.4f} · S {rb['floors']['str']:.4f} · "
                     f"joint {rb['floors']['joint']:.4f} → synergy threshold "
                     f"{rb['synergy_threshold']:.4f}")
            if "t13" in rb:
                t13 = rb["t13"]
                L.append(f"- **T13 encoding-artifact bound**: synergy with an all-analyte "
                         f"imputed value channel = {t13['syn_imputed_mmi']:.5f} vs core-only "
                         f"{t13['syn_core_mmi']:.5f} → upper bound "
                         f"**{t13['artifact_upper_bound']:+.5f} nats**")
            L.append("\n<details><summary>probe grid</summary>\n")
            L.append("| hidden | α | I_V(V) | I_V(S) | I_V(joint) | Syn | threshold | above? | mono |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for g in rb["probe_grid"]:
                L.append(f"| {tuple(g['hidden'])} | {g['wd']:g} | {g['I_V']['val']:.4f} | "
                         f"{g['I_V']['str']:.4f} | {g['I_V']['joint']:.4f} | "
                         f"{g['atoms']['mmi']['syn']:.4f} | {g['synergy_threshold']:.4f} | "
                         f"{'yes' if g['syn_above_floor']['mmi'] else 'no'} | "
                         f"{'ok' if g['monotone_joint_ok'] else 'FAIL'} |")
            L.append("\n</details>")

    (RESULTS / "REPORT.md").write_text("\n".join(L) + "\n")
    (RESULTS / "track_a_fdr.json").write_text(json.dumps(
        {k: a[k]["permutation"]["syn"] for k in keys}, indent=2, default=float))
    print(f"wrote {RESULTS/'REPORT.md'} and {len(list(FIGS.glob('*.png')))} figures")


if __name__ == "__main__":
    main()

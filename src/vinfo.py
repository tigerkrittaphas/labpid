"""V-information primitives.

    I_V(X -> Y) = H_V(Y) - H_V(Y | X)

Every quantity is cross-fitted with patient-grouped, stratified folds and
returned *per sample*, so confidence intervals can be cluster-bootstrapped over
patients rather than rows. Units are nats throughout.

Probes are small torch MLPs trained on GPU when one is available. A probe spec
is (hidden_sizes, weight_decay); `hidden_sizes=()` is plain logistic regression,
which is the linear end of the probe family.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-7

# The T7 grid. Capacity is swept from linear up to the oversized regime that is
# known to read a real effect as negative information.
PROBE_GRID = [
    ((), 1e-2),
    ((16,), 1.0),
    ((32,), 1e-1),
    ((64, 32), 1e-2),
    ((128, 64), 1e-3),
    ((512, 512, 256), 1e-4),
]


def _mlp(d_in: int, hidden: tuple[int, ...]) -> nn.Module:
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def _fit_predict(Xtr, ytr, Xte, hidden, wd, seed=0, epochs=200, lr=1e-3, batch=4096,
                 val_frac=0.15, patience=15):
    """Train one probe, return held-out probabilities. Early stopping on an inner
    validation split keeps the oversized capacities from diverging outright while
    still letting them overfit — which is the phenomenon T7 is meant to measure."""
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    n = len(ytr)
    idx = g.permutation(n)
    n_val = max(64, int(val_frac * n))
    va, tr = idx[:n_val], idx[n_val:]

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=DEVICE)

    model = _mlp(Xtr.shape[1], hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = np.inf, None, 0
    tr_t = torch.as_tensor(tr, device=DEVICE)
    va_t = torch.as_tensor(va, device=DEVICE)
    for ep in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xtr_t[b]).squeeze(-1), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xtr_t[va_t]).squeeze(-1), ytr_t[va_t]).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte_t).squeeze(-1)).cpu().numpy()
    return np.clip(p, EPS, 1 - EPS)


def crossfit_nll(X, y, groups, hidden=(128, 64), wd=1e-3, n_splits=5, seed=0):
    """Per-sample held-out negative log-likelihood under the probe family."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    nll = np.empty(len(y), dtype=np.float64)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        p = _fit_predict(sc.transform(X[tr]), y[tr], sc.transform(X[te]),
                         hidden, wd, seed=seed * 100 + k)
        nll[te] = -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    return nll


def h_marginal_binary(y, groups, n_splits=5, seed=0):
    """H_V(Y) for binary Y: the analytic cross-fitted base rate. This is the
    exact infimum over any probe family with a sigmoid head, so it is used in
    place of an ablated-probe fit, which overestimates at low prevalence."""
    y = np.asarray(y, dtype=np.int64)
    nll = np.empty(len(y), dtype=np.float64)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros((len(y), 1)), y, groups):
        p = (y[tr].sum() + 1.0) / (len(tr) + 2.0)          # Laplace smoothing
        p = float(np.clip(p, EPS, 1 - EPS))
        nll[te] = -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    return nll


def v_information(X, y, groups, hidden=(128, 64), wd=1e-3, n_splits=5, seed=0,
                  h_marg=None):
    """I_V(X -> Y) with its per-sample decomposition."""
    if h_marg is None:
        h_marg = h_marginal_binary(y, groups, n_splits=n_splits, seed=seed)
    h_cond = crossfit_nll(X, y, groups, hidden, wd, n_splits=n_splits, seed=seed)
    per_sample = h_marg - h_cond
    return float(per_sample.mean()), per_sample


def noise_floor(X, y, groups, hidden=(128, 64), wd=1e-3, n_splits=5, seeds=(0, 1, 2)):
    """Empirical Rademacher check: permute labels, recompute I_V, take the max
    absolute value. Anything below this is noise."""
    vals = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        yp = rng.permutation(np.asarray(y))
        iv, _ = v_information(X, yp, groups, hidden, wd, n_splits=n_splits, seed=s)
        vals.append(iv)
    return float(np.max(np.abs(vals))), vals


def select_probe(X, y, groups, grid=PROBE_GRID, n_splits=5, floor_seeds=(0, 1),
                 verbose=True):
    """T7 — sweep capacity at the actual n and pick argmax(I_V - shuffle floor).

    Never reuse a capacity chosen at a different sample size: I_V is not
    monotone in probe capacity, and an oversized probe can read a real linear
    effect as negative information.
    """
    h_marg = h_marginal_binary(y, groups, n_splits=n_splits)
    rows = []
    for hidden, wd in grid:
        iv, _ = v_information(X, y, groups, hidden, wd, n_splits=n_splits, h_marg=h_marg)
        fl, _ = noise_floor(X, y, groups, hidden, wd, n_splits=n_splits, seeds=floor_seeds)
        rows.append({"hidden": hidden, "wd": wd, "I_V": iv, "floor": fl,
                     "net": iv - fl, "snr": iv / fl if fl > 0 else np.inf})
        if verbose:
            print(f"  {str(hidden):16s} a={wd:<7.0e} I_V={iv:+.4f} floor={fl:.4f} "
                  f"net={iv - fl:+.4f} snr={rows[-1]['snr']:.2f}")
    best = max(rows, key=lambda r: r["net"])
    return best, rows


def cluster_bootstrap_ci(per_sample, groups, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CI resampling *patients*, not rows."""
    per_sample = np.asarray(per_sample, dtype=np.float64)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    starts = np.searchsorted(sorted_inv, np.arange(len(uniq)))
    ends = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="right")
    sums = np.add.reduceat(per_sample[order], starts) if len(uniq) else np.array([])
    counts = (ends - starts).astype(np.float64)

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = sums[pick].sum() / counts[pick].sum()
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def to_bits(nats: float) -> float:
    return nats / np.log(2)

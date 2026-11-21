"""Shared calibration / interval utilities for the IJPHM revision experiments.

Every experiment script imports from here so that the quantile rule, interval
score and stage definitions are identical across PHM2010 and the nine-tool data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = (0.50, 0.70, 0.80, 0.90, 0.95)


def conformal_q(residuals, level, weights=None):
    """Split-conformal half-width for absolute residual scores.

    Unweighted: the ceil((n+1)*level)-th smallest residual, capped at n
    (identical to the rule used in the MST manuscript).

    Weighted (e.g. equal weight per tool): the smallest residual whose
    normalised weighted CDF reaches min(1, ceil((n+1)*level)/n). With equal
    weights this reduces exactly to the unweighted rule.
    """
    r = np.asarray(residuals, dtype=float)
    n = len(r)
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * level))
    order = np.argsort(r, kind="stable")
    if weights is None:
        return float(r[order][min(k, n) - 1])
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    cw = np.cumsum(w[order])
    target = min(1.0, k / n)
    idx = int(np.searchsorted(cw, target - 1e-12))
    return float(r[order][min(idx, n - 1)])


def group_equal_weights(groups):
    """Weight 1/n_g per observation so each group (tool) contributes equally."""
    g = pd.Series(np.asarray(groups))
    return 1.0 / g.map(g.value_counts()).to_numpy(dtype=float)


def interval_metrics(y, yhat, q, level):
    """Coverage (%), width, Winkler interval score, miss direction for a
    symmetric interval yhat +/- q."""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    alpha = 1.0 - level
    lo, hi = yhat - q, yhat + q
    below, above = y < lo, y > hi
    score = (hi - lo) + (2 / alpha) * (lo - y) * below + (2 / alpha) * (y - hi) * above
    return {
        "coverage": 100.0 * np.mean(~below & ~above),
        "width": 2.0 * q,
        "interval_score": float(np.mean(score)),
        "pct_below": 100.0 * below.mean(),
        "pct_above": 100.0 * above.mean(),
    }


def stage_index(order, n_stages=3):
    """Stage label 1..n_stages from equal-count thirds of tool life (by run
    order). For PHM2010 (315 cuts) this gives cuts 1-105, 106-210, 211-315."""
    ranks = pd.Series(np.asarray(order)).rank(method="first").to_numpy()
    n = len(ranks)
    return np.minimum((np.ceil(ranks * n_stages / n)).astype(int), n_stages)


def stage_coverage(y, yhat, q, order, n_stages=3):
    st = stage_index(order, n_stages)
    hit = np.abs(np.asarray(y, float) - np.asarray(yhat, float)) <= q
    return {f"cov_stage{s}": 100.0 * hit[st == s].mean() for s in range(1, n_stages + 1)}


def stage_wear_rates(order, y, n_stages=3):
    """Least-squares wear slope (wear units per run/cut) in each life stage."""
    order = np.asarray(order, float)
    y = np.asarray(y, float)
    st = stage_index(order, n_stages)
    return {s: float(np.polyfit(order[st == s], y[st == s], 1)[0]) for s in range(1, n_stages + 1)}


def evaluate_all_levels(y, yhat, order, residuals, weights=None, levels=LEVELS):
    """One dict per nominal level: metrics + stage coverage + q."""
    rows = []
    for lv in levels:
        q = conformal_q(residuals, lv, weights)
        row = {"level": lv, "q": q, "n_cal": len(residuals)}
        row.update(interval_metrics(y, yhat, q, lv))
        row.update(stage_coverage(y, yhat, q, order))
        rows.append(row)
    return rows

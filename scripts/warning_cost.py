"""Decision cost of first warnings (PHM2010), computed from t2_warning_timing.csv.

A warning is judged against a window of W cuts that ends at the first measured crossing c
of the 150 um threshold (cuts c-W .. c):
  premature : warning w < c - W           cost = c - W - w        (cuts of tool life wasted)
  timely    : c - W <= w <= c             cost = 0
  late      : w > c                       cost = 5 * (w - c)      (cuts run past the crossing)
  missed    : no warning in 315 cuts      cost = 5 * (315 - c + 1)
The factor 5 is the same false-negative weight used to tune the cutoffs (FP + 5 FN).
W = 30 is the main window; W = 20 and W = 50 are sensitivity checks.

  python warning_cost.py --results ../results
Writes results/t2_warning_cost.csv (one row per protocol x model x test x rule x window).
If t2_warning_timing.csv carries per-cut fp_cuts / fn_cuts (pipeline after 21 Nov 2026),
they are copied through.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CUTS = 315
LATE_WEIGHT = 5.0
WINDOWS = (30, 20, 50)
MAIN_WINDOW = 30


def classify(w, c, W):
    if w is None or np.isnan(w):
        return "missed", LATE_WEIGHT * (N_CUTS - c + 1)
    w = int(w)
    if w < c - W:
        return "premature", float(c - W - w)
    if w <= c:
        return "timely", 0.0
    return "late", LATE_WEIGHT * (w - c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    a = ap.parse_args()
    res = Path(a.results)
    d = pd.read_csv(res / "t2_warning_timing.csv")
    rows = []
    for _, r in d.iterrows():
        for W in WINDOWS:
            cls, cost = classify(r.first_warning, int(r.first_crossing), W)
            row = {"protocol": r.protocol, "model": r.model, "test": r.test, "rule": r.rule,
                   "window": W, "first_crossing": int(r.first_crossing),
                   "first_warning": r.first_warning, "class": cls, "cost": cost}
            for k in ("fp_cuts", "fn_cuts"):
                if k in d.columns:
                    row[k] = r[k]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(res / "t2_warning_cost.csv", index=False)
    tot = out.groupby(["window", "protocol", "rule", "model"]).cost.sum().unstack("model")
    print("Total cost over the three cutters (lower is better):")
    print(tot.to_string())


if __name__ == "__main__":
    main()

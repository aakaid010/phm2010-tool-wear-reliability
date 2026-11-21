"""Paired Wilcoxon tests for the nine-tool calibration schemes (nominal 90%).

Per held-out tool, three paired statistics are compared between two schemes:
  abs_dev_from_90 : |coverage - 90|   (percentage points)
  raw_coverage    : coverage (%)
  interval_score  : mean Winkler interval score (um; lower = better coverage-width trade-off)
Test: two-sided exact Wilcoxon signed-rank (scipy.stats.wilcoxon, method="exact",
zero_method="wilcox").

Dependence. The 9 tools sit on 3 machines. Tools on one machine share a machine
and, in the new-machine runs, share the SAME calibration set and model. So each
comparison is reported at two levels:
  unit = tool    : n = 9 paired tool values (treats tools as independent; optimistic)
  unit = machine : n = 3 paired machine means (mean over that machine's 3 tools).
With n = 3 the smallest attainable exact two-sided p is 0.25, so the machine-level
test can never reach p < 0.05; its value is the direction count (k/3 machines).

  python stats_wilcoxon.py --results ../results              # RF results
  python stats_wilcoxon.py --results ../results --suffix _hgb  # HGB results
Writes results/t3_wilcoxon_pertool{suffix}.csv and results/t3_wilcoxon_tests{suffix}.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COMPARISONS = [
    # (name, file stem, scheme_a, block_a, scheme_b, block_b); differences are a - b
    ("tool_interleaved_vs_loto", "t3_deploy", "S1_same_tools", "interleaved", "S4_loto", "-"),
    ("tool_late_vs_loto", "t3_deploy", "S1_same_tools", "late", "S4_loto", "-"),
    ("machine_loto_vs_lomo", "t3_machine_out", "S4_loto", "-", "S5_lomo", "-"),
]
STATS = ["abs_dev_from_90", "raw_coverage", "interval_score"]


def per_tool(d, scheme, block):
    s = d[(d.scheme == scheme) & (d.block == block) & np.isclose(d.level, 0.9)]
    g = s.groupby("test").agg(coverage=("coverage", "mean"),
                              interval_score=("interval_score", "mean"),
                              width=("width", "mean"), machine=("machine", "first"))
    g["abs_dev_from_90"] = (g.coverage - 90).abs()
    g["raw_coverage"] = g.coverage
    return g


def exact_test(x, y):
    diff = np.asarray(x, float) - np.asarray(y, float)
    nz = diff[diff != 0]
    out = {"n_units": len(diff), "n_nonzero": len(nz),
           "n_a_greater": int(np.sum(diff > 0)), "n_b_greater": int(np.sum(diff < 0)),
           "mean_diff_a_minus_b": float(np.mean(diff)),
           "median_diff_a_minus_b": float(np.median(diff)),
           "min_diff": float(np.min(diff)), "max_diff": float(np.max(diff))}
    if len(nz) == 0:
        out.update(W=np.nan, p_exact_two_sided=1.0, min_attainable_p=np.nan)
        return out
    r = wilcoxon(x, y, alternative="two-sided", method="exact", zero_method="wilcox")
    out.update(W=float(r.statistic), p_exact_two_sided=float(r.pvalue),
               min_attainable_p=float(2 / 2 ** len(nz)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    res = Path(a.results)
    pt_rows, test_rows = [], []
    for name, stem, sa, ba, sb, bb in COMPARISONS:
        d = pd.read_csv(res / f"{stem}{a.suffix}.csv")
        A, B = per_tool(d, sa, ba), per_tool(d, sb, bb)
        A, B = A.align(B, join="inner", axis=0)
        for t in A.index:
            row = {"comparison": name, "test_tool": t, "machine": int(A.machine[t])}
            for s in STATS + ["width"]:
                row[f"{s}_a"], row[f"{s}_b"] = A[s][t], B[s][t]
                row[f"{s}_diff_a_minus_b"] = A[s][t] - B[s][t]
            pt_rows.append(row)
        Am, Bm = A.groupby("machine")[STATS].mean(), B.groupby("machine")[STATS].mean()
        for unit, (X, Y) in {"tool": (A, B), "machine": (Am, Bm)}.items():
            for s in STATS:
                r = exact_test(X[s], Y[s])
                r.update(comparison=name, unit=unit, statistic_compared=s,
                         scheme_a=f"{sa}/{ba}", scheme_b=f"{sb}/{bb}")
                test_rows.append(r)
    pd.DataFrame(pt_rows).to_csv(res / f"t3_wilcoxon_pertool{a.suffix}.csv", index=False)
    t = pd.DataFrame(test_rows)
    cols = ["comparison", "unit", "statistic_compared", "scheme_a", "scheme_b", "n_units",
            "n_nonzero", "n_a_greater", "n_b_greater", "mean_diff_a_minus_b",
            "median_diff_a_minus_b", "min_diff", "max_diff", "W", "p_exact_two_sided",
            "min_attainable_p"]
    t = t[cols]
    t.to_csv(res / f"t3_wilcoxon_tests{a.suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.drop(columns=["scheme_a", "scheme_b", "min_diff", "max_diff"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""T5: wear-rate mismatch vs stage-wise coverage shortfall (PHM2010, RF, nominal 90%).

For every matched combination (split x test cutter T x training cutter A x calibration
source) and every life stage s (cuts 1-105, 106-210, 211-315):
  x = |slope(calibration cutter, s) - slope(T, s)|   OLS wear slope, um/cut
  y = 90 - coverage of T's stage-s cuts               percentage points (positive = shortfall)
The calibration cutter is A for 'same_cutter' and the other source cutter B otherwise.
For 'other_cutter_full_life' the residual pool spans all of B's life; the stage-s slope
of B is still used, so x is the same stage-matched mismatch for every source.

Spearman rho is reported per split (all points) and per split x source, with n.
Dependence: there are only 3 cutters, so x takes at most 3 cutter pairs x 3 stages = 9
distinct values per split; points sharing a cutter pair are not independent. The rho
values are descriptive associations, not tests of a mechanism.

  python fig_wear_mismatch.py --results ../results --out ../manuscript/generated
Writes results/t5_wear_mismatch.csv, results/t5_spearman.csv,
       <out>/fig_wear_mismatch.pdf (+ .png preview)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Validated categorical slots 1-3 (dataviz reference palette, light surface);
# marker shape is a second encoding so identity never rests on colour alone.
SOURCES = [("same_cutter", "Training cutter", "#2a78d6", "o"),
           ("other_cutter_same_range", "Other cutter, same life range", "#eb6834", "s"),
           ("other_cutter_full_life", "Other cutter, full life", "#1baf7a", "^")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLITS = [("chrono", "Chronological split"), ("v2", "Interleaved split (V2)")]


def build(results):
    m = pd.read_csv(results / "t1_matched_rf.csv")
    m = m[np.isclose(m.level, 0.9)]
    wr = pd.read_csv(results / "wear_rate_phm.csv").set_index(["cutter", "stage"]).slope_um_per_cut
    rows = []
    for _, r in m.iterrows():
        for s in (1, 2, 3):
            st, sc = wr[(r.test, s)], wr[(r.cal_cutter, s)]
            rows.append({"protocol": r.protocol, "test": r.test, "train": r.train,
                         "cal_cutter": r.cal_cutter, "cal_source": r.cal_source, "stage": s,
                         "slope_test": st, "slope_cal": sc, "mismatch": abs(sc - st),
                         "coverage_stage": r[f"cov_stage{s}"],
                         "shortfall_pp": 90.0 - r[f"cov_stage{s}"]})
    return pd.DataFrame(rows)


def spearman_table(d):
    rows = []
    for prot, _ in SPLITS:
        groups = [("all", d[d.protocol == prot])] + \
                 [(k, d[(d.protocol == prot) & (d.cal_source == k)]) for k, *_ in SOURCES]
        for name, g in groups:
            rho, p = spearmanr(g.mismatch, g.shortfall_pp)
            rows.append({"protocol": prot, "cal_source": name, "n_points": len(g),
                         "n_distinct_mismatch": g.mismatch.round(9).nunique(),
                         "spearman_rho": rho, "p_nominal": p})
    return pd.DataFrame(rows)


def plot(d, sp, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, constrained_layout=True)
    ymin = min(-15, d.shortfall_pp.min() - 5)
    ymax = d.shortfall_pp.max() + 6
    for ax, (prot, title) in zip(axes, SPLITS):
        g = d[d.protocol == prot]
        ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.text(0.99, 0.5, "nominal 90%", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=7, color=INK2)
        for key, label, col, mk in SOURCES:
            s = g[g.cal_source == key]
            ax.scatter(s.mismatch, s.shortfall_pp, s=34, marker=mk, facecolor=col,
                       edgecolor="white", linewidth=0.9, alpha=0.9, label=label, zorder=3)
        r = sp[(sp.protocol == prot) & (sp.cal_source == "all")].iloc[0]
        ax.set_title(f"{title}\nSpearman $\\rho$ = {r.spearman_rho:.2f}, n = {r.n_points}",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Wear-rate mismatch, |calibration − test| (µm/cut)")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Coverage shortfall, 90 − stage coverage (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=8,
               title="Calibration residuals from", title_fontsize=8)
    fig.savefig(out / "fig_wear_mismatch.pdf")
    fig.savefig(out / "fig_wear_mismatch.png", dpi=200)
    print("wrote fig_wear_mismatch.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../manuscript/generated")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = build(res)
    d.to_csv(res / "t5_wear_mismatch.csv", index=False)
    sp = spearman_table(d)
    sp.to_csv(res / "t5_spearman.csv", index=False)
    print(sp.round(3).to_string(index=False))
    plot(d, sp, out)


if __name__ == "__main__":
    main()

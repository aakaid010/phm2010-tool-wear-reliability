"""Results figures for the nine-tool data (RF point model).

  fig_coverage_pertool.pdf : per-tool coverage at nominal 90%, new tool | new machine
  fig_coverage_levels.pdf  : empirical vs nominal coverage (50-95%), mean and worst tool

  python fig_results.py --results ../results --out <paper>/figures

Inputs: results/t3_deploy.csv (new tool) and results/t3_machine_out.csv (new machine).
Palette: validated categorical slots 1-4 (dataviz reference palette); each scheme also has
its own marker shape, so identity never relies on colour alone (print / CVD).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SCHEMES = [  # (scheme, block, label, colour, marker)
    ("S1_same_tools", "interleaved", "S1 interleaved", "#2a78d6", "o"),
    ("S1_same_tools", "late", "S1 late", "#eb6834", "s"),
    ("S4_loto", "-", "S4 LOTO", "#1baf7a", "^"),
    ("S5_lomo", "-", "S5 LOMO", "#eda100", "D"),
]
PANELS = [("t3_deploy.csv", "a  New tool"), ("t3_machine_out.csv", "b  New machine")]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FULL_WIDTH_IN = 174 / 25.4  # Springer full-width figure (174 mm)


def present(d):
    return [s for s in SCHEMES if ((d.scheme == s[0]) & (d.block == s[1])).any()]


def per_tool(d, level):
    d = d[np.isclose(d.level, level)]
    return d.groupby(["scheme", "block", "test"]).coverage.mean()


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_pertool(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.5), sharey=True,
                             gridspec_kw={"width_ratios": [3, 4]})
    rng = np.random.default_rng(0)  # horizontal jitter only, for visibility
    for ax, (f, title) in zip(axes, PANELS):
        d = pd.read_csv(res / f)
        pt = per_tool(d, 0.9)
        sch = present(d)
        for i, (s, b, lab, col, mk) in enumerate(sch):
            v = pt.loc[(s, b)].to_numpy()
            x = i + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=22, color=col, marker=mk, edgecolor="white", linewidth=0.6,
                       zorder=3)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color=INK, linewidth=1.6, zorder=4)
            ax.text(i + 0.31, v.mean(), f"{v.mean():.1f}", va="center", ha="left",
                    fontsize=7, color=INK)
        ax.axhline(90, color=INK2, linestyle="--", linewidth=0.8, zorder=2)
        ax.set_xticks(range(len(sch)), [s[2] for s in sch])
        ax.set_xlim(-0.5, len(sch) - 0.3)
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        style(ax)
    axes[0].set_ylabel("Coverage at nominal 90% (%)")
    axes[0].set_ylim(-3, 103)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4,
                          label="one held-out tool"),
               plt.Line2D([], [], color=INK, linewidth=1.6, label="mean over 9 tools"),
               plt.Line2D([], [], color=INK2, linestyle="--", linewidth=0.8, label="nominal 90%")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig_coverage_pertool.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "fig_coverage_pertool.png", dpi=600)
    plt.close(fig)


def fig_levels(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.7), sharey=True)
    for ax, (f, title) in zip(axes, PANELS):
        d = pd.read_csv(res / f)
        levels = sorted(d.level.unique())
        ax.plot([45, 100], [45, 100], color=GRID, linewidth=1.0, zorder=1)
        for s, b, lab, col, mk in present(d):
            g = d[(d.scheme == s) & (d.block == b)].groupby(["level", "test"]).coverage.mean()
            mean = [g.loc[lv].mean() for lv in levels]
            worst = [g.loc[lv].min() for lv in levels]
            x = [100 * lv for lv in levels]
            ax.plot(x, mean, color=col, marker=mk, markersize=4.5, linewidth=1.5, label=lab,
                    zorder=3)
            ax.plot(x, worst, color=col, linestyle=":", linewidth=1.1, zorder=2)
        ax.set_xlim(45, 100)
        ax.set_xticks([50, 70, 80, 90, 95])
        ax.set_xlabel("Nominal coverage (%)")
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        style(ax)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_ylim(-3, 103)
    sch_all = present(pd.read_csv(res / PANELS[1][0]))
    handles = [plt.Line2D([], [], color=c, marker=m, markersize=4.5, linewidth=1.5, label=l)
               for _, _, l, c, m in sch_all]
    handles += [plt.Line2D([], [], color=INK2, linewidth=1.5, label="mean over tools"),
                plt.Line2D([], [], color=INK2, linestyle=":", linewidth=1.1, label="worst tool"),
                plt.Line2D([], [], color=GRID, linewidth=1.0, label="nominal = empirical")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_coverage_levels.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "fig_coverage_levels.png", dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../manuscript/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_pertool(res, out)
    fig_levels(res, out)
    print("wrote fig_coverage_pertool.pdf and fig_coverage_levels.pdf in", out)


if __name__ == "__main__":
    main()

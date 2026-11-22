"""Schematic figures for an earlier manuscript draft (no data; layout only).

  fig_overview.pdf : study overview - only components that were evaluated
  fig_schemes.pdf  : calibration schemes S1-S5 as train / calibrate / test bars per tool

  python fig_schemes.py --out ../manuscript/generated
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
# validated categorical slots (dataviz reference palette) + neutral
TRAIN, CAL, TEST = "#2a78d6", "#eb6834", "#1baf7a"
UNUSED = "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})


def overview(out):
    """Four columns; arrows connect whole columns (not rows), because every model is
    combined with every calibration scheme. Dataset-specific items are labelled."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    cols = [
        ("1  Data", ["PHM2010\n3 cutters × 315 cuts\nforce, vibration, AE",
                     "Nine-tool data\n9 tools on 3 machines\n6,418 force recordings"]),
        ("2  Point models", ["Random forest (RF)\nand gradient boosting\n(HGB), same features",
                             "Quadratic cut-index\ncurve, no sensors\n(PHM2010 only)"]),
        ("3  Calibration\nresiduals from", ["S1 same tools:\ninterleaved or\nlate-life block",
                                            "S2/S3 another tool\nS4 tool held out\nS5 machine held out"]),
        ("4  Evaluated on\nheld-out units", ["Coverage, width,\ninterval score,\nworst-tool coverage",
                                             "First warning at\n150 µm vs crossing\n(PHM2010 only)"]),
    ]
    xs = [0.5, 25.5, 50.5, 75.5]
    w, h = 23, 14
    for (title, boxes), x in zip(cols, xs):
        ax.text(x + w / 2, 43.5, title, ha="center", va="top", fontsize=8.2, color=INK, weight="bold")
        for j, txt in enumerate(boxes):
            y = 19.5 - j * 18
            ax.add_patch(FancyBboxPatch((x + .4, y), w - .8, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                        fc=SURF, ec=INK2, lw=0.8))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2, color=INK,
                    linespacing=1.25)
    for x in xs[:-1]:  # one arrow per column boundary, centred between the two boxes
        ax.add_patch(FancyArrowPatch((x + w - .2, 17.5), (x + 25 + .2, 17.5), arrowstyle="-|>",
                                     mutation_scale=9, color=INK2, lw=0.9))
    fig.savefig(out / "fig_overview.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def schemes(out):
    tools = [f"M{(i // 3) + 1}T{i + 1}" for i in range(9)]
    rows = [
        ("S1  same tools,\ninterleaved block", "int"),
        ("S1  same tools,\nlate-life block", "late"),
        ("S2  other tool,\nsame machine", "s2"),
        ("S3  other tool,\nother machine", "s3"),
        ("S4  leave-one-\ntool-out (LOTO)", "loto"),
        ("S5  leave-one-\nmachine-out (LOMO)", "lomo"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bw, gap, mgap = 0.86, 0.14, 0.45
    xpos, x = [], 0.0
    for i in range(9):
        if i and i % 3 == 0:
            x += mgap
        xpos.append(x)
        x += bw + gap
    for r, (label, kind) in enumerate(rows):
        y = len(rows) - 1 - r
        for i, t in enumerate(tools):
            x0 = xpos[i]
            is_test = (i == 0) if kind != "lomo" else (i < 3)
            if is_test:
                ax.add_patch(Rectangle((x0, y + .18), bw, .64, fc=TEST, ec="white", lw=1))
                continue
            if kind in ("int", "late"):
                ax.add_patch(Rectangle((x0, y + .18), bw, .64, fc=TRAIN, ec="white", lw=1))
                segs = [(0.8, 1.0)] if kind == "late" else [(.10, .15), (.35, .40), (.60, .65), (.85, .90)]
                for a, b in segs:
                    ax.add_patch(Rectangle((x0 + a * bw, y + .18), (b - a) * bw, .64, fc=CAL, ec="none"))
            elif kind in ("s2", "s3"):
                ext = 1 if kind == "s2" else 3
                ax.add_patch(Rectangle((x0, y + .18), bw, .64, fc=CAL if i == ext else TRAIN,
                                       ec="white", lw=1))
            else:  # loto / lomo: every source unit is fitted on and, out-of-fold, calibrates
                ax.add_patch(Rectangle((x0, y + .18), bw, .64, fc=TRAIN, ec="white", lw=1))
                ax.add_patch(Rectangle((x0, y + .18), bw, .64, fc="none", ec=CAL, lw=0,
                                       hatch="/////"))
        ax.text(-0.25, y + .5, label, ha="right", va="center", fontsize=7.6, color=INK)
    for i, t in enumerate(tools):
        ax.text(xpos[i] + bw / 2, len(rows) + .05, t, ha="center", va="bottom", fontsize=7, color=INK2)
    for m in range(3):
        xa, xb = xpos[3 * m], xpos[3 * m + 2] + bw
        ax.plot([xa, xb], [len(rows) + .55] * 2, color=INK2, lw=.8)
        ax.text((xa + xb) / 2, len(rows) + .62, f"Machine {m + 1}", ha="center", va="bottom",
                fontsize=7.6, color=INK)
    ax.set_xlim(-3.0, xpos[-1] + bw + .1)
    ax.set_ylim(-0.9, len(rows) + 1.2)
    ax.axis("off")
    plt.rcParams["hatch.linewidth"] = 1.2
    handles = [Patch(fc=TRAIN, label="Fit"), Patch(fc=CAL, label="Calibration residuals"),
               Patch(fc=TRAIN, ec=CAL, hatch="/////", lw=0,
                     label="Fit + out-of-group residuals (one group held out per fold)"),
               Patch(fc=TEST, label="Test (never used for fitting or calibration)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              frameon=False, fontsize=7.4)
    fig.savefig(out / "fig_schemes.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_schemes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../manuscript/generated")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["hatch.color"] = CAL
    plt.rcParams["hatch.linewidth"] = 1.2
    overview(out)
    schemes(out)
    print("wrote fig_overview.pdf, fig_schemes.pdf (+ .png previews)")


if __name__ == "__main__":
    main()

"""Schematic (no-data) vector figures for the IJAMT manuscript.

  Fig1_overview.pdf : deployment workflow - tool on a machine -> force/vibration/AE signals ->
                      features -> wear regressor -> calibrated interval -> inspect/replace decision,
                      with the evaluation design (held-out tool or machine) underneath
  Fig2_sources.pdf  : calibration sources S1-S5 for one test tool / test machine (nine-tool data)

Design facts only (from Section 2 of the paper); no result numbers. Greyscale-safe: fills
differ in lightness and the calibration share is also hatched; text is sans-serif 7-8 pt at
174 mm width.

  python fig_schematics.py --out ../manuscript/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

INK, INK2 = "#111111", "#555555"
FIT, CAL, TEST, BOX = "#9ec5ee", "#e8833a", "#ffffff", "#f4f4f2"
HATCH = "////"
MM = 1 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "hatch.linewidth": 0.8, "svg.fonttype": "none"})


def box(ax, x, y, w, h, title, body, fc=BOX, lw=0.8, title_size=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=fc, ec=INK2, lw=lw))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=title_size,
            weight="bold", color=INK)
    ax.text(x + w / 2, y + (h - 3.2) / 2, body, ha="center", va="center", fontsize=6.6,
            color=INK, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=INK2, lw=0.9, shrinkA=0, shrinkB=0))


def overview(out):
    fig, ax = plt.subplots(figsize=(174 * MM, 84 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    # top row: the deployment chain on the shop floor
    ax.text(0.5, 45.5, "a  Deployment on a tool that contributed no training data", ha="left",
            va="top", fontsize=8, weight="bold", color=INK)
    w, h, y = 18.0, 15.0, 25.5
    xs = [0.5, 20.8, 41.1, 61.4, 81.7]
    items = [
        ("Milling", "new tool on a\nknown machine,\nor a new machine"),
        ("Signals", "force, vibration,\nacoustic emission\n(per cut or run)"),
        ("Wear model", "12 features per\nchannel; random\nforest or boosting"),
        ("Wear interval", "$[\\hat y-q,\\ \\hat y+q]$\n$q$ from calibration\nresiduals (S1-S5)"),
        ("Tool decision", "inspect or replace\nwhen the upper\nbound reaches\nthe wear limit"),
    ]
    for x, (t, b) in zip(xs, items):
        box(ax, x, y, w, h, t, b, fc="#ffffff" if t != "Wear interval" else "#fde9d9")
    for x in xs[:-1]:
        arrow(ax, x + w + 0.15, y + h / 2, x + 20.3 - 0.15, y + h / 2)
    # bottom row: what is evaluated, offline
    ax.text(0.5, 21.5, "b  Offline evaluation (this study)", ha="left", va="top", fontsize=8,
            weight="bold", color=INK)
    y2, h2 = 1.0, 17.0
    items2 = [
        (0.5, 31.0, "Data", "PHM2010: 3 tools (C1, C4, C6),\n315 cuts each, max. flute wear\n"
                            "Nine-tool data: 9 tools on\n3 machines, flank wear $VB$"),
        (34.0, 31.0, "Held-out unit", "new tool: one tool left out\nnew machine: one machine's\n"
                                      "three tools left out\nmodel fixed; only the\ncalibration source varies"),
        (67.5, 32.0, "Reported per tool", "coverage, width, interval score\n(mean and worst tool),\n"
                                          "final third of tool life;\nfirst warning and its\ndecision cost (PHM2010)"),
    ]
    for x, ww, t, b in items2:
        box(ax, x, y2, ww, h2, t, b)
    arrow(ax, 31.65, y2 + h2 / 2, 33.85, y2 + h2 / 2)
    arrow(ax, 65.15, y2 + h2 / 2, 67.35, y2 + h2 / 2)
    fig.savefig(out / "Fig1_overview.pdf", bbox_inches="tight")
    fig.savefig(out / "Fig1_overview.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def sources(out):
    tools = [f"T{i + 1}" for i in range(9)]
    rows = [
        ("S1  same tools,\ninterleaved blocks", "int"),
        ("S1  same tools,\nlate-life block", "late"),
        ("S2  another tool,\nsame machine", "s2"),
        ("S3  another tool,\nother machine", "s3"),
        ("S4  leave-one-\ntool-out (LOTO)", "loto"),
        ("S5  leave-one-\nmachine-out (LOMO)", "lomo"),
    ]
    fig, ax = plt.subplots(figsize=(174 * MM, 92 * MM))
    bw, gap, mgap = 0.86, 0.14, 0.45
    xpos, x = [], 0.0
    for i in range(9):
        if i and i % 3 == 0:
            x += mgap
        xpos.append(x)
        x += bw + gap

    def rect(x0, y, a, b, **kw):
        ax.add_patch(Rectangle((x0 + a * bw, y + .18), (b - a) * bw, .64, **kw))

    for r, (label, kind) in enumerate(rows):
        y = len(rows) - 1 - r
        for i in range(9):
            x0 = xpos[i]
            is_test = (i == 0) if kind != "lomo" else (i < 3)
            if is_test:
                rect(x0, y, 0, 1, fc=TEST, ec=INK, lw=0.9)
                ax.text(x0 + bw / 2, y + .5, "test", ha="center", va="center", fontsize=6.3,
                        color=INK)
                continue
            if kind in ("int", "late"):
                rect(x0, y, 0, 1, fc=FIT, ec="white", lw=1)
                segs = [(0.8, 1.0)] if kind == "late" else [(.10, .15), (.35, .40), (.60, .65), (.85, .90)]
                for a, b in segs:
                    rect(x0, y, a, b, fc=CAL, ec=INK, lw=0.3, hatch=HATCH)
            elif kind in ("s2", "s3"):
                ext = 1 if kind == "s2" else 3
                if i == ext:
                    rect(x0, y, 0, 1, fc=CAL, ec=INK, lw=0.3, hatch=HATCH)
                else:
                    rect(x0, y, 0, 1, fc=FIT, ec="white", lw=1)
            else:  # LOTO / LOMO: each source unit is fitted on and, when held out, calibrates
                rect(x0, y, 0, 1, fc=FIT, ec=CAL, lw=1.2, hatch="..")
        ax.text(-0.25, y + .5, label, ha="right", va="center", fontsize=7.2, color=INK)
    for i, t in enumerate(tools):
        ax.text(xpos[i] + bw / 2, len(rows) + .05, t, ha="center", va="bottom", fontsize=6.8, color=INK2)
    for m in range(3):
        xa, xb = xpos[3 * m], xpos[3 * m + 2] + bw
        ax.plot([xa, xb], [len(rows) + .55] * 2, color=INK2, lw=.8)
        ax.text((xa + xb) / 2, len(rows) + .62, f"Machine M{m + 1}", ha="center", va="bottom",
                fontsize=7.4, color=INK)
    ax.set_xlim(-3.0, xpos[-1] + bw + .1)
    ax.set_ylim(-1.5, len(rows) + 1.2)
    ax.axis("off")
    handles = [Patch(fc=FIT, ec="white", label="Fitting data"),
               Patch(fc=CAL, ec=INK, lw=0.3, hatch=HATCH, label="Calibration residuals"),
               Patch(fc=FIT, ec=CAL, lw=1.2, hatch="..",
                     label="Fitted on; held out in turn to give residuals"),
               Patch(fc=TEST, ec=INK, lw=0.9, label="Test tool(s), never fitted or calibrated on")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
              frameon=False, fontsize=6.9, handlelength=1.6)
    fig.savefig(out / "Fig2_sources.pdf", bbox_inches="tight")
    fig.savefig(out / "Fig2_sources.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../manuscript/figures")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    overview(out)
    sources(out)
    print("wrote Fig1_overview.pdf, Fig2_sources.pdf in", out)


if __name__ == "__main__":
    main()

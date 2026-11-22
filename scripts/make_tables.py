"""Turn result CSVs into LaTeX tables + a macro file of in-text numbers.

  python make_tables.py --results ../results --out ../manuscript/generated

Writes
  tab_phm_matched.tex     T1: calibration source x split x model (RF; HGB if run) at nominal 90%,
                          plus training-label-range reference rows
  tab_point_accuracy.tex  T2: macro MAE, chronological + interleaved split, RF vs cut index
  tab_warning.tex         T2: first warnings at 150 um, RF vs cut index, both splits
  tab_ninetool.tex        T3 (RF): new tool, new machine, matched S1/S2/S3 panels at 90%
  tab_ninetool_hgb.tex    T3 (HGB): same layout, model-agnosticism check (if run)
  numbers.tex             \\newcommand macros for every number quoted in the text
Missing CSVs are skipped with a message, so this can run after each experiment.

Scope (revision of 20 Nov 2026): the CNN-BiLSTM and window-RF are out of the paper.
results/t2_point_accuracy_legacy.csv is kept for the record but is NOT read here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SRC_LABEL = {"same_cutter": "Training tool",
             "other_cutter_same_range": "Other tool, same life range",
             "other_cutter_full_life": "Other tool, full life"}
SCHEME_LABEL = {"S1_same_tools": "S1 held-out block of training tools",
                "S2_same_machine": "S2 other tool, same machine",
                "S3_other_machine": "S3 other tool, other machine",
                "S4_loto": "S4 leave-one-tool-out",
                "S4u_loto_unweighted": "S4 (unweighted quantile)",
                "S5_lomo": "S5 leave-one-machine-out"}
MACROS: dict[str, str] = {}
RAW: dict[str, float] = {}  # unrounded values, so derived ratios are not computed from rounded text
PCT_UM = "{:.1f}"  # coverage (%) and width / interval score (um): one decimal, as in the tables


def macro(name, value, fmt="{:.2f}"):
    assert name.isalpha(), f"LaTeX macro names must be letters only: {name}"
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value
    if not isinstance(value, str):
        RAW[name] = float(value)


def write_table(path, caption, label, header, rows, align, fit=True, wide=False):
    """fit=True scales the tabular to the line width (needs adjustbox). wide=True writes a
    table* that spans both columns of the two-column Springer Nature [iicol] layout. The
    caption is placed ABOVE the table (Springer style). Footnote rows (single-cell rows) are
    set as a note under the table."""
    notes = [r for r in rows if len(r) == 1]
    rows = [r for r in rows if len(r) > 1]
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    tab = (f"\\begin{{tabular}}{{{align}}}\\toprule\n{header} \\\\ \\midrule\n{body}\n"
           "\\bottomrule\\end{tabular}")
    if fit and wide:  # shrink to the page width only if wider; never enlarge (needs adjustbox)
        tab = "\\begin{adjustbox}{max width=\\linewidth}\n" + tab + "\n\\end{adjustbox}"
    elif fit:  # sn-jnl single-column floats break inside adjustbox: tighten instead
        tab = "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + tab + "}"
    note_txt = ""
    for n in notes:  # strip the \multicolumn wrapper; keep the note text
        t = n[0]
        if t.startswith("\\multicolumn"):
            t = t[t.index("}{", t.index("}{") + 2) + 2:]  # drop "\multicolumn{k}{spec}{"
            t = t[t.index("}") + 1:] if t.startswith("\\linewidth") else t
            t = t[:-1] if t.endswith("}") else t
        note_txt += f"\n\\par\\smallskip\\parbox{{\\linewidth}}{{{t}}}"
    env = "table*" if wide else "table"
    path.write_text(
        f"\\begin{{{env}}}[tbp]\n\\centering\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{tab}{note_txt}\n\\end{{{env}}}\n",
        encoding="utf-8")
    print("wrote", path.name)


def load(results, name):
    p = results / name
    if not p.exists():
        print("skip (missing):", name)
        return None
    return pd.read_csv(p)


def phm_matched(results, out):
    parts = []
    for f, model in [("t1_matched_rf.csv", "RF"), ("t1_matched_hgb.csv", "HGB")]:
        d = load(results, f)
        if d is not None:
            parts.append(d.assign(model_name=model))
    if not parts:
        return
    d = pd.concat(parts)
    d = d[np.isclose(d.level, 0.9)]
    cols = ["coverage", "width", "interval_score", "cov_stage3"]
    g = d.groupby(["protocol", "model_name", "cal_source"])[cols].agg(["mean", "std", "min"])
    lr = load(results, "t1_label_range.csv")
    gl = lr.groupby(["protocol", "reference"])[cols].agg(["mean", "std", "min"]) if lr is not None else None
    split_name = {"chrono": "Chronological", "v2": "Interleaved"}
    ref_name = {"minmax": "Reference: training-label range, min--max",
                "p05_p95": "Reference: training-label range, 5th--95th pct."}

    def fmt_row(prot, model, label, r):
        return [split_name[prot], model, label,
                f"{r[('coverage', 'mean')]:.1f} $\\pm$ {r[('coverage', 'std')]:.1f}",
                f"{r[('coverage', 'min')]:.1f}",
                f"{r[('width', 'mean')]:.1f}", f"{r[('interval_score', 'mean')]:.1f}",
                f"{r[('cov_stage3', 'mean')]:.1f}"]

    rows = []
    for prot in ["chrono", "v2"]:
        for (p, model, src), r in g.iterrows():
            if p == prot:
                rows.append(fmt_row(prot, model, SRC_LABEL[src], r))
        if gl is not None:
            for ref in ["minmax", "p05_p95"]:
                r = gl.loc[(prot, ref)]
                rows.append(fmt_row(prot, "--", ref_name[ref], r))
                tag = {"chrono": "Chrono", "v2": "Ilv"}[prot] + {"minmax": "MinMax", "p05_p95": "Pct"}[ref]
                if ref == "minmax":  # highest training label any single training cutter saw
                    ptag = {"chrono": "Chrono", "v2": "Ilv"}[prot]
                    macro(f"maxTrainLabel{ptag}",
                          lr[(lr.protocol == prot) & (lr.reference == ref)].upper.max())
                macro(f"covRange{tag}", r[("coverage", "mean")], PCT_UM)
                macro(f"widRange{tag}", r[("width", "mean")], PCT_UM)
                macro(f"isRange{tag}", r[("interval_score", "mean")], PCT_UM)
                macro(f"covLastRange{tag}", r[("cov_stage3", "mean")], PCT_UM)
    for (prot, model, src), r in g.iterrows():
        tag = {"chrono": "Chrono", "v2": "Ilv"}[prot] + {"RF": "RF", "HGB": "Hgb"}[model] + \
            {"same_cutter": "Same", "other_cutter_same_range": "OtherRange",
             "other_cutter_full_life": "OtherFull"}[src]
        macro(f"cov{tag}", r[("coverage", "mean")], PCT_UM)
        macro(f"covmin{tag}", r[("coverage", "min")], PCT_UM)
        macro(f"wid{tag}", r[("width", "mean")], PCT_UM)
        macro(f"is{tag}", r[("interval_score", "mean")], PCT_UM)
        macro(f"covLast{tag}", r[("cov_stage3", "mean")], PCT_UM)      # final third of tool life
        macro(f"covminLast{tag}", r[("cov_stage3", "min")], PCT_UM)
    write_table(out / "tab_phm_matched.tex",
                "PHM2010 matched calibration-source comparison at nominal 90\\%: for each model, identical "
                "test predictions; only the calibration residuals change. RF: random forest; HGB: "
                "histogram gradient boosting (same features). Mean $\\pm$ SD and worst case over "
                "six directed train/test combinations from three tools. Reference rows are fixed bands "
                "spanning the training tool's own training labels (no prediction involved). Width and "
                "interval score in $\\mu$m.",
                "tab:phm_matched",
                "Split & Model & Calibration residuals & Coverage (\\%) & Worst (\\%) & Width & "
                "Interval score & Final-third cov. (\\%)",
                rows, "lllrrrrr", wide=True)


def wilcoxon_macros(results, sfx="", mprefix=""):
    """Macros from stats_wilcoxon.py output.  wilBetter... = number of units on which
    scheme B (LOTO, or LOMO) is better: higher raw coverage, or lower |cov-90| /
    interval score."""
    t = load(results, f"t3_wilcoxon_tests{sfx}.csv")
    if t is None:
        return
    names = {"tool_interleaved_vs_loto": "IntLoto", "tool_late_vs_loto": "LateLoto",
             "machine_loto_vs_lomo": "LotoLomo"}
    stats = {"abs_dev_from_90": "Dev", "raw_coverage": "Raw", "interval_score": "IS"}
    for _, r in t.iterrows():
        tag = mprefix + names[r.comparison] + stats[r.statistic_compared] + \
            {"tool": "Tool", "machine": "Mach"}[r.unit]
        macro(f"wilP{tag}", r.p_exact_two_sided, "{:.4f}")
        macro(f"wilW{tag}", r.W, "{:.0f}")
        macro(f"wilN{tag}", str(int(r.n_units)))
        macro(f"wilMeanDiff{tag}", r.mean_diff_a_minus_b, PCT_UM)
        macro(f"wilMedDiff{tag}", r.median_diff_a_minus_b, PCT_UM)
        better = r.n_b_greater if r.statistic_compared == "raw_coverage" else r.n_a_greater
        macro(f"wilBetter{tag}", str(int(better)))


def spearman_macros(results):
    """T5 wear-rate mismatch association (fig_wear_mismatch.py output)."""
    sp = load(results, "t5_spearman.csv")
    if sp is None:
        return
    src = {"all": "All", "same_cutter": "Same", "other_cutter_same_range": "OtherRange",
           "other_cutter_full_life": "OtherFull"}
    for _, r in sp.iterrows():
        tag = {"chrono": "Chrono", "v2": "Ilv"}[r.protocol] + src[r.cal_source]
        macro(f"rhoMismatch{tag}", r.spearman_rho)
        macro(f"nMismatch{tag}", str(int(r.n_points)))
        macro(f"nDistinctMismatch{tag}", str(int(r.n_distinct_mismatch)))


def derived_macros():
    """Ratios quoted in the text, computed from macros already generated (never typed)."""
    def f(k):
        return RAW[k]
    pairs = {
        "ratioWidLateLoto": ("widToolSOneLate", "widToolSFour"),
        "ratioWidHgbLateLoto": ("widHgbToolSOneLate", "widHgbToolSFour"),
        "ratioWidLomoLoto": ("widMachSFive", "widMachSFour"),
        "ratioWidHgbLomoLoto": ("widHgbMachSFive", "widHgbMachSFour"),
        "ratioWidChronoSameRange": ("widChronoRFSame", "widRangeChronoMinMax"),
        "ratioWidIlvOtherRange": ("widIlvRFOtherRange", "widRangeIlvMinMax"),
    }
    for name, (a, b) in pairs.items():
        if a in MACROS and b in MACROS:
            macro(name, f(a) / f(b), "{:.1f}")


def deployment_guidance(results, out, level=0.9):
    """Shop-floor reading of the nine-tool RF intervals (no new experiment; t3_deploy.csv,
    t3_machine_out.csv and wear_rate_ninetool.csv only).
    Per test tool: half-width q (um of VB), share of measurements above the upper bound
    (risk of replacing too late if the upper bound is used as the replacement trigger), and
    q divided by the tool's own final-third wear rate (um/run) = runs of tool life given up
    when replacement is triggered by the upper bound instead of the point prediction.
    Reported as mean over the nine tools (q, above) and median over tools (runs)."""
    rate = load(results, "wear_rate_ninetool.csv")
    dep, mo = load(results, "t3_deploy.csv"), load(results, "t3_machine_out.csv")
    if rate is None or dep is None or mo is None:
        return
    last = rate[rate.stage == rate.stage.max()].set_index("tool").slope_um_per_run
    macro("slopeLastMedNine", float(last.median()), "{:.2f}")
    macro("slopeLastMinNine", float(last.min()), "{:.2f}")
    macro("slopeLastMaxNine", float(last.max()), "{:.2f}")
    # Total runs per tool, derived from the result files themselves: under leave-one-tool-out,
    # n_cal of test tool i is the run count of the eight other tools, N - n_i, so
    # N = sum_i n_cal_i / 8 and n_i = N - n_cal_i. Cross-checked against the leave-one-machine-out
    # calibration sizes (N minus the runs of the test machine's three tools).
    lo = dep[(dep.scheme == "S4_loto") & np.isclose(dep.level, level)].set_index("test").n_cal
    n_total = lo.sum() / (len(lo) - 1)
    n_runs = n_total - lo
    lm = mo[(mo.scheme == "S5_lomo") & np.isclose(mo.level, level)].groupby("machine").n_cal.first()
    for m_id, ncal in lm.items():
        own = n_runs[[t for t in n_runs.index if t.startswith(f"M{m_id}T")]].sum()
        assert abs(ncal - (n_total - own)) < 0.5, f"run-count cross-check failed for machine {m_id}"
    macro("nRunsTotalNine", f"{n_total:,.0f}".replace(",", "{,}"))
    macro("nRunsMinNine", float(n_runs.min()), "{:.0f}")
    macro("nRunsMaxNine", float(n_runs.max()), "{:.0f}")
    specs = [("New tool", dep, "S1_same_tools", "interleaved", "ToolSOneInt", "S1 interleaved blocks"),
             ("New tool", dep, "S1_same_tools", "late", "ToolSOneLate", "S1 late-life blocks"),
             ("New tool", dep, "S4_loto", "-", "ToolSFour", "S4 leave-one-tool-out"),
             ("New machine", mo, "S4_loto", "-", "MachSFour", "S4 leave-one-tool-out"),
             ("New machine", mo, "S5_lomo", "-", "MachSFive", "S5 leave-one-machine-out")]
    rows = []
    for shift, d, scheme, block, tag, lab in specs:
        g = d[(d.scheme == scheme) & (d.block == block) & np.isclose(d.level, level)].set_index("test")
        q = g.q
        runs = q / last.reindex(q.index)
        life = 100 * runs / n_runs.reindex(q.index)  # share of the tool's total runs
        macro(f"halfWid{tag}", float(q.mean()), "{:.1f}")
        macro(f"pctAbove{tag}", float(g.pct_above.mean()), "{:.1f}")
        macro(f"pctAboveWorst{tag}", float(g.pct_above.max()), "{:.1f}")
        macro(f"runsHalf{tag}", float(runs.median()), "{:.0f}")
        macro(f"lifeHalf{tag}", float(life.median()), "{:.1f}")
        rows.append([shift, lab, f"{g.coverage.mean():.1f}", f"{g.cov_stage3.mean():.1f}",
                     f"{q.mean():.1f}", f"{g.pct_above.mean():.1f} / {g.pct_above.max():.1f}",
                     f"{runs.median():.0f}", f"{life.median():.1f}"])
    write_table(out / "tab_deploy_guidance.tex",
                "Practical reading of the nine-tool RF intervals at nominal 90\\%. Half-width: "
                "interval half-width in $\\mu$m of flank wear $VB$ (mean over the nine held-out tools). "
                "Above upper bound: share of measurements above the interval, mean / worst tool; "
                "if the upper bound triggers replacement, these are the cases in which the tool is "
                "more worn than the interval admits. Runs given up: half-width divided by each tool's "
                "own final-third wear rate, median over tools; this is the tool life sacrificed when "
                "replacement is triggered by the upper bound rather than by the point prediction. "
                "Tool life given up: each tool's runs given up divided by its total number of runs, "
                "median over tools",
                "tab:deploy_guidance",
                "Shift & Calibration source & Coverage (\\%) & Final third (\\%) & Half-width "
                "($\\mu$m) & Above upper bound (\\%) & Runs given up & Tool life given up (\\%)",
                rows, "llrrrrrr", wide=True)


def point_accuracy(results, out):
    d = load(results, "t2_point_accuracy.csv")
    if d is None:
        return
    d = d[d.model.isin(["rf", "cut_index"])]  # window-RF / CNN are out of scope
    g = d.groupby(["model", "protocol"]).mae.mean().unstack().loc[["rf", "cut_index"]]
    per = d.pivot_table(index=["protocol", "test"], columns="model", values="mae")
    names = {"rf": "RF", "cut_index": "Wear curve"}
    fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    tests = sorted(d.test.unique())
    rows = []
    for prot, pname in [("chrono", "Chronological"), ("v2", "Interleaved")]:
        for m in ["rf", "cut_index"]:
            rows.append([pname, names[m]] + [fmt(per.loc[(prot, c), m]) for c in tests]
                        + [fmt(g.loc[m, prot])])
            for c in tests:  # per-cutter MAE macros, e.g. \maeRFChronoCOne
                ctag = c.replace("1", "One").replace("4", "Four").replace("6", "Six")
                macro(f"mae{ {'rf': 'RF', 'cut_index': 'CutIdx'}[m] }"
                      f"{ {'chrono': 'Chrono', 'v2': 'Ilv'}[prot] }{ctag}".replace(" ", ""),
                      per.loc[(prot, c), m])
    for prot in g.columns:  # how many held-out cutters the RF beats the index on
        p = per.loc[prot]
        ptag = {"chrono": "Chrono", "v2": "Ilv"}[prot]
        macro(f"nRFbeatsIdx{ptag}", str(int((p.rf < p.cut_index).sum())))
    for m, r in g.iterrows():
        key = {"rf": "RF", "cut_index": "CutIdx"}[m]
        for prot in g.columns:
            ptag = {"chrono": "Chrono", "v2": "Ilv"}[prot]
            if not pd.isna(r[prot]):
                macro(f"mae{key}{ptag}", r[prot])
    write_table(out / "tab_point_accuracy.tex",
                "MAE ($\\mu$m) on each held-out PHM2010 tool and the tool-macro mean; each fold "
                "uses the other two tools as sources.",
                "tab:point_accuracy",
                "Split & Model & " + " & ".join(tests) + " & Mean", rows,
                "ll" + "r" * (len(tests) + 1))


def warning(results, out):
    d = load(results, "t2_warning_timing.csv")
    if d is None:
        return
    rules = ["point_150", "tuned", "tuned_3consec"]

    def cell(r):
        if pd.isna(r.first_warning):
            return "none"
        return f"{int(r.first_warning)} ({int(r.delay_cuts):+d})".replace("-", "$-$")

    rows = []
    for prot in ["chrono", "v2"]:
        dp = d[d.protocol == prot]
        for test, g in dp.groupby("test"):
            r = {(m, rule): g[(g.model == m) & (g.rule == rule)].iloc[0]
                 for m in ["rf", "cut_index"] for rule in rules}
            rows.append([{"chrono": "Chrono.", "v2": "Interleaved"}[prot], test,
                         int(r[("rf", "point_150")].first_crossing)]
                        + [cell(r[(m, rule)]) for m in ["rf", "cut_index"] for rule in rules]
                        + [f"{r[('rf', 'point_150')].auc:.3f}", f"{r[('cut_index', 'point_150')].auc:.3f}"])
            for m, mk in [("rf", "RF"), ("cut_index", "CutIdx")]:
                t = r[(m, "tuned")]
                tag = {"chrono": "Chrono", "v2": "Ilv"}[prot] + mk + test.replace("C", "C") \
                    .replace("1", "One").replace("4", "Four").replace("6", "Six")
                macro(f"auc{tag}", t.auc, "{:.3f}")
                macro(f"warn{tag}", "none" if pd.isna(t.first_warning) else str(int(t.first_warning)))
                if not pd.isna(t.delay_cuts):
                    macro(f"delay{tag}", str(int(t.delay_cuts)))
                    macro(f"absDelay{tag}", str(abs(int(t.delay_cuts))))  # for "N cuts before/after"
    write_table(out / "tab_warning.tex",
                "First-warning cut (lead $-$ / delay $+$ relative to the measured crossing) at the "
                "illustrative 150~$\\mu$m wear limit, PHM2010. RF: random forest; Curve: quadratic wear "
                "curve fitted to cut number (no sensors). Tuned cutoffs minimise false positives plus "
                "5 $\\times$ false negatives on source-only out-of-tool predictions; ``3c'' requires "
                "three consecutive exceedances. AUC: per-cut area under the receiver operating "
                "characteristic curve for wear $\\geq$150~$\\mu$m.",
                "tab:warning",
                "Split & Tool & Crossing & RF 150 & RF tuned & RF tuned 3c & Curve 150 & Curve tuned "
                "& Curve tuned 3c & AUC RF & AUC curve",
                rows, "llrllllllrr", wide=True)


def warning_cost(results, out):
    """Decision cost of first warnings (warning_cost.py output); main window W = 30 cuts."""
    d = load(results, "t2_warning_cost.csv")
    if d is None:
        return
    W0 = 30
    rules = [("tuned", "Tuned"), ("tuned_3consec", "TunedThree"), ("point_150", "Fixed")]
    models = [("rf", "RF"), ("cut_index", "CutIdx")]
    ptags = {"chrono": "Chrono", "v2": "Ilv"}
    short = {"premature": "P", "timely": "T", "late": "L", "missed": "M"}
    has_conf = "fp_cuts" in d.columns and d.fp_cuts.notna().all()
    m = d[d.window == W0]

    def cell(r):
        return f"{r.cost:.0f} ({short[r['class']]})"

    rows = []
    for prot in ["chrono", "v2"]:
        dp = m[m.protocol == prot]
        for test, g in dp.groupby("test"):
            get = {(mm, rr): g[(g.model == mm) & (g.rule == rr)].iloc[0] for mm, _ in models
                   for rr, _ in rules}
            row = [{"chrono": "Chrono.", "v2": "Interleaved"}[prot], test,
                   int(get[("rf", "tuned")].first_crossing)]
            for mm, _ in models:
                for rr, _ in rules:
                    row.append(cell(get[(mm, rr)]))
            if has_conf:
                for mm, _ in models:
                    t = get[(mm, "tuned")]
                    row.append(f"{int(t.fp_cuts)}/{int(t.fn_cuts)}")
            rows.append(row)
            ctag = test.replace("1", "One").replace("4", "Four").replace("6", "Six")
            for mm, mk in models:
                for rr, rk in rules:
                    t = get[(mm, rr)]
                    macro(f"cost{ptags[prot]}{mk}{ctag}{rk}", t.cost, "{:.0f}")
                    macro(f"class{ptags[prot]}{mk}{ctag}{rk}", t["class"])
        tot = dp.groupby(["model", "rule"]).cost.sum()
        trow = [{"chrono": "Chrono.", "v2": "Interleaved"}[prot], "Total", ""]
        for mm, mk in models:
            for rr, rk in rules:
                trow.append(f"{tot[(mm, rr)]:.0f}")
                macro(f"costTot{ptags[prot]}{mk}{rk}", tot[(mm, rr)], "{:.0f}")
        if has_conf:
            trow += ["", ""]
        rows.append(trow)
        # number of cutters on which the RF's cost is lower than the cut-index curve's
        for rr, rk in rules:
            a = dp[(dp.model == "rf") & (dp.rule == rr)].set_index("test").cost
            b = dp[(dp.model == "cut_index") & (dp.rule == rr)].set_index("test").cost
            macro(f"nRFlowerCost{ptags[prot]}{rk}", str(int((a < b).sum())))
            macro(f"nRFtiesCost{ptags[prot]}{rk}", str(int((a == b).sum())))
    # window sensitivity: does the RF-vs-index ordering of total cost change with W?
    changed = []
    for prot in ["chrono", "v2"]:
        for rr, _ in rules:
            signs = set()
            for W, g in d[(d.protocol == prot) & (d.rule == rr)].groupby("window"):
                s = g.groupby("model").cost.sum()
                signs.add(int(np.sign(s["rf"] - s["cut_index"])))
            if len(signs) > 1:
                changed.append(f"{prot}/{rr}")
    macro("warnWindowMain", str(W0))
    macro("warnWindowsSens", " and ".join(str(w) for w in sorted(d.window.unique()) if w != W0))
    macro("warnOrderSens", "did not change" if not changed else "changed for " + ", ".join(changed))
    header = ("Split & Tool & Crossing & RF tuned & RF tuned 3c & RF 150 & Curve tuned & "
              "Curve tuned 3c & Curve 150")
    align = "llr" + "r" * 6
    if has_conf:
        header += " & RF FP/FN & Curve FP/FN"
        align += "rr"
    write_table(out / "tab_warning_cost.tex",
                f"Decision cost of the first warnings in Table~\\ref{{tab:warning}} (lower is better), "
                f"scored against a window of {W0} cuts ending at the measured 150~$\\mu$m crossing. "
                "Premature (P): cost = cuts of tool life given up before the window; timely (T): 0; "
                "late (L): 5 $\\times$ cuts after the crossing; missed (M): 5 $\\times$ cuts from the "
                "crossing to cut 315. Totals are sums over the three tools."
                + (" FP/FN: per-cut false positives/negatives of the tuned rule." if has_conf else ""),
                "tab:warning_cost", header, rows, align, wide=True)


def ninetool(results, out, sfx="", model_label="RF", mprefix=""):
    rows = []
    panels = [("t3_deploy", "New tool", "Tool"), ("t3_machine_out", "New machine", "Mach"),
              ("t3_matched", "New tool, matched$^{\\ast}$", "Matched")]
    for stem, part, ktag in panels:
        d = load(results, f"{stem}{sfx}.csv")
        if d is None:
            continue
        d = d[np.isclose(d.level, 0.9)]
        # per test tool first (Part A: averaged over the external tools c in each scheme)
        per_tool = d.groupby(["scheme", "block", "test"])[
            ["coverage", "width", "interval_score", "cov_stage3"]].mean()
        g = per_tool.groupby(["scheme", "block"]).agg(["mean", "std", "min"])
        if ktag == "Matched":
            g = g.sort_index(level=["block", "scheme"])  # group rows by block, then scheme
        for (scheme, block), r in g.iterrows():
            lab = SCHEME_LABEL[scheme] + ("" if block == "-" else f" ({block})")
            rows.append([part, lab,
                         f"{r[('coverage', 'mean')]:.1f} $\\pm$ {r[('coverage', 'std')]:.1f}",
                         f"{r[('coverage', 'min')]:.1f}", f"{r[('width', 'mean')]:.1f}",
                         f"{r[('interval_score', 'mean')]:.1f}", f"{r[('cov_stage3', 'mean')]:.1f}"])
            key = mprefix + ktag + \
                {"S1_same_tools": "SOne", "S2_same_machine": "STwo", "S3_other_machine": "SThree",
                 "S4_loto": "SFour", "S4u_loto_unweighted": "SFourU",
                 "S5_lomo": "SFive"}[scheme] + {"interleaved": "Int", "late": "Late", "-": ""}[block]
            macro(f"cov{key}", r[("coverage", "mean")], PCT_UM)
            macro(f"covmin{key}", r[("coverage", "min")], PCT_UM)
            macro(f"wid{key}", r[("width", "mean")], PCT_UM)
            macro(f"is{key}", r[("interval_score", "mean")], PCT_UM)
            macro(f"covLast{key}", r[("cov_stage3", "mean")], PCT_UM)     # final third of tool life
            macro(f"covminLast{key}", r[("cov_stage3", "min")], PCT_UM)
        if ktag in ("Tool", "Mach"):  # point error of the final (all-source) model, per tool then mean
            s4 = d[d.scheme == "S4_loto"].groupby("test").test_mae.mean()
            macro(f"mae{mprefix}{ktag}", s4.mean())
    if rows:
        if any(r[0].startswith("New tool, matched") for r in rows):
            rows.append(["\\multicolumn{6}{p{0.95\\linewidth}}{\\footnotesize $^{\\ast}$Matched design: "
                         f"for each test tool and each external tool $c$, one {model_label} is fit on the "
                         "other seven tools minus a calibration block, and its test predictions are "
                         "calibrated with either that block (S1) or all of tool $c$ (S2 if $c$ shares the "
                         "test tool's machine, S3 otherwise). Training sets therefore differ from the "
                         "upper panels.}"])
        write_table(out / f"tab_ninetool{sfx}.tex",
                    f"Nine-tool dataset, {model_label} point model, nominal 90\\%: coverage mean $\\pm$ SD "
                    "and worst tool over the nine held-out tools (new tool: 9 folds; new machine: "
                    "3 machines $\\times$ 3 tools). Width and interval score in $\\mu$m (flank wear $VB$). "
                    "Final-third coverage is computed on the last third of each test tool's recordings.",
                    f"tab:ninetool{sfx}",
                    "Shift & Calibration scheme & Coverage (\\%) & Worst tool (\\%) & Width & Interval score "
                    "& Final-third cov. (\\%)",
                    rows, "llrrrrr", wide=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../manuscript/generated")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    phm_matched(res, out)
    point_accuracy(res, out)
    warning(res, out)
    warning_cost(res, out)
    ninetool(res, out)
    ninetool(res, out, sfx="_hgb", model_label="HGB", mprefix="Hgb")
    wilcoxon_macros(res)
    wilcoxon_macros(res, sfx="_hgb", mprefix="Hgb")
    spearman_macros(res)
    deployment_guidance(res, out)
    derived_macros()
    stale = out / "tab_ninetool_matched.tex"  # merged into tab_ninetool.tex
    if stale.exists():
        stale.unlink()
    (out / "numbers.tex").write_text(
        "% AUTO-GENERATED by make_tables.py - do not edit by hand\n" +
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n",
        encoding="utf-8")
    print(f"wrote numbers.tex ({len(MACROS)} macros)")


if __name__ == "__main__":
    main()

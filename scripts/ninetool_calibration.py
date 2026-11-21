"""T3: calibration-source experiment on the nine-tool milling dataset
(Stiehl/Denkena et al., Mendeley Data 10.17632/zpxs87bjt8).

Input : nine_tool_features.csv  (6418 rows; 36 force features; produced by cell 3 of
        wear-reliability-nine-tool-independent-validation.ipynb)
Output: <out>/t3_matched.csv        Part A - identical predictions, calibration source varied
        <out>/t3_deploy.csv         Part B - realistic procedures, S1 (held-out block) vs S4 (LOTO)
        <out>/t3_machine_out.csv    Part C - leave-one-machine-out test, S1 vs S4 (LOTO) vs S5 (LOMO)
        <out>/t3_summary.txt        macro tables (paired tests: stats_wilcoxon.py)
        HGB run (--model hgb): same files with suffix _hgb
        <out>/wear_rate_ninetool.csv

Design
------
Outer loop: every one of the 9 tools is the test tool once. Nothing from the test
tool is used for fitting, calibration or tuning.

Part A (matched, the causal comparison). For test tool T and each other tool c:
  fit ONE RF on the remaining 7 tools minus a calibration block of each,
  predict T once, then calibrate with
    S1  residuals on the held-out block of the 7 training tools
        (block = 'interleaved' 20% or 'late' = final 20% of life)
    S2  residuals on all of tool c, if c is on T's machine
    S3  residuals on all of tool c, if c is on another machine
  -> 9 x 8 x 2 = 144 fits.

Part B (deployment). For test tool T, source = the other 8 tools:
    S1  fit on 80% of each source tool, calibrate on the held-out 20% block
    S4  fit on 100% of source; residuals = leave-one-source-tool-out (LOTO)
    S4u same as S4 but unweighted quantile (shows why tool weighting matters)

Part C (new machine). Test = all 3 tools of machine m; source = the 6 other tools.
    S1 (blocks), S4 LOTO (groups = tool), S5 LOMO (groups = machine: train on one
    source machine, residuals on the other, both directions).

All quantiles weight each calibration tool equally unless marked 'unweighted'.

Kaggle:  !python ninetool_calibration.py --features /kaggle/working/nine_tool_validation/nine_tool_features.csv --out /kaggle/working/results
Self-test (synthetic, no claims):  python ninetool_calibration.py --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from calib import evaluate_all_levels, group_equal_weights, stage_index, stage_wear_rates

FEATURE_NAMES = ["mean", "std", "rms", "min", "max", "ptp", "skew", "kurtosis",
                 "energy_0", "energy_1", "energy_2", "energy_3"]
FEATS = [f"{ch}_{n}" for ch in ["Fx", "Fy", "Fz"] for n in FEATURE_NAMES]
N_TREES = 300
BLOCKS = ("interleaved", "late")
MODEL = "rf"  # "rf" (primary) or "hgb" (model-agnosticism check); set by --model


def make_rf(seed=42):
    """Point model used by every scheme (name kept for history)."""
    if MODEL == "hgb":
        # sklearn defaults except max_iter; early_stopping='auto' is inactive below
        # 10,000 training samples (all fits here use < 6,000).
        return HistGradientBoostingRegressor(max_iter=N_TREES, random_state=seed)
    return RandomForestRegressor(n_estimators=N_TREES, max_features="sqrt",
                                 min_samples_leaf=3, random_state=seed, n_jobs=-1)


def block_mask(d, kind):
    """Boolean calibration-block mask computed separately inside each tool.
    interleaved: life split into 20 contiguous chunks, chunks 3, 8, 13, 18 (20%)
    late       : final 20% of the tool's runs."""
    mask = np.zeros(len(d), dtype=bool)
    for _, idx in d.groupby("physical_tool").indices.items():
        n = len(idx)
        rank = d["run"].to_numpy()[idx].argsort().argsort()  # 0..n-1 in run order
        if kind == "late":
            mask[idx] = rank >= int(np.floor(0.8 * n))
        else:
            chunk = (rank * 20) // n + 1
            mask[idx] = np.isin(chunk, [3, 8, 13, 18])
    return mask


def fit_predict(train, *targets):
    m = make_rf().fit(train[FEATS], train["wear"])
    return [m.predict(t[FEATS]) for t in targets]


def group_out_residuals(src, group_col):
    """Out-of-group residuals: each group predicted by a model fit on the others."""
    res = np.empty(len(src))
    for g in src[group_col].unique():
        tr, ca = src[group_col] != g, src[group_col] == g
        res[ca.to_numpy()] = np.abs(src.loc[ca, "wear"].to_numpy()
                                    - fit_predict(src[tr], src[ca])[0])
    return res


def record(rows, T, yhat, res, groups, weighted=True, **tags):
    for rec in evaluate_all_levels(T["wear"].to_numpy(), yhat, T["run"].to_numpy(), res,
                                   group_equal_weights(groups) if weighted else None):
        rec.update(test=T["physical_tool"].iloc[0], machine=int(T["machine"].iloc[0]),
                   test_mae=mean_absolute_error(T["wear"], yhat), **tags)
        rows.append(rec)


def part_a(df, tools, machine_of):
    rows = []
    for t in tools:
        T = df[df.physical_tool == t].sort_values("run")
        for c in [x for x in tools if x != t]:
            C = df[df.physical_tool == c]
            tr_tools = df[~df.physical_tool.isin([t, c])].reset_index(drop=True)
            ext = "S2_same_machine" if machine_of[c] == machine_of[t] else "S3_other_machine"
            for blk in BLOCKS:
                bm = block_mask(tr_tools, blk)
                cal_in = tr_tools[bm]
                yT, y_in, y_c = fit_predict(tr_tools[~bm], T, cal_in, C)
                record(rows, T, yT, np.abs(cal_in.wear.to_numpy() - y_in),
                       cal_in.physical_tool, scheme="S1_same_tools", block=blk, cal_tool=c)
                record(rows, T, yT, np.abs(C.wear.to_numpy() - y_c), C.physical_tool,
                       scheme=ext, block=blk, cal_tool=c)
        print(f"  Part A done: {t}", flush=True)
    return pd.DataFrame(rows)


def deploy_schemes(src, T_list, rows, part):
    """S1 (both blocks), S4 LOTO (weighted + unweighted) and, if >1 machine-group
    structure is requested, S5 LOMO. Appends rows for every test tool in T_list."""
    src = src.reset_index(drop=True)
    for blk in BLOCKS:
        bm = block_mask(src, blk)
        preds = fit_predict(src[~bm], src[bm], *T_list)
        res = np.abs(src[bm].wear.to_numpy() - preds[0])
        for T, yT in zip(T_list, preds[1:]):
            record(rows, T, yT, res, src[bm].physical_tool, scheme="S1_same_tools",
                   block=blk, part=part)
    final = fit_predict(src, *T_list)
    loto = group_out_residuals(src, "physical_tool")
    for T, yT in zip(T_list, final):
        record(rows, T, yT, loto, src.physical_tool, scheme="S4_loto", block="-", part=part)
        record(rows, T, yT, loto, src.physical_tool, weighted=False,
               scheme="S4u_loto_unweighted", block="-", part=part)
    if part == "machine_out":
        lomo = group_out_residuals(src, "machine")
        for T, yT in zip(T_list, final):
            record(rows, T, yT, lomo, src.physical_tool, scheme="S5_lomo", block="-", part=part)


def part_b(df, tools):
    rows = []
    for t in tools:
        T = df[df.physical_tool == t].sort_values("run")
        deploy_schemes(df[df.physical_tool != t], [T], rows, "tool_out")
        print(f"  Part B done: {t}", flush=True)
    return pd.DataFrame(rows)


def part_c(df):
    rows = []
    for m in sorted(df.machine.unique()):
        T_list = [g.sort_values("run") for _, g in df[df.machine == m].groupby("physical_tool")]
        deploy_schemes(df[df.machine != m], T_list, rows, "machine_out")
        print(f"  Part C done: machine {m}", flush=True)
    return pd.DataFrame(rows)


def macro(d, keys):
    """Average per test tool first (over cal tools / combos), then mean +/- SD over tools."""
    cols = ["coverage", "width", "interval_score", "cov_stage1", "cov_stage2", "cov_stage3"]
    per_tool = d.groupby(keys + ["test"])[cols].mean().reset_index()
    agg = per_tool.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return per_tool, agg.reset_index()


def wear_rates(df):
    rows = []
    for t, d in df.groupby("physical_tool"):
        d = d.sort_values("run")
        for s, slope in stage_wear_rates(d.run, d.wear).items():
            rows.append({"tool": t, "machine": int(d.machine.iloc[0]), "stage": s,
                         "slope_um_per_run": slope})
    return pd.DataFrame(rows)


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m, n in zip([1, 2, 3], [120, 160, 120]):
        mshift = rng.normal(0, 1, 36)
        for k in range(3):
            tool = 3 * (m - 1) + k + 1
            run = np.arange(1, n + 1)
            rate = rng.uniform(.8, 1.3) * 150 / n
            wear = 3 + rate * run + rng.normal(0, 2, n)
            F = (np.outer(wear, rng.normal(1, .3, 36)) + 30 * mshift
                 + rng.normal(0, 10, (n, 36)))
            d = pd.DataFrame(F, columns=FEATS)
            d["machine"], d["tool"], d["run"], d["wear"] = m, tool, run, wear
            d["physical_tool"] = f"M{m}T{tool}"
            out.append(d)
    return pd.concat(out, ignore_index=True)


def main():
    global N_TREES, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--skip-a", action="store_true", help="skip Part A (slowest)")
    ap.add_argument("--only-b", action="store_true", help="run only Part B (new-tool rows)")
    ap.add_argument("--model", choices=["rf", "hgb"], default="rf")
    a = ap.parse_args()
    MODEL = a.model
    sfx = "" if MODEL == "rf" else f"_{MODEL}"  # RF keeps the original file names
    if a.self_test:
        N_TREES = 30
        df = synthetic()
    else:
        df = pd.read_csv(a.features)
        assert len(df) == 6418 and df.physical_tool.nunique() == 9
    assert all(c in df.columns for c in FEATS)
    df = df.sort_values(["physical_tool", "run"]).reset_index(drop=True)
    tools = sorted(df.physical_tool.unique())
    machine_of = df.groupby("physical_tool").machine.first().to_dict()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    def say(s=""):
        print(s)
        lines.append(str(s))

    pd.set_option("display.width", 220)
    say(f"Model: {MODEL}")
    # Part B first: it holds the primary new-tool rows, so it is saved even if a
    # time-limited session stops during Part A or C.
    B = part_b(df, tools)
    B.to_csv(out / f"t3_deploy{sfx}.csv", index=False)
    _, aggB = macro(B[np.isclose(B.level, 0.9)], ["scheme", "block"])
    say("\n=== Part B: tool-held-out deployment, nominal 90% ===")
    say(aggB.round(2).to_string(index=False))
    if not (a.only_b or a.skip_a):
        A = part_a(df, tools, machine_of)
        A.to_csv(out / f"t3_matched{sfx}.csv", index=False)
        _, agg = macro(A[np.isclose(A.level, 0.9)], ["block", "scheme"])
        say("\n=== Part A: matched predictions, nominal 90% (per-tool mean, then mean/SD over 9 tools) ===")
        say(agg.round(2).to_string(index=False))
    if not a.only_b:
        C = part_c(df)
        C.to_csv(out / f"t3_machine_out{sfx}.csv", index=False)
        _, aggC = macro(C[np.isclose(C.level, 0.9)], ["scheme", "block"])
        say("\n=== Part C: leave-one-machine-out, nominal 90% ===")
        say(aggC.round(2).to_string(index=False))
    say("\nPaired tests: run stats_wilcoxon.py on these CSVs (exact, tool- and machine-level).")
    if MODEL == "rf":
        wear_rates(df).to_csv(out / "wear_rate_ninetool.csv", index=False)
    (out / f"t3_summary{sfx}.txt").write_text("\n".join(lines), encoding="utf-8")
    json.dump({"model": MODEL, "n_trees_or_iter": N_TREES, "blocks": BLOCKS,
               "only_b": a.only_b, "skip_a": a.skip_a, "self_test": a.self_test},
              open(out / f"t3_config{sfx}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

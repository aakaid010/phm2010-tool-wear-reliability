"""T1 (RF part) + T2 + wear-rate table for PHM2010.

Input : phm_features.csv  (945 x 87; output of phm2010-01-data-exploration)
        optional phm_sequences.npz (for the window-RF row of Table 1)
Output: <out>/t1_matched_rf.csv          one row per protocol x test x train x cal-source x level
        <out>/t1_matched_rf_summary.csv  mean over the 6 directed combinations
        <out>/t2_point_accuracy.csv      Table 1 (chronological + interleaved split, RF / window-RF / cut-index)
        <out>/t2_warning_timing.csv      Table 4 incl. cut-index rows
        <out>/wear_rate_phm.csv          per-cutter, per-stage wear slope (um/cut)

Kaggle:  !python phm2010_rf_chrono.py --features /kaggle/input/.../phm_features.csv \
             --sequences /kaggle/working/phm_sequences.npz --out /kaggle/working/results
Local self-test (synthetic data, no claims):  python phm2010_rf_chrono.py --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

from calib import LEVELS, evaluate_all_levels, stage_wear_rates

CUTTERS = [1, 4, 6]
TAU = 150.0          # illustrative wear threshold (um) - same as MST Table 4
FN_COST = 5.0        # FP + 5 FN tuning criterion
PERSIST = 3          # consecutive exceedances
ILV_TRAIN = [1, 3, 5, 6, 7, 9, 10, 11, 13, 15]
ILV_UNUSED = [4, 12]  # interleaved split: blocks fitted on neither model nor calibration (RF has no early stopping)
ILV_CAL = [2, 8, 14]
N_TREES = 300
MODEL = "rf"  # "rf" (primary) or "hgb" (model-agnosticism check); set by --model


def make_rf(seed=42):
    """Point model used by every experiment in this file (name kept for history)."""
    if MODEL == "hgb":
        # sklearn defaults except max_iter; early_stopping='auto' is inactive below
        # 10,000 training samples, so no internal validation split is taken here.
        return HistGradientBoostingRegressor(max_iter=N_TREES, random_state=seed)
    return RandomForestRegressor(n_estimators=N_TREES, max_features="sqrt",
                                 min_samples_leaf=3, random_state=seed, n_jobs=-1)


def split_masks(protocol, cut):
    cut = np.asarray(cut)
    if protocol == "chrono":
        return {"train": cut <= 220, "val": (cut >= 221) & (cut <= 260), "cal": cut >= 261}
    block = (cut - 1) // 21 + 1
    return {"train": np.isin(block, ILV_TRAIN), "val": np.isin(block, ILV_UNUSED),
            "cal": np.isin(block, ILV_CAL)}


# ----------------------------------------------------------------------------
# T1: matched calibration-source experiment (identical test predictions)
# ----------------------------------------------------------------------------
def run_matched(df, feats):
    rows = []
    for protocol in ["chrono", "v2"]:
        for T in CUTTERS:
            dT = df[df.cutter == T].sort_values("cut")
            for A in [c for c in CUTTERS if c != T]:
                B = [c for c in CUTTERS if c not in (T, A)][0]
                dA = df[df.cutter == A].sort_values("cut")
                dB = df[df.cutter == B].sort_values("cut")
                mA = split_masks(protocol, dA.cut)
                mB = split_masks(protocol, dB.cut)
                # RF needs no early stopping; validation cuts are simply unused,
                # exactly as in the MST interleaved matched RF.
                model = make_rf().fit(dA.loc[mA["train"], feats], dA.loc[mA["train"], "wear_max"])
                yhatT = model.predict(dT[feats])

                def resid(d, m=None):
                    d = d if m is None else d[m]
                    return np.abs(d["wear_max"].to_numpy() - model.predict(d[feats]))

                sources = {
                    "same_cutter": (A, resid(dA, mA["cal"])),
                    "other_cutter_same_range": (B, resid(dB, mB["cal"])),
                    "other_cutter_full_life": (B, resid(dB)),
                }
                for name, (cal_cutter, r) in sources.items():
                    for rec in evaluate_all_levels(dT.wear_max.to_numpy(), yhatT, dT.cut.to_numpy(), r):
                        rec.update(protocol=protocol, test=f"C{T}", train=f"C{A}",
                                   cal_cutter=f"C{cal_cutter}", cal_source=name,
                                   test_mae=mean_absolute_error(dT.wear_max, yhatT))
                        rows.append(rec)
    out = pd.DataFrame(rows)
    num = ["coverage", "width", "interval_score", "pct_below", "pct_above",
           "cov_stage1", "cov_stage2", "cov_stage3", "n_cal"]
    summary = out.groupby(["protocol", "cal_source", "level"])[num].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return out, summary.reset_index()


# ----------------------------------------------------------------------------
# T2: two-source point accuracy, cut-index control, warning timing
# ----------------------------------------------------------------------------
def cut_index_model(train_df):
    coef = np.polyfit(train_df.cut.to_numpy(float), train_df.wear_max.to_numpy(float), 2)
    return lambda d: np.polyval(coef, d.cut.to_numpy(float))


def window_features(seq_path, df_cuts):
    """84 features per processed neural window (MST Sec. 2.2), returned per window."""
    with np.load(seq_path) as z:
        X, cutter, cut = z["X"], z["cutter"].astype(int), z["cut"].astype(int)
    fs = 12500.0
    bands = [(0, 250), (250, 1000), (1000, 2500), (2500, 6250)]
    feats = []
    for w in X:
        row = []
        for ch in range(w.shape[1]):
            x = w[:, ch].astype(float)
            row += [x.mean(), x.std(), np.sqrt(np.mean(x ** 2)), x.min(), x.max(), np.ptp(x),
                    skew(x), kurtosis(x)]
            f, p = welch(x, fs=fs, nperseg=512)
            df_ = f[1] - f[0]
            for i, (lo, hi) in enumerate(bands):
                m = (f >= lo) & ((f <= hi) if i == len(bands) - 1 else (f < hi))
                row.append(p[m].sum() * df_)
        feats.append(row)
    W = pd.DataFrame(feats, columns=[f"w{i}" for i in range(len(feats[0]))])
    W["cutter"], W["cut"] = cutter, cut
    return W.merge(df_cuts[["cutter", "cut", "wear_max"]], on=["cutter", "cut"])


def two_source_predictions(df, feats, protocol, T, kind, W=None):
    """Predictions for test cutter T from a model on both source cutters."""
    src = df[(df.cutter != T)]
    m = split_masks(protocol, src.cut)
    tr = src[m["train"]]
    dT = df[df.cutter == T].sort_values("cut")
    if kind == "rf":
        return dT, make_rf().fit(tr[feats], tr.wear_max).predict(dT[feats])
    if kind == "cut_index":
        return dT, cut_index_model(tr)(dT)
    if kind == "window_rf":
        wf = [c for c in W.columns if c.startswith("w")]
        Ws = W[W.cutter != T]
        Ws = Ws[split_masks(protocol, Ws.cut)["train"]]
        model = make_rf().fit(Ws[wf], Ws.wear_max)
        WT = W[W.cutter == T].copy()
        WT["p"] = model.predict(WT[wf])
        p = WT.groupby("cut").p.mean().reindex(dT.cut).to_numpy()
        return dT, p
    raise ValueError(kind)


def single_source_predictor(df, feats, protocol, A, kind):
    dA = df[df.cutter == A]
    tr = dA[split_masks(protocol, dA.cut)["train"]]
    if kind == "rf":
        m = make_rf().fit(tr[feats], tr.wear_max)
        return lambda d: m.predict(d[feats])
    return cut_index_model(tr)


def tune_cutoff(df, feats, protocol, T, kind):
    """Cutoff c minimising FP + 5 FN on source-only out-of-cutter predictions:
    model on source A predicts all 315 cuts of source B, and vice versa."""
    A, B = [c for c in CUTTERS if c != T]
    preds, labels = [], []
    for s, o in [(A, B), (B, A)]:
        f = single_source_predictor(df, feats, protocol, s, kind)
        d = df[df.cutter == o].sort_values("cut")
        preds.append(f(d))
        labels.append(d.wear_max.to_numpy() >= TAU)
    p, lab = np.concatenate(preds), np.concatenate(labels)
    cands = np.unique(np.concatenate([p, [TAU]]))
    costs = np.array([np.sum((p >= c) & ~lab) + FN_COST * np.sum((p < c) & lab) for c in cands])
    best = np.flatnonzero(costs == costs.min())
    return float(cands[best[np.argmin(np.abs(cands[best] - TAU))]])  # ties -> nearest to TAU


def first_index(flags, cuts):
    i = np.flatnonzero(flags)
    return int(cuts[i[0]]) if len(i) else None


def persistent(flags, k):
    out = np.zeros_like(flags)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= k
    return out


def warning_rows(dT, yhat, cutoff, protocol, kind):
    cuts = dT.cut.to_numpy()
    y = dT.wear_max.to_numpy()
    cross = first_index(y >= TAU, cuts)
    rules = {"point_150": yhat >= TAU, "tuned": yhat >= cutoff,
             "tuned_3consec": persistent(yhat >= cutoff, PERSIST)}
    rows = []
    worn = y >= TAU
    for rule, flags in rules.items():
        fw = first_index(flags, cuts)
        pre = flags & (cuts < cross) if cross is not None else flags
        episodes = int(np.sum(pre & ~np.r_[False, pre[:-1]]))
        # per-cut confusion counts of this rule on the test cutter (added for the IJPHM
        # warning-cost table; requires a rerun - not in the 20 Nov 2026 results)
        fp_cuts, fn_cuts = int(np.sum(flags & ~worn)), int(np.sum(~flags & worn))
        rows.append({"protocol": protocol, "model": kind, "test": f"C{int(dT.cutter.iloc[0])}",
                     "rule": rule, "cutoff": cutoff if rule != "point_150" else TAU,
                     "first_crossing": cross, "first_warning": fw,
                     "delay_cuts": (fw - cross) if (fw is not None and cross is not None) else None,
                     "pre_crossing_episodes": episodes,
                     "fp_cuts": fp_cuts, "fn_cuts": fn_cuts,
                     "auc": roc_auc_score(y >= TAU, yhat) if 0 < (y >= TAU).sum() < len(y) else np.nan})
    return rows


def run_t2(df, feats, seq_path=None):
    W = window_features(seq_path, df) if seq_path else None
    kinds = ["rf", "cut_index"] + (["window_rf"] if W is not None else [])
    acc, warn = [], []
    for protocol in ["chrono", "v2"]:
        for T in CUTTERS:
            for kind in kinds:
                dT, yhat = two_source_predictions(df, feats, protocol, T, kind, W)
                acc.append({"protocol": protocol, "model": kind, "test": f"C{T}",
                            "mae": mean_absolute_error(dT.wear_max, yhat),
                            "rmse": float(np.sqrt(np.mean((dT.wear_max - yhat) ** 2)))})
                if kind in ("rf", "cut_index"):
                    cutoff = tune_cutoff(df, feats, protocol, T, kind)
                    warn += warning_rows(dT, yhat, cutoff, protocol, kind)
    return pd.DataFrame(acc), pd.DataFrame(warn)


def wear_rates(df):
    rows = []
    for c in CUTTERS:
        d = df[df.cutter == c].sort_values("cut")
        for s, slope in stage_wear_rates(d.cut, d.wear_max).items():
            rows.append({"cutter": f"C{c}", "stage": s, "slope_um_per_cut": slope})
    return pd.DataFrame(rows)


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c, rate in zip(CUTTERS, [0.35, 0.45, 0.55]):
        cut = np.arange(1, 316)
        wear = 40 + rate * cut + 4e-4 * (rate / .45) * cut ** 2 + rng.normal(0, 3, 315)
        F = np.outer(wear, rng.normal(1, .2, 84)) * (1 + .1 * (c - 4)) + rng.normal(0, 15, (315, 84))
        d = pd.DataFrame(F, columns=[f"f{i}" for i in range(84)])
        d["cutter"], d["cut"], d["wear_max"] = c, cut, wear
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    global N_TREES, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--sequences", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", choices=["rf", "hgb"], default="rf")
    ap.add_argument("--matched-only", action="store_true",
                    help="run only the T1 matched calibration experiment")
    a = ap.parse_args()
    MODEL = a.model
    if a.self_test:
        N_TREES = 50
        df = synthetic()
    else:
        df = pd.read_csv(a.features)
        assert df.shape == (945, 87), df.shape
    feats = [c for c in df.columns if c not in ("cutter", "cut", "wear_max")]
    assert len(feats) == 84
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m, ms = run_matched(df, feats)
    m.to_csv(out / f"t1_matched_{MODEL}.csv", index=False)
    ms.to_csv(out / f"t1_matched_{MODEL}_summary.csv", index=False)
    if a.matched_only:
        s90 = ms[np.isclose(ms.level, 0.9)][["protocol", "cal_source", "coverage_mean",
                                              "width_mean", "interval_score_mean", "cov_stage3_mean"]]
        print(f"\n=== T1 matched {MODEL.upper()}, nominal 90% (mean of 6 directed combos) ===")
        print(s90.round(2).to_string(index=False))
        json.dump({"model": MODEL, "n_iter_or_trees": N_TREES, "matched_only": True,
                   "self_test": a.self_test},
                  open(out / f"phm2010_{MODEL}_config.json", "w"), indent=2)
        return
    acc, warn = run_t2(df, feats, a.sequences)
    acc.to_csv(out / "t2_point_accuracy.csv", index=False)
    warn.to_csv(out / "t2_warning_timing.csv", index=False)
    wear_rates(df).to_csv(out / "wear_rate_phm.csv", index=False)
    json.dump({"n_trees": N_TREES, "tau": TAU, "fn_cost": FN_COST, "persist": PERSIST,
               "self_test": a.self_test}, open(out / "phm2010_rf_config.json", "w"), indent=2)

    pd.set_option("display.width", 200)
    s90 = ms[ms.level == 0.9][["protocol", "cal_source", "coverage_mean", "width_mean",
                               "interval_score_mean", "cov_stage3_mean"]]
    print("\n=== T1 matched RF, nominal 90% (mean of 6 directed combos) ===")
    print(s90.round(2).to_string(index=False))
    print("Sanity: interleaved same/other should reproduce MST Table 3 (14.87/84.18 %, 9.51/84.45 um).")
    print("\n=== T2 point accuracy (macro MAE) ===")
    print(acc.groupby(["protocol", "model"]).mae.mean().round(2).to_string())
    print("\n=== T2 first warnings @150 um (chronological) ===")
    print(warn[warn.protocol == "chrono"].to_string(index=False))


if __name__ == "__main__":
    main()

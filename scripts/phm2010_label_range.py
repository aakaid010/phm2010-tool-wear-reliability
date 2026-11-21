"""Label-range reference intervals for the PHM2010 matched experiment (step 1b).

For each protocol (chrono, v2) and each of the 6 directed combinations (test T,
training cutter A), the reference interval is a FIXED band built only from the
training labels of A (the same cuts used to fit the matched single-source RF):
    minmax : [min, max] of A's training wear labels
    p05_p95: [5th, 95th percentile] of A's training wear labels
It does not depend on any prediction. Reported: width, coverage of T's 315
labels, interval score at nominal 90% (alpha = 0.10), and stage coverage.

  !python phm2010_label_range.py --features <phm_features.csv> --out /kaggle/working/results
  python phm2010_label_range.py --self-test
Writes <out>/t1_label_range.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from calib import stage_index
from phm2010_rf_chrono import CUTTERS, split_masks, synthetic


def band_metrics(y, lo, hi, cut, alpha=0.10):
    below, above = y < lo, y > hi
    score = (hi - lo) + (2 / alpha) * (lo - y) * below + (2 / alpha) * (y - hi) * above
    st = stage_index(cut)
    hit = ~below & ~above
    out = {"lower": lo, "upper": hi, "width": hi - lo, "coverage": 100 * hit.mean(),
           "interval_score": float(score.mean()),
           "pct_below": 100 * below.mean(), "pct_above": 100 * above.mean()}
    out.update({f"cov_stage{s}": 100 * hit[st == s].mean() for s in (1, 2, 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    df = synthetic() if a.self_test else pd.read_csv(a.features, usecols=["cutter", "cut", "wear_max"])
    rows = []
    for protocol in ["chrono", "v2"]:
        for T in CUTTERS:
            dT = df[df.cutter == T].sort_values("cut")
            for A in [c for c in CUTTERS if c != T]:
                dA = df[df.cutter == A].sort_values("cut")
                yA = dA.loc[split_masks(protocol, dA.cut)["train"], "wear_max"].to_numpy()
                for ref, (lo, hi) in {"minmax": (yA.min(), yA.max()),
                                      "p05_p95": tuple(np.percentile(yA, [5, 95]))}.items():
                    r = band_metrics(dT.wear_max.to_numpy(), lo, hi, dT.cut.to_numpy())
                    r.update(protocol=protocol, test=f"C{T}", train=f"C{A}", reference=ref,
                             n_train_labels=len(yA), level=0.9)
                    rows.append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "t1_label_range.csv", index=False)
    pd.set_option("display.width", 200)
    print(d[["protocol", "test", "train", "reference", "lower", "upper", "width", "coverage",
             "interval_score", "cov_stage3"]].round(2).to_string(index=False))
    print("\nMean over 6 combinations:")
    print(d.groupby(["protocol", "reference"])[["width", "coverage", "interval_score", "cov_stage3"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()

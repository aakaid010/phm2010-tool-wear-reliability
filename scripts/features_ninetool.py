"""Nine-tool per-recording features (36 = 3 force channels x 12), ported unchanged from cells
2-3 of notebooks/wear-reliability-nine-tool-independent-validatio.ipynb (the Kaggle run that
produced nine_tool_features.csv).

Input : the unpacked Mendeley Data record doi:10.17632/zpxs87bjt8.3 (Stiehl, 2023): filelist.csv
        and the 6,418 HDF5 recordings it lists (force_sensor_x/y/z at 25 kHz).
Output: nine_tool_features.csv (one row per recording; metadata + 36 feature columns).

  python features_ninetool.py --data <dataset dir> --out ../data/nine_tool_features.csv
  python features_ninetool.py --data <dir> --out test.csv --limit 5    # quick check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

CHANNELS = {"Fx": "signals_sensor/force_sensor_x", "Fy": "signals_sensor/force_sensor_y",
            "Fz": "signals_sensor/force_sensor_z"}
FS = 25000
BANDS = [(0, 500), (500, 1500), (1500, 5000), (5000, 12500)]
FEATURE_NAMES = ["mean", "std", "rms", "min", "max", "ptp", "skew", "kurtosis",
                 "energy_0", "energy_1", "energy_2", "energy_3"]
TEST_TOOLS = ["M1T3", "M2T6", "M3T9"]   # 'split' column of the original notebook (not used here)


def extract_channel_features(signal):
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x[int(len(x) * 0.10):int(len(x) * 0.90)]       # central 80% of the recording
    if len(x) < 2048 or not np.isfinite(x).all():
        raise ValueError("Signal too short or contains nonfinite data")
    feats = [np.mean(x), np.std(x), np.sqrt(np.mean(x ** 2)), np.min(x), np.max(x), np.ptp(x),
             skew(x, bias=False), kurtosis(x, fisher=True, bias=False)]
    f, psd = welch(x, fs=FS, nperseg=2048, noverlap=1024, detrend="constant", scaling="density")
    df = f[1] - f[0]
    for i, (lo, hi) in enumerate(BANDS):
        mask = (f >= lo) & ((f <= hi) if i == len(BANDS) - 1 else (f < hi))
        feats.append(np.sum(psd[mask]) * df)
    return feats


def extract_recording(root, rec):
    out = {"filename": rec["filename"], "machine": int(rec["machine"]), "tool": int(rec["tool"]),
           "physical_tool": rec["physical_tool"], "run": int(rec["run"]),
           "cumulated_tool_contact_time": float(rec["cumulated_tool_contact_time"]),
           "wear": float(rec["wear"]), "split": rec["split"]}
    with h5py.File(root / rec["filename"], "r") as h5:
        for prefix, key in CHANNELS.items():
            for name, v in zip(FEATURE_NAMES, extract_channel_features(h5[key][()])):
                out[f"{prefix}_{name}"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="../data/nine_tool_features.csv")
    ap.add_argument("--limit", type=int, default=None, help="first N recordings only (testing)")
    a = ap.parse_args()
    root = Path(a.data)
    meta = pd.read_csv(root / "filelist.csv")
    meta["physical_tool"] = "M" + meta["machine"].astype(str) + "T" + meta["tool"].astype(str)
    meta["split"] = np.where(meta.physical_tool.isin(TEST_TOOLS), "test", "source")
    if a.limit is None:
        assert len(meta) == 6418 and meta["filename"].is_unique
    else:
        meta = meta[[(root / f).exists() for f in meta.filename]].head(a.limit)
    rows, errors = [], []
    for i, rec in enumerate(meta.to_dict("records")):
        try:
            rows.append(extract_recording(root, rec))
        except Exception as exc:  # noqa: BLE001 - recorded and reported below
            errors.append((rec["filename"], str(exc)))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(meta)}", flush=True)
    feats = pd.DataFrame(rows).sort_values(["machine", "tool", "run"]).reset_index(drop=True)
    cols = [f"{c}_{n}" for c in CHANNELS for n in FEATURE_NAMES]
    assert np.isfinite(feats[cols].to_numpy()).all()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(feats)} recordings, {len(cols)} features, {len(errors)} failures")
    if errors:
        print("first failures:", errors[:5])


if __name__ == "__main__":
    main()

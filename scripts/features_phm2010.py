"""PHM2010 per-cut features (84 = 7 channels x 12), ported unchanged from cell 3 of
notebooks/phm2010-01-data-exploration.ipynb (the Kaggle run that produced phm_features.csv).

Input : a directory (searched recursively) holding the PHM2010 challenge files for the three
        labelled cutters: c_<k>_<nnn>.csv sensor files (7 columns: Fx Fy Fz Vx Vy Vz AE_RMS,
        50 kHz) and c<k>_wear.csv (columns cut, flute_1, flute_2, flute_3), k in {1, 4, 6}.
Output: phm_features.csv, 945 rows x 87 columns (84 features + cutter, cut, wear_max).

  python features_phm2010.py --data <PHM2010 dir> --out ../data/phm_features.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew

CHANNELS = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE_RMS"]
CUTTERS = [1, 4, 6]
FS = 50000
BANDS = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000)]


def extract_features(path):
    data = pd.read_csv(path, header=None, dtype=np.float32).values
    if data.shape[1] != 7:
        raise ValueError("Invalid channel count: " + str(path))
    if not np.isfinite(data).all():
        raise ValueError("Invalid signal values: " + str(path))
    n = len(data)
    data = data[int(n * 0.10):int(n * 0.90)]          # central 80% of the recording
    if len(data) < 100:
        raise ValueError("Too few samples: " + str(path))
    features = {}
    for j, channel in enumerate(CHANNELS):
        x = data[:, j].astype(np.float64)
        features[channel + "_mean"] = np.mean(x)
        features[channel + "_std"] = np.std(x)
        features[channel + "_rms"] = np.sqrt(np.mean(x ** 2))
        features[channel + "_min"] = np.min(x)
        features[channel + "_max"] = np.max(x)
        features[channel + "_ptp"] = np.ptp(x)
        features[channel + "_skew"] = skew(x)
        features[channel + "_kurtosis"] = kurtosis(x)
        freq, psd = periodogram(x, fs=FS)
        dfreq = freq[1] - freq[0]
        for low, high in BANDS:
            mask = (freq >= low) & (freq < high)
            features["{}_energy_{}_{}".format(channel, low, high)] = np.sum(psd[mask]) * dfreq
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="../data/phm_features.csv")
    a = ap.parse_args()
    all_csv = list(Path(a.data).rglob("*.csv"))
    rows, t0 = [], time.time()
    for cutter in CUTTERS:
        wear_files = [p for p in all_csv if p.name.lower() == f"c{cutter}_wear.csv"]
        assert len(wear_files) == 1, f"c{cutter}_wear.csv not found exactly once"
        wear = pd.read_csv(wear_files[0])
        assert len(wear) == 315 and wear["cut"].is_unique and set(wear["cut"]) == set(range(1, 316))
        wear = wear.set_index("cut")
        wear["wear_max"] = wear[["flute_1", "flute_2", "flute_3"]].max(axis=1)
        sensor = {}
        prefix = f"c_{cutter}_"
        for p in all_csv:
            name = p.name.lower()
            if name.startswith(prefix) and name.endswith(".csv") and name[len(prefix):-4].isdigit():
                cut = int(name[len(prefix):-4])
                if cut in sensor:
                    raise ValueError("Duplicate sensor cut")
                sensor[cut] = p
        assert set(sensor) == set(range(1, 316)), f"C{cutter}: sensor files incomplete"
        print(f"Processing C{cutter}...", flush=True)
        for cut in sorted(sensor):
            f = extract_features(sensor[cut])
            f.update(cutter=cutter, cut=cut, wear_max=wear.loc[cut, "wear_max"])
            rows.append(f)
    df = pd.DataFrame(rows)
    assert df.shape == (945, 87)
    assert np.isfinite(df.select_dtypes(include=[np.number]).values).all()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} {df.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

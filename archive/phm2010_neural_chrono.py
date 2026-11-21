"""T1 (neural part): chronological matched calibration-source test for the
five-seed CNN-BiLSTM-attention regressor on PHM2010. Run on a Kaggle GPU.

Input : phm_sequences.npz (cell 2 of phm2010-02-deep-learning: X (3780,2048,7), y, cutter, cut)
Output: <out>/t1_matched_neural.csv, <out>/t1_matched_neural_summary.csv

Design (per directed combination; test T, training cutter A, other source B):
  fit  : A cuts 1-220   (normalisation from these windows only)
  stop : A cuts 221-260 (early stopping on cut-level MAE, patience 5, <=25 epochs)
  test : all 315 cuts of T (predicted once; 5 seeds averaged)
  calibration residuals from
     same_cutter              A cuts 261-315 (n=55)
     other_cutter_same_range  B cuts 261-315 (n=55)
     other_cutter_full_life   B cuts 1-315   (n=315)
Architecture and optimiser are identical to the MST manuscript.

  !python phm2010_neural_chrono.py --sequences /kaggle/working/phm_sequences.npz --out /kaggle/working/results
"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from calib import evaluate_all_levels

CUTTERS = [1, 4, 6]
SEEDS = [42, 43, 44, 45, 46]
MAX_EPOCHS, PATIENCE = 25, 5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNBiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.lstm(self.cnn(x).transpose(1, 2))
        w = torch.softmax(self.attention(h), dim=1)
        return self.head((h * w).sum(dim=1)).squeeze(-1)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Fold:
    """Normalisation + tensors for one training cutter."""

    def __init__(self, X, y, train_m):
        self.X, self.y = X, y
        self.cm = X[train_m].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.cs = np.maximum(X[train_m].std(axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)
        self.ym = float(y[train_m].mean())
        self.ys = max(float(y[train_m].std()), 1e-6)

    def tensors(self, m):
        xx = np.transpose((self.X[m] - self.cm) / self.cs, (0, 2, 1)).copy()
        yy = ((self.y[m] - self.ym) / self.ys).astype(np.float32)
        return TensorDataset(torch.from_numpy(xx), torch.from_numpy(yy))

    def predict(self, net, m):
        net.eval()
        out = []
        with torch.no_grad():
            for xb, _ in DataLoader(self.tensors(m), batch_size=128):
                out.append(net(xb.to(DEV)).cpu().numpy())
        return np.concatenate(out) * self.ys + self.ym


def cut_mean(pred, cut, m):
    return pd.Series(pred).groupby(cut[m]).mean()  # index = cut number, sorted


def train_one(fold, train_m, val_m, cut, seed):
    seed_all(seed)
    net = CNNBiLSTMAttention().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(fold.tensors(train_m), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    yv = pd.Series(fold.y[val_m]).groupby(cut[val_m]).mean()
    best, best_state, bad = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(net(xb.to(DEV)), yb.to(DEV))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        mae = float(np.mean(np.abs(cut_mean(fold.predict(net, val_m), cut, val_m) - yv)))
        if mae < best - 0.01:
            best, best_state, bad = mae, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    with np.load(a.sequences) as z:
        X, y = z["X"].astype(np.float32), z["y"].astype(np.float32)
        cutter, cut = z["cutter"].astype(int), z["cut"].astype(int)
    assert X.shape == (3780, 2048, 7)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for T in CUTTERS:
        mT = cutter == T
        yT = pd.Series(y[mT]).groupby(cut[mT]).mean()
        for A in [c for c in CUTTERS if c != T]:
            B = [c for c in CUTTERS if c not in (T, A)][0]
            tr = (cutter == A) & (cut <= 220)
            va = (cutter == A) & (cut >= 221) & (cut <= 260)
            cal_sets = {"same_cutter": (cutter == A) & (cut >= 261),
                        "other_cutter_same_range": (cutter == B) & (cut >= 261),
                        "other_cutter_full_life": cutter == B}
            fold = Fold(X, y, tr)
            preds = {k: [] for k in ["test", *cal_sets]}
            for s in SEEDS:
                net = train_one(fold, tr, va, cut, s)
                preds["test"].append(cut_mean(fold.predict(net, mT), cut, mT))
                for k, m in cal_sets.items():
                    preds[k].append(cut_mean(fold.predict(net, m), cut, m))
                print(f"  T=C{T} A=C{A} seed {s} done", flush=True)
            yhatT = pd.concat(preds["test"], axis=1).mean(axis=1).reindex(yT.index).to_numpy()
            for k, m in cal_sets.items():
                ycal = pd.Series(y[m]).groupby(cut[m]).mean()
                pcal = pd.concat(preds[k], axis=1).mean(axis=1).reindex(ycal.index)
                res = np.abs(ycal.to_numpy() - pcal.to_numpy())
                for rec in evaluate_all_levels(yT.to_numpy(), yhatT, yT.index.to_numpy(), res):
                    rec.update(protocol="chrono", model="cnn_bilstm_5seed", test=f"C{T}",
                               train=f"C{A}", cal_cutter=f"C{A if k == 'same_cutter' else B}",
                               cal_source=k, test_mae=float(np.mean(np.abs(yT.to_numpy() - yhatT))))
                    rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "t1_matched_neural.csv", index=False)
    num = ["coverage", "width", "interval_score", "cov_stage1", "cov_stage2", "cov_stage3"]
    s = df.groupby(["cal_source", "level"])[num].agg(["mean", "std"])
    s.columns = [f"{x}_{y}" for x, y in s.columns]
    s.reset_index().to_csv(out / "t1_matched_neural_summary.csv", index=False)
    print(s.xs(0.9, level="level").round(2).to_string())


if __name__ == "__main__":
    main()

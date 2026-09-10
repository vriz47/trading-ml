#!/usr/bin/env python3
"""Grid-scan horizon fwd untuk cari region profitable (walk-forward GBM).

Satu proses: bangun fitur sekali, lalu utk tiap fwd buat label, train GBM
walk-forward + holdout, backtest dgn biaya asli. Output metrics per fwd.

Contoh:
  python3 scripts/scan_fwd.py --fwd 4,8,12,16,24,48 --folds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_dataset import build_features  # noqa: E402
from train_gbm import HistGBM, aggregate  # noqa: E402

M15 = 900.0


def load_csv(path: Path) -> np.ndarray:
    a = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1)
    return a[np.isfinite(a[:, 0])]


def make_labels(close: np.ndarray, ts: np.ndarray, fwd: int,
                pct: float) -> np.ndarray:
    fut = close[fwd:]
    past = close[:-fwd]
    reg = fut / past - 1.0
    y = np.zeros_like(reg, dtype=np.int8)
    y[reg > pct] = 1
    y[reg < -pct] = -1
    return np.concatenate([y, np.zeros(fwd, np.int8)])[: len(ts)]


def folds(n: int, k: int):
    sizes = [n // (k + 1)] * (k + 1)
    for i in range(n % (k + 1)):
        sizes[i] += 1
    edges = np.cumsum([0] + sizes)
    return [(slice(0, edges[i]), slice(edges[i], edges[i + 1])) for i in range(1, k + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohlc", type=Path, default=ROOT / "data/raw/MT5_XAUUSD_M15_120d.csv")
    ap.add_argument("--h1", type=Path, default=ROOT / "data/raw/MT5_XAUUSD_H1_365d.csv")
    ap.add_argument("--fwd", type=str, default="4,8,12,16,24,48")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--spread", type=float, default=0.26)
    args = ap.parse_args()

    t0 = time.perf_counter()
    m15 = load_csv(args.ohlc)
    h1 = load_csv(args.h1)
    feat, ts, close = build_features(m15, h1)
    print(f"bars={len(close)} feat={feat.shape[1]}", flush=True)

    closes = dict(zip(ts.astype(int), close))
    n_cv = int(len(close) * args.split)
    A = feat.astype(np.float64)
    fwds = [int(x) for x in args.fwd.split(",")]

    print(f"\n{'fwd':>4} {'trades':>7} {'win%':>6} {'payoff':>6} {'pnl$':>10} "
          f"{'shrp':>6}  verdict")
    for fwd in fwds:
        y3 = make_labels(close, ts, fwd, pct=0.0004)
        keep = y3 != 0
        y = (y3 > 0).astype(float)
        prob = np.zeros((len(y3), 2), np.float32)

        for tr, va in folds(n_cv, args.folds):
            g = HistGBM(nbins=32, max_depth=4, min_leaf=40, lr=0.05,
                        n_est=200, patience=15)
            mtr, mva = keep[tr], keep[va]
            g.fit(A[tr][mtr], y[tr][mtr], A[va][mva], y[va][mva])
            p = g.predict_prob(A[va])
            prob[va, 0] = p
            prob[va, 1] = 1 - p

        g = HistGBM(nbins=32, max_depth=4, min_leaf=40, lr=0.05,
                    n_est=200, patience=15)
        mtr, mho = keep[:n_cv], keep[n_cv:]
        g.fit(A[:n_cv][mtr], y[:n_cv][mtr], A[n_cv:][mho], y[n_cv:][mho])
        ph = g.predict_prob(A[n_cv:])
        prob[n_cv:, 0] = ph
        prob[n_cv:, 1] = 1 - ph

        trades = []
        for i in range(n_cv, len(ts)):
            t_in = int(ts[i])
            if t_in not in closes:
                continue
            pu, pd = float(prob[i, 0]), float(prob[i, 1])
            if max(pu, pd) < 0.5:
                continue
            side = 1 if pu >= pd else -1
            t_ex = t_in + fwd * M15
            if t_ex not in closes:
                continue
            move = (closes[t_ex] - closes[t_in]) * side
            trades.append(move - args.spread)

        if not trades:
            print(f"{fwd:>4}  no trades", flush=True)
            continue
        ret = np.array(trades)
        n = len(ret)
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        wr = len(wins) / n
        payoff = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan
        pnl = ret.sum()
        sharpe = ret.mean() / ret.std() * np.sqrt(96) if ret.std() > 0 else 0.0
        ver = "LOSS" if pnl < 0 else "PROFIT"
        print(f"{fwd:>4} {n:>7} {100*wr:>5.1f}% {payoff:>6.2f} "
              f"{pnl:>9.2f} {sharpe:>6.2f}  {ver}", flush=True)
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
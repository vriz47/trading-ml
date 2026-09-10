#!/usr/bin/env python3
"""Backtest prediksi GBM (up/down) pada XAUUSDc M15.

Model: prob up di kolom 0, prob down di kolom 1 (hasil train_gbm.py).
Entry: close bar ts (akhir window). Exit: close bar ts + fwd*900s.
Biaya: spread asumsi + komisi per lot (~100 oz utk XAUUSDc).

Contoh:
  python3 scripts/backtest.py \
    --prob models/MT5_XAUUSD_M15_120d_L64_F12_h1_gbm.npz \
    --ohlc data/raw/MT5_XAUUSD_M15_120d.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M15 = 900.0


def load_ohlc(path: Path) -> dict:
    a = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float,
                      filling_values=np.nan)
    a = a[np.isfinite(a[:, 0])]
    return {int(t): c for t, c in zip(a[:, 0], a[:, 4])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prob", type=Path, required=True)
    ap.add_argument("--ohlc", type=Path, required=True)
    ap.add_argument("--fwd", type=int, default=12)       # 12 bar = 3 jam
    ap.add_argument("--spread", type=float, default=0.26)  # USD per oz (poin)
    ap.add_argument("--commission", type=float, default=0.00)  # USD per lot per sisi
    ap.add_argument("--oz", type=float, default=1.0)     # lot 0.01 = 1 oz
    ap.add_argument("--th", type=float, default=0.5)     # threshold prob
    ap.add_argument("--max-exit-bar", type=int, default=48)  # cap hold 12 jam
    args = ap.parse_args()

    d = np.load(args.prob)
    closes = load_ohlc(args.ohlc)

    def run(prob, ts, y):
        trades = []
        for i in range(len(ts)):
            t_in = int(ts[i])
            if t_in not in closes:
                continue
            pu, pd = float(prob[i, 0]), float(prob[i, 1])
            if max(pu, pd) < args.th:
                continue
            side = 1 if pu >= pd else -1
            t_exit = t_in + args.fwd * M15
            if t_exit not in closes:
                continue
            p_in = closes[t_in]
            p_out = closes[t_exit]
            move = (p_out - p_in) if side == 1 else (p_in - p_out)
            cost = args.spread + 2 * args.commission / 100.0
            trades.append((side, p_in, p_out, move - cost, ts[i], y[i]))
        return trades

    tr_cv = run(d["prob_cv"], d["ts_cv"], d["y_cls_cv"])
    tr_ho = run(d["prob_hold"], d["ts_hold"], d["y_cls_hold"])

    def report(name, trades):
        if not trades:
            print(f"{name}: no trades")
            return
        ret = np.array([t[3] for t in trades], float)
        n = len(ret)
        wins = ret[(np.isfinite(ret)) & (ret > 0)]
        losses = ret[(np.isfinite(ret)) & (ret <= 0)]
        pnl = ret.sum()
        cum = np.cumsum(ret)
        mdd = float((cum - np.maximum.accumulate(cum)).min()) if n else 0.0
        sharpe = float(ret.mean() / ret.std() * np.sqrt(96)) if ret.std() > 0 else 0.0
        acc = float(np.mean([1 if (t[0] == 1 and t[5] == 1) or (t[0] == -1 and t[5] == -1) else 0 for t in trades]))
        print(f"\n== {name} ==  trades={n} win={len(wins)}/{n} ({100*len(wins)/n:.1f}%)")
        print(f"   acc_direction(idx)={acc:.3f}  payoff(w/l)={wins.mean()/abs(losses.mean()) if len(wins) and len(losses) else float('nan'):.2f}")
        print(f"   gross pnl=${pnl:.2f}  avg={ret.mean():.3f}/trade  win_avg={wins.mean():.3f}  loss_avg={losses.mean():.3f}")
        print(f"   maxDD=${mdd:.2f}  sharpe(annual, 96/bar)={sharpe:.2f}")
        return trades

    report("CV (walk-forward OOS)", tr_cv)
    report("FORWARD TEST (holdout)", tr_ho)


if __name__ == "__main__":
    main()
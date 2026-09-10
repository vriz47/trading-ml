#!/usr/bin/env python3
"""Scan strategi trading klasik di M15 XAUUSDc dgn spread real Exness (0.26).

Walks-forward: cari best strategy di paruh AWAL data, laporan hasil di paruh
AKHIR (out-of-sample). Biaya = spread per round trip, 1 oz per trade.

Strategi:
  - MA cross (SMA & EMA, beberapa periode)
  - Donchian breakout (channel break - trend following)
  - Mean reversion z-score
  - RSI-style (harga vs MA +/- k*ATR)
  - ATR trailing trend

Contoh:
  python3 scripts/scan_strategies.py --hold 12 --first 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> np.ndarray:
    a = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1)
    return a[np.isfinite(a[:, 0])]


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    a = 2.0 / (n + 1)
    for i in range(1, len(x)):
        out[i] = x[i] * a + out[i - 1] * (1 - a)
    return out


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.zeros_like(c)
    if len(c) >= n:
        out[n - 1] = tr[:n].mean()
        a = 1.0 / n
        for i in range(n, len(c)):
            out[i] = out[i - 1] * (1 - a) + tr[i] * a
    return out


def roll_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(n, len(x)):
        out[i] = x[i - n : i].max()
    return out


def roll_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(n, len(x)):
        out[i] = x[i - n : i].min()
    return out


def rolling_z(c: np.ndarray, n: int = 40) -> np.ndarray:
    out = np.zeros_like(c)
    for i in range(n, len(c)):
        w = c[i - n : i]
        mu, sd = w.mean(), w.std() + 1e-9
        out[i] = (c[i] - mu) / sd
    return out


def run_backtest(close: np.ndarray, ts: np.ndarray, side: np.ndarray,
                 hold: int, spread: float, M15: float = 900.0):
    closes = {int(t): x for t, x in zip(ts, close)}
    rets, wins, n_tr = [], 0, 0
    for i, s in enumerate(side):
        if not np.isfinite(s) or s == 0:
            continue
        t_in, t_ex = int(ts[i]), int(ts[i]) + hold * M15
        if t_in not in closes or t_ex not in closes:
            continue
        mv = (closes[t_ex] - closes[t_in]) * s - spread
        rets.append(mv)
        n_tr += 1
        if mv > 0:
            wins += 1
    if not rets:
        return None
    r = np.array(rets)
    return n_tr, wins / n_tr, r.sum(), r.std()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohlc", type=Path, default=ROOT / "data/raw/MT5_XAUUSD_M15_120d.csv")
    ap.add_argument("--first", type=float, default=0.5)
    ap.add_argument("--hold", type=int, default=12)
    ap.add_argument("--spread", type=float, default=0.26)
    args = ap.parse_args()

    a = load_csv(args.ohlc)
    t, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    at = atr(h, l, c)

    strat = {}
    for n in (10, 20):
        strat[f"maSMA_{n*2}_{n*4}"] = np.sign(sma(c, n * 2) - sma(c, n * 4))
    strat["maEMA_5_20"] = np.sign(ema(c, 5) - ema(c, 20))
    strat["maEMA_10_40"] = np.sign(ema(c, 10) - ema(c, 40))
    for n in (20, 40, 80):
        strat[f"donch_{n}"] = 1 * (c > roll_max(c, n)) - 1 * (c < roll_min(c, n))
    for n, z in ((40, 1.5), (80, 2.0)):
        strat[f"zscore_{n}"] = np.where(rolling_z(c, n) < -z, 1,
                          np.where(rolling_z(c, n) > z, -1, 0))
    for k in (1.0, 1.5, 2.0):
        strat[f"MA20_ATR_{k}"] = np.where(c < sma(c, 20) - k * at, 1,
                             np.where(c > sma(c, 20) + k * at, -1, 0))

    cut = int(len(c) * args.first)
    print(f"walk-forward: pilih di awal({cut}) -> uji akhir({len(c)-cut}) "
          f"hold={args.hold} spread={args.spread}")
    print()
    res = []
    for name, side in strat.items():
        side = np.asarray(side)
        bt_tr = run_backtest(c[:cut], t[:cut], side[:cut], args.hold, args.spread)
        bt_te = run_backtest(c[cut:], t[cut:], side[cut:], args.hold, args.spread)
        if bt_te is None:
            print(f"{name:<13} test: no trades")
            continue
        n, wr, pnl, sd = bt_te
        tr_pnl = bt_tr[2] if bt_tr else 0.0
        res.append((name, tr_pnl, pnl))
        print(f"{name:<13} test n={n:>5} win={100*wr:>5.1f}% pnl=${pnl:>9.2f} "
              f"(train ${tr_pnl:>9.2f})")

    if res:
        best = max(res, key=lambda r: r[2])
        print()
        print(f"BEST out-of-sample: {best[0]}  train${best[1]:.0f} test${best[2]:.0f}")


if __name__ == "__main__":
    main()
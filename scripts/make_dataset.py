#!/usr/bin/env python3
"""Bangun dataset multi-timeframe (M15 + H1 anchor) untuk LSTM/GRU.

- Align H1 ke M15 secara KAUSAL: fitur H1 hanya memakai bar yang sudah TUTUP
  sebelum/tepat saat ts M15, tanpa lookahead.
- Sequence window L bar, target fwd M15 bar ke depan (reg + cls).
- Split TEMPORAL: train = bagian awal data, val = bagian akhir, tanpa
  shuffle lintas waktu (hindari leakage).
- NPZ menyimpan ts_train/ts_val (epoch bar terakhir tiap sample) agar
  backtest/forward test bisa dipetakan temporal.

Contoh:
  python3 scripts/make_dataset.py \
    --input data/raw/MT5_XAUUSD_M15_120d.csv \
    --h1 data/raw/MT5_XAUUSD_H1_365d.csv \
    --len 64 --fwd 12 --split 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> np.ndarray:
    a = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1)
    if a.ndim != 2 or a.shape[1] < 6:
        print(f"CSV tidak valid: {path} shape={a.shape}", file=sys.stderr)
        sys.exit(1)
    return a


def atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray,
          n: int = 14) -> np.ndarray:
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = np.zeros_like(close)
    if len(close) >= n:
        atr[n - 1] = tr[:n].mean()
        alpha = 1.0 / n
        for i in range(n, len(close)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def wilder_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close)
    up = np.maximum(diff, 0.0)
    dn = np.maximum(-diff, 0.0)
    rsi = np.full(len(close), 50.0)
    if len(close) <= n:
        return rsi
    au = up[:n].mean()
    ad = dn[:n].mean()
    rsi[n] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / (ad + 1e-12))
    for i in range(n + 1, len(close)):
        au = au * (n - 1) / n + up[i - 1] / n
        ad = ad * (n - 1) / n + dn[i - 1] / n
        rsi[i] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / (ad + 1e-12))
    return rsi


def h1_anchor_features(h1: np.ndarray) -> np.ndarray:
    """Return fitur per bar H1: (ret, range, body, pos_in_day, vol) ~ est."""
    o = h1[:, 1]
    h = h1[:, 2]
    l = h1[:, 3]
    c = h1[:, 4]
    v = h1[:, 5]
    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1.0
    rng = (h - l) / (c + 1e-9)
    body = (c - o) / (o + 1e-9)
    ts = h1[:, 0]
    pos = (ts % 86400) / 86400.0
    vn = v / (np.mean(v) + 1e-9)
    return np.column_stack([ret, rng, body, pos, vn])


def causal_align(h1_feat: np.ndarray, h1_ts: np.ndarray, m15_ts: np.ndarray) -> np.ndarray:
    """Untuk tiap M15 ts ambil fitur H1 dari bar yang sudah TUTUP <= ts."""
    idx = np.searchsorted(h1_ts, m15_ts, side="right") - 1
    idx = np.clip(idx, 0, len(h1_feat) - 1)
    return h1_feat[idx]


def build_features(m15: np.ndarray, h1: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = m15[:, 0]
    o = m15[:, 1]
    h = m15[:, 2]
    l = m15[:, 3]
    c = m15[:, 4]
    vol = m15[:, 5]

    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1.0
    logr = np.zeros_like(c)
    logr[1:] = np.log(c[1:] / c[:-1])
    rng = (h - l) / (c + 1e-9)
    body = (c - o) / (o + 1e-9)
    v_ma = np.convolve(vol, np.ones(20) / 20, mode="same")
    v_ma[:20] = vol[:20].mean()
    vol_norm = vol / (v_ma + 1e-9)

    atr = atr14(h, l, c)
    atr_body = atr / (c + 1e-9)

    rsi = wilder_rsi(c) / 100.0

    mom5 = np.zeros_like(c)
    mom5[5:] = c[5:] / c[:-5] - 1.0
    mom20 = np.zeros_like(c)
    mom20[20:] = c[20:] / c[:-20] - 1.0

    z20 = np.zeros_like(c)
    for i in range(20, len(c)):
        mu = c[i - 20 : i].mean()
        sd = c[i - 20 : i].std() + 1e-9
        z20[i] = (c[i] - mu) / sd

    hoc = (ts % 86400) / 86400.0
    dow = (ts // 86400) % 7 / 7.0

    feats = [ret, logr, rng, body, vol_norm, atr_body, rsi, mom5, mom20, z20, hoc, dow]

    if h1 is not None:
        h1f = causal_align(h1_anchor_features(h1), h1[:, 0], ts)
        for k in range(h1f.shape[1]):
            feats.append(h1f[:, k])

    F = np.column_stack(feats).astype(np.float32)
    return F, ts, c


def make_sequences(feat: np.ndarray, close: np.ndarray,
                   L: int, fwd: int, pct: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fut = close[fwd:]
    past = close[:-fwd]
    reg = fut / past - 1.0
    cls = np.zeros_like(reg, dtype=np.int8)
    cls[reg > pct] = 1
    cls[reg < -pct] = -1

    feat_c = feat[: len(reg)]
    N = len(feat_c) - L + 1
    X = np.stack([feat_c[i : i + L] for i in range(N)])
    y_reg = reg[L - 1 :]
    y_cls = cls[L - 1 :]
    # ts bar terakhir yang terlihat oleh sample ke-i (bar stale, bukan label)
    ts_s = feat_c[L - 1 :, :1] * 0  # placeholder; diganti di caller
    return X, y_reg, y_cls, ts_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="CSV M15 dari data/raw")
    ap.add_argument("--h1", type=Path, default=None, help="CSV H1 untuk anchor (opsional)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--fwd", type=int, default=12)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--pct", type=float, default=0.0004, help="threshold up/down")
    args = ap.parse_args()

    m15 = load_csv(args.input)
    h1 = load_csv(args.h1) if args.h1 else None

    feat, ts_full, close = build_features(m15, h1)
    X, y_reg, y_cls, _ = make_sequences(feat, close, args.len, args.fwd, args.pct)

    # ts sample ke-i = epoch bar M15 penutup terakhir pd window (stale, pre-label)
    n_full = len(feat) - args.fwd
    ts_i = np.full(len(X), np.nan)
    for i, j in enumerate(range(args.len - 1, args.len - 1 + len(X))):
        ts_i[i] = ts_full[j]

    n_train = int(len(X) * args.split)

    outdir = ROOT / "data" / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.input.stem}_L{args.len}_F{args.fwd}"
    if args.h1 is not None:
        tag += "_h1"
    out = args.out or outdir / (tag + ".npz")
    np.savez_compressed(
        out,
        X_train=X[:n_train], y_reg_train=y_reg[:n_train], y_cls_train=y_cls[:n_train],
        X_val=X[n_train:], y_reg_val=y_reg[n_train:], y_cls_val=y_cls[n_train:],
        ts_train=ts_i[:n_train], ts_val=ts_i[n_train:],
        L=args.len, fwd=args.fwd, features=feat.shape[1], pct=args.pct,
    )
    tr_cnt = np.bincount(y_cls[:n_train] + 1, minlength=3)
    va_cnt = np.bincount(y_cls[n_train:] + 1, minlength=3)
    print(f"OK: {out}")
    print(f"  total seq={len(X)} train={n_train} val={len(X)-n_train} "
          f"X_shape={X.shape} features={feat.shape[1]}")
    print(f"  y_cls train down/flat/up={tr_cnt / n_train}")
    print(f"  y_cls val   down/flat/up={va_cnt / (len(X) - n_train)}")


if __name__ == "__main__":
    main()
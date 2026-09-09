#!/usr/bin/env python3
"""Training cepat tanpa panas di HP: agregat fitur + logistic multinomial (walk-forward).

- Fitur per sequence: dari tiap fitur window (L=64) diambil statistik
  [last, mean, std, slope, min, max] -> D = 17*6 = 102 kolom. Tanpa
  lookahead (semua dari window sejarah saja).
- Model: softmax logistic regression (scipy L-BFGS, sudah terinstall).
  Training hitungan detik, ringan, tidak bikin HP panas.
- Walk-forward temporal (expanding window) + holdout akhir sebagai forward
  test. Output prob prediksi ke NPZ untuk backtest.

Contoh:
  python3 scripts/train_logreg.py \
    --dataset data/processed/MT5_XAUUSD_M15_120d_L64_F12_h1.npz --folds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


def aggregate(X: np.ndarray) -> np.ndarray:
    """(N,L,F) -> (N, F*6): last, mean, std, slope, min, max per fitur."""
    L = X.shape[1]
    idx = np.arange(L, dtype=np.float64)
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    denom = X.std(axis=1, ddof=0) + 1e-9
    slope = ((X - X.mean(axis=1, keepdims=True)) * idx[None, :, None]).mean(axis=1) / denom
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([last, mean, std, slope, mn, mx], axis=1).astype(np.float32)


def make_folds(n: int, k: int) -> list[tuple[slice, slice]]:
    sizes = [n // (k + 1)] * (k + 1)
    for i in range(n % (k + 1)):
        sizes[i] += 1
    edges = np.cumsum([0] + sizes)
    folds = []
    for i in range(1, k + 1):
        folds.append((slice(0, edges[i]), slice(edges[i], edges[i + 1])))
    return folds


def fit_softmax(Xtr: np.ndarray, ytr: np.ndarray, W: np.ndarray | None = None,
                reg: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """L-BFGS pada softmax cross-entropy dgn L2. Return (W (F,K), bias (K,))."""
    F = Xtr.shape[1]
    K = 3
    yoh = (ytr[:, None] == np.arange(K)).astype(np.float64)

    def pack(Wf, bf):
        return np.concatenate([Wf.ravel(), bf])

    def unpack(p):
        return p[: F * K].reshape(F, K), p[F * K:]

    p0 = pack(W if W is not None else np.zeros((F, K), np.float64),
              np.zeros(K, np.float64))
    X32 = Xtr.astype(np.float64)

    def loss_grad(p):
        Wf, bf = unpack(p)
        z = X32 @ Wf + bf
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        loss = -np.mean(yoh * np.log(p + 1e-12)) + reg / 2 * (Wf ** 2).sum()
        gW = X32.T @ (p - yoh) / len(Xtr) + reg * Wf
        gb = (p - yoh).mean(axis=0)
        return loss, pack(gW, gb)

    res = minimize(loss_grad, p0, method="L-BFGS-B", jac=True,
                   options={"maxiter": 200, "ftol": 1e-9})
    return unpack(res.x)


def predict(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = X.astype(np.float64) @ W + b
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    d = np.load(args.dataset)
    X = np.concatenate([d["X_train"], d["X_val"]], 0).astype(np.float32)
    y = np.concatenate([d["y_cls_train"], d["y_cls_val"]], 0).astype(np.int64) + 1
    ts = np.concatenate([d["ts_train"], d["ts_val"]])
    n_cv = int(d["X_train"].shape[0])
    n_last = int(d["X_val"].shape[0])

    t0 = time.perf_counter()
    A = aggregate(X)
    print(f"fitur agregat: {A.shape[1]} (dari {X.shape[2]} x 6 stat)")
    A32 = A.astype(np.float64)

    accs = []
    folds = make_folds(n_cv, args.folds)
    prob_cv = np.zeros((n_cv, 3), np.float32)
    prob_hold = np.zeros((n_last, 3), np.float32)
    t_ep = time.perf_counter()
    for i, (tr, va) in enumerate(folds):
        W, b = fit_softmax(A32[tr], y[tr], reg=args.reg)
        p = predict(W, b, A32[va])
        prob_cv[va] = p
        acc = (p.argmax(1) == y[va]).mean()
        accs.append(acc)
        print(f"fold {i}: acc={acc:.4f} t={(time.perf_counter()-t_ep):.0f}s", flush=True)
        t_ep = time.perf_counter()

    W, b = fit_softmax(A32[:n_cv], y[:n_cv], reg=args.reg)
    ph = predict(W, b, A32[n_cv:])
    prob_hold = ph
    acc_hold = (ph.argmax(1) == y[n_cv:]).mean()

    print(f"\nCV OOS mean_acc={np.mean(accs):.4f} std={np.std(accs):.4f}")
    print(f"HOLDOUT (forward test, tak pernah dilihat) acc={acc_hold:.4f}")
    print(f"total t={(time.perf_counter()-t0):.0f}s")

    tag = args.out or f"{args.dataset.stem}_logreg"
    out = ROOT / "models" / f"{tag}.npz"
    conf = np.zeros((3, 3), int)
    for i in range(3):
        m = y[n_cv:] == i
        conf[i] = np.bincount(ph[m].argmax(1), minlength=3)
    np.savez_compressed(out,
                        prob_cv=prob_cv, y_cls_cv=(y[:n_cv] - 1),
                        ts_cv=ts[:n_cv],
                        prob_hold=prob_hold, y_cls_hold=(y[n_cv:] - 1),
                        ts_hold=ts[n_cv:],
                        cv_acc=np.array(accs, np.float32), hold_acc=acc_hold,
                        nfolds=args.folds, reg=args.reg)
    print(f"conf matrix (hold) row=actual:\n{conf}")
    print(f"prob -> {out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Histogram Gradient Boosting murni numpy (binary up/down), training cepat.

Hanya numpy — tanpa sklearn/xgboost (tak ada wheel aarch64). Desain mirip
LightGBM: quantile binning, split gain histogram, early stopping.

Fitur: agregat (last, mean, std, slope, min, max) tiap fitur window -> 102 kolom.
Target biner: up=1, down=0 (flat dibuang agar fokus arah).
Output prob & y ke NPZ untuk backtest (termasuk prob OOS walk-forward).

Contoh:
  python3 scripts/train_gbm.py \
    --dataset data/processed/MT5_XAUUSD_M15_120d_L64_F12_h1.npz --folds 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def aggregate(X: np.ndarray) -> np.ndarray:
    L = X.shape[1]
    idx = np.arange(L, dtype=np.float64)
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    denom = std + 1e-9
    slope = ((X - X.mean(axis=1, keepdims=True)) * idx[None, :, None]).mean(axis=1) / denom
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([last, mean, std, slope, mn, mx], axis=1).astype(np.float32)


@dataclass
class Node:
    feat: int = -1
    thr: int = -1
    val: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None


def best_split(grad: np.ndarray, cnt: np.ndarray, feat: np.ndarray,
               nbins: int, min_leaf: int, parent_sum: float, parent_cnt: int):
    best = (None, -1.0)
    for f in range(feat.shape[1]):
        col = feat[:, f]
        hb = np.bincount(col, weights=grad, minlength=nbins)
        hc = np.bincount(col, minlength=nbins)
        cb = np.cumsum(hb)
        cc = np.cumsum(hc)
        tot = cc[-1]
        left = cc[:-1] >= min_leaf
        right = (tot - cc[:-1]) >= min_leaf
        cand = left & right
        if not cand.any():
            continue
        gain_all = cb[:-1] ** 2 / (cc[:-1] + 1e-9) + (parent_sum - cb[:-1]) ** 2 / ((tot - cc[:-1]) + 1e-9)
        k = int(np.nanargmax(np.where(cand, gain_all, -np.inf)))
        g = float(gain_all[k]) - parent_sum ** 2 / parent_cnt
        if best[1] is None or g > best[1]:
            best = (f, k, g)
    return best


def build_tree(grad: np.ndarray, feat: np.ndarray, nbins: int, max_depth: int,
               min_leaf: int, depth: int = 0) -> Node:
    n = feat.shape[0]
    s = float(grad.sum())
    if depth >= max_depth or n < 2 * min_leaf:
        return Node(val=s / max(n, 1))
    best = best_split(grad, None, feat, nbins, min_leaf, s, n)
    if best is None:
        return Node(val=s / max(n, 1))
    f, k, g = best
    if g <= 1e-9:
        return Node(val=s / max(n, 1))
    col = feat[:, f]
    ml = col <= k
    mr = ~ml
    if ml.sum() < min_leaf or mr.sum() < min_leaf:
        return Node(val=s / max(n, 1))
    nd = Node(feat=f, thr=k)
    nd.left = build_tree(grad[ml], feat[ml], nbins, max_depth, min_leaf, depth + 1)
    nd.right = build_tree(grad[mr], feat[mr], nbins, max_depth, min_leaf, depth + 1)
    return nd


def predict_tree(t: Node, feat: np.ndarray) -> np.ndarray:
    if t.left is None and t.right is None:
        return np.full(feat.shape[0], t.val)
    out = np.empty(feat.shape[0], np.float64)
    li = feat[:, t.feat] <= t.thr
    out[li] = predict_tree(t.left, feat[li])
    out[~li] = predict_tree(t.right, feat[~li])
    return out


class HistGBM:
    def __init__(self, nbins: int = 32, max_depth: int = 4, min_leaf: int = 40,
                 lr: float = 0.05, n_est: int = 300, patience: int = 30) -> None:
        self.nbins = nbins
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.lr = lr
        self.n_est = n_est
        self.patience = patience
        self.offsets: list[int] = []
        self.bounds: np.ndarray | None = None
        self.trees: list[Node] = []
        self.best_trees: list[Node] = []
        self.best_ep = 0

    def bin_data(self, X: np.ndarray, bounds: np.ndarray | None = None) -> np.ndarray:
        X = X.astype(np.float64)
        D = X.shape[1]
        if bounds is None:
            bounds = np.zeros((D, self.nbins - 1))
            for f in range(D):
                q = np.quantile(X[:, f], np.linspace(0.05, 0.95, self.nbins - 1))
                if np.unique(q).size < 2:
                    q = np.linspace(X[:, f].min() - 1e-9, X[:, f].max() + 1e-9, self.nbins - 1)
                bounds[f] = q
        Xb = np.zeros_like(X, dtype=np.int32)
        for f in range(D):
            Xb[:, f] = np.digitize(X[:, f], bounds[f])
        self.bounds = bounds
        return Xb

    def partial_fit(self, Xb: np.ndarray, y: np.ndarray) -> None:
        F = np.zeros(y.shape[0], np.float64)
        for est in range(self.n_est):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
            grad = y - p
            t = build_tree(grad, Xb, self.nbins, self.max_depth, self.min_leaf)
            pred = predict_tree(t, Xb)
            F += self.lr * pred
            self.trees.append(t)

    def fit(self, Xtr: np.ndarray, ytr: np.ndarray,
            Xva: np.ndarray, yva: np.ndarray) -> dict:
        Xb = self.bin_data(Xtr)
        Xvb = self.bin_data(Xva, bounds=self.bounds)
        F = np.zeros(ytr.shape[0], np.float64)
        Fv = np.zeros(yva.shape[0], np.float64)
        best_auc = -1.0
        best_ep = 0
        since = 0
        eps = []
        for est in range(self.n_est):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
            grad = ytr - p
            t = build_tree(grad, Xb, self.nbins, self.max_depth, self.min_leaf)
            pred = predict_tree(t, Xb)
            F += self.lr * pred
            pv = 1.0 / (1.0 + np.exp(-np.clip(Fv, -30, 30)))
            Fv += self.lr * predict_tree(t, Xvb)
            pv = 1.0 / (1.0 + np.exp(-np.clip(Fv, -30, 30)))
            auc = fast_auc(yva, pv)
            eps.append(auc)
            if auc > best_auc:
                best_auc = auc
                best_ep = est + 1
                self.best_trees = list(self.trees)
                since = 0
            else:
                since += 1
                if since >= self.patience:
                    break
        self.best_ep = best_ep
        return {"auc": best_auc, "best_ep": best_ep, "n_trees": len(self.trees),
                "last_auc": float(eps[-1])}

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        Xb = self.bin_data(X, bounds=self.bounds)
        F = np.zeros(Xb.shape[0], np.float64)
        for t in self.best_trees:
            F += self.lr * predict_tree(t, Xb)
        return 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    y = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)[y == 1]
    return float((ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def make_folds(n: int, k: int) -> list[tuple[slice, slice]]:
    sizes = [n // (k + 1)] * (k + 1)
    for i in range(n % (k + 1)):
        sizes[i] += 1
    edges = np.cumsum([0] + sizes)
    return [(slice(0, edges[i]), slice(edges[i], edges[i + 1])) for i in range(1, k + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-est", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    d = np.load(args.dataset)
    X = np.concatenate([d["X_train"], d["X_val"]], 0).astype(np.float32)
    y3 = np.concatenate([d["y_cls_train"], d["y_cls_val"]], 0).astype(np.int64)
    ts = np.concatenate([d["ts_train"], d["ts_val"]])
    n_cv = int(d["X_train"].shape[0])
    n_last = int(d["X_val"].shape[0])

    A = aggregate(X)
    keep = y3 != 0
    y = (y3 == 1).astype(np.float64)
    print(f"fitur={A.shape[1]} sample_total={len(y)} up/down={int(y[keep].sum())}/{int((~y[keep].astype(bool)).sum())} "
          f"flat_dibuang={int((~keep).sum())}")

    t0 = time.perf_counter()
    folds = make_folds(n_cv, args.folds)
    accs, aucs = [], []
    prob_cv = np.zeros((n_cv, 2), np.float32)
    tn = time.perf_counter()
    for i, (tr, va) in enumerate(folds):
        g = HistGBM(nbins=args.nbins, max_depth=args.max_depth, min_leaf=args.min_leaf,
                    lr=args.lr, n_est=args.n_est, patience=args.patience)
        mtr = keep[tr]
        mva = keep[va]
        r = g.fit(A[tr][mtr], y[tr][mtr], A[va][mva], y[va][mva])
        p = g.predict_prob(A[va])
        prob_cv[va, 0] = p
        pr = g.predict_prob(A[va][mva])
        acc = float(((pr > 0.5) == y[va][mva]).mean())
        accs.append(acc)
        aucs.append(r["auc"])
        print(f"fold {i}: auc={r['auc']:.4f} acc={acc:.4f} best_ep={r['best_ep']} "
              f"t={(time.perf_counter()-tn):.0f}s", flush=True)
        tn = time.perf_counter()
    prob_cv[keep[:n_cv], 1] = prob_cv[keep[:n_cv], 0]
    prob_cv[keep[:n_cv], 0] = 1 - prob_cv[keep[:n_cv], 1]

    g = HistGBM(nbins=args.nbins, max_depth=args.max_depth, min_leaf=args.min_leaf,
                lr=args.lr, n_est=args.n_est, patience=args.patience)
    r = g.fit(A[:n_cv][keep[:n_cv]], y[:n_cv][keep[:n_cv]],
              A[n_cv:][keep[n_cv:]], y[n_cv:][keep[n_cv:]])
    ph = g.predict_prob(A[n_cv:][keep[n_cv:]])
    prob_hold = np.zeros((n_last, 2), np.float32)
    prob_hold[keep[n_cv:], 0] = ph
    prob_hold[keep[n_cv:], 1] = 1 - ph
    acc_hold = float(((ph > 0.5) == y[n_cv:][keep[n_cv:]]).mean())
    print(f"\nCV  OOS  auc={np.mean(aucs):.4f} (+-{np.std(aucs):.4f}) "
          f"acc={np.mean(accs):.4f}")
    print(f"HOLDOUT auc={r['auc']:.4f} acc={acc_hold:.4f} "
          f"(flat excluded {int((~keep[n_cv:]).sum())})")
    print(f"total t={(time.perf_counter()-t0):.0f}s (cool, no heat)")

    tag = args.out or f"{args.dataset.stem}_gbm"
    out = ROOT / "models" / f"{tag}.npz"
    np.savez_compressed(
        out,
        prob_cv=prob_cv, y_cls_cv=np.int8(y3[:n_cv]), ts_cv=ts[:n_cv],
        prob_hold=prob_hold, y_cls_hold=np.int8(y3[n_cv:]), ts_hold=ts[n_cv:],
        cv_auc=np.array(aucs, np.float32), cv_acc=np.array(accs, np.float32),
        hold_auc=r["auc"], hold_acc=acc_hold,
        params=np.array([args.nbins, args.max_depth, args.min_leaf, args.lr]),
    )
    print(f"prob -> {out}")


if __name__ == "__main__":
    main()
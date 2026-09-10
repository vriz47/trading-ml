#!/usr/bin/env python3
"""ML filter untuk entry sinyal MA40/80 cross (M15 XAUUSDc).

Strategi dasar: entry baris = bar di mana arah MA40-MA80-cross berubah
(turn signal). ML (GBM) memfilter: manakah entry yang profitable?

Clip fitur window L bar menjelang entry (tanpa lookahead):
  ret, logr, rng, body, vol_norm, atr_body, rsi, mom5, mom20, z20
agregat per fitur: last, mean, std, slope, min, max -> D kolom.

Target: +1 jika entry long profit (close exit > entry), -1 jika rugi,
        -1 jika entry short profit, dst -> binary "profitable".
       Trade profitable => pnl_sebelum_biaya + spread > 0.

Walk-forward: folds temporal pada entry_train, lalu holdout akhir.
Grafik: baseline (semua entry) vs filtered (hanya prob>th).
"""

from __future__ import annotations

import argparse
import time
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


def wilder_rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(c)
    up = np.maximum(diff, 0.0)
    dn = np.maximum(-diff, 0.0)
    rsi = np.full(len(c), 50.0)
    if len(c) > n:
        au = up[:n].mean()
        ad = dn[:n].mean()
        rsi[n] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / (ad + 1e-12))
        for i in range(n + 1, len(c)):
            au = (au * (n - 1) + up[i - 1]) / n
            ad = (ad * (n - 1) + dn[i - 1]) / n
            rsi[i] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / (ad + 1e-12))
    return rsi


def feature_matrix(a: np.ndarray) -> np.ndarray:
    o, h, l, c, vol = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    ts = a[:, 0]
    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1.0
    logr = np.zeros_like(c)
    logr[1:] = np.log(c[1:] / c[:-1])
    rng = (h - l) / (c + 1e-9)
    body = (c - o) / (o + 1e-9)
    v_ma = np.convolve(vol, np.ones(20) / 20, mode="same")
    v_ma[:20] = vol[:20].mean()
    vol_norm = vol / (v_ma + 1e-9)
    atrb = atr(h, l, c) / (c + 1e-9)
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
    return np.column_stack([ret, logr, rng, body, vol_norm, atrb, rsi,
                            mom5, mom20, z20, hoc, dow]).astype(np.float32)


def aggregate_window(Xw: np.ndarray) -> np.ndarray:
    """(N,L,F) -> (N,F*6): last, mean, std, slope, min, max."""
    L = Xw.shape[1]
    idx = np.arange(L, dtype=np.float64)
    last = Xw[:, -1, :]
    mean = Xw.mean(axis=1)
    std = Xw.std(axis=1)
    denom = std + 1e-9
    slope = ((Xw - Xw.mean(axis=1, keepdims=True)) * idx[None, :, None]).mean(axis=1) / denom
    mn = Xw.min(axis=1)
    mx = Xw.max(axis=1)
    return np.concatenate([last, mean, std, slope, mn, mx], axis=1).astype(np.float32)


def folds(n: int, k: int):
    sizes = [n // (k + 1)] * (k + 1)
    for i in range(n % (k + 1)):
        sizes[i] += 1
    edges = np.cumsum([0] + sizes)
    return [(slice(0, edges[i]), slice(edges[i], edges[i + 1])) for i in range(1, k + 1)]


class HistGBMLite:
    def __init__(self, nbins=32, max_depth=4, min_leaf=30, lr=0.05,
                 n_est=200, patience=20):
        self.nbins, self.max_depth, self.min_leaf = nbins, max_depth, min_leaf
        self.lr, self.n_est, self.patience = lr, n_est, patience
        self.bounds, self.trees, self.best_trees = None, [], []
        self.best_auc, self.best_ep = -1.0, 0

    def _bin(self, X, bounds=None):
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
        if bounds is not None:
            self.bounds = bounds
        return Xb

    def fit(self, Xtr, ytr, Xva, yva):
        Xb = self._bin(Xtr)
        Xvb = self._bin(Xva, self.bounds)
        F = np.zeros(Xtr.shape[0])
        Fv = np.zeros(Xva.shape[0])
        no_improve = 0
        for est in range(self.n_est):
            p = 1 / (1 + np.exp(-np.clip(F, -30, 30)))
            grad = ytr - p
            tree = self._build(grad, Xb)
            pred = self._pred(tree, Xb)
            F += self.lr * pred
            Fv += self.lr * self._pred(tree, Xvb)
            pv = 1 / (1 + np.exp(-np.clip(Fv, -30, 30)))
            auc = self._auc(yva, pv)
            if auc > self.best_auc:
                self.best_auc, self.best_ep, no_improve = auc, est + 1, 0
                self.best_trees = list(self.trees)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
            self.trees.append(tree)
        return self.best_auc

    def predict(self, X):
        Xb = self._bin(X, self.bounds)
        F = np.zeros(Xb.shape[0])
        for t in self.best_trees:
            F += self.lr * self._pred(t, Xb)
        return 1 / (1 + np.exp(-np.clip(F, -30, 30)))

    def _build(self, grad, Xb, depth=0):
        n = Xb.shape[0]
        s = float(grad.sum())
        if depth >= self.max_depth or n < 2 * self.min_leaf:
            return ("leaf", s / max(n, 1))
        best = None
        for f in range(Xb.shape[1]):
            col = Xb[:, f]
            hb = np.bincount(col, weights=grad, minlength=self.nbins)
            hc = np.bincount(col, minlength=self.nbins)
            cb, cc = np.cumsum(hb), np.cumsum(hc)
            tot = cc[-1]
            left = cc[:-1] >= self.min_leaf
            right = tot - cc[:-1] >= self.min_leaf
            cand = left & right
            if not cand.any():
                continue
            gain = cb[:-1] ** 2 / (cc[:-1] + 1e-9) + \
                (s - cb[:-1]) ** 2 / (tot - cc[:-1] + 1e-9)
            k = int(np.nanargmax(np.where(cand, gain, -np.inf)))
            g = gain[k] - s ** 2 / n
            if best is None or g > best[0]:
                best = (g, f, k)
        if best is None or best[0] <= 1e-9:
            return ("leaf", s / max(n, 1))
        f, k = best[1], best[2]
        ml, mr = Xb[:, f] <= k, Xb[:, f] > k
        return ("node", f, k,
                self._build(grad[ml], Xb[ml], depth + 1),
                self._build(grad[mr], Xb[mr], depth + 1))

    def _pred(self, t, Xb):
        if t[0] == "leaf":
            return np.full(Xb.shape[0], t[1])
        _, f, k, lt, rt = t
        li = Xb[:, f] <= k
        out = np.empty(Xb.shape[0])
        out[li] = self._pred(lt, Xb[li])
        out[~li] = self._pred(rt, Xb[~li])
        return out

    @staticmethod
    def _auc(y, s):
        o = np.argsort(s)
        y, s = y[o], s[o]
        np_, nn = int(y.sum()), len(y) - int(y.sum())
        if np_ == 0 or nn == 0:
            return 0.5
        ranks = np.arange(1, len(y) + 1)[y == 1]
        return float((ranks.sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohlc", type=Path, default=ROOT / "data/raw/MT5_XAUUSD_M15_120d.csv")
    ap.add_argument("--hold", type=int, default=12)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--len", type=int, default=48)
    ap.add_argument("--spread", type=float, default=0.26)
    ap.add_argument("--first", type=float, default=0.8)
    ap.add_argument("--th", type=float, default=None, help="kalau kosong: dipilih dari train fold")
    args = ap.parse_args()

    a = load_csv(args.ohlc)
    c = a[:, 4]
    t = a[:, 0]
    F = feature_matrix(a)
    M15 = 900.0
    spread = args.spread
    hold = args.hold
    L = args.len

    diff = sma(c, 40) - sma(c, 80)
    dir_prev = np.sign(diff)
    turn = np.zeros(len(c), bool)
    turn[1:] = (dir_prev[1:] != dir_prev[:-1]) & np.isfinite(dir_prev[1:]) & np.isfinite(dir_prev[:-1])

    idx = np.where(turn)[0]
    idx = idx[(idx >= L) & (idx < len(c) - hold)]
    closes = {int(x): v for x, v in zip(t, c)}

    rows = []
    for i in idx:
        side = int(np.sign(diff[i]))
        if side == 0:
            continue
        t_in = int(t[i])
        t_ex = t_in + hold * M15
        if t_in not in closes or t_ex not in closes:
            continue
        pnl = (closes[t_ex] - closes[t_in]) * side - spread
        rows.append((i, side, pnl))
    if not rows:
        print("tidak ada turn entry")
        return
    i_arr = np.array([r[0] for r in rows])
    side_arr = np.array([r[1] for r in rows])
    pnl_arr = np.array([r[2] for r in rows])
    y = (pnl_arr > 0).astype(np.float64)
    Xw = np.stack([F[i - L : i] for i in i_arr])
    X = aggregate_window(Xw)
    print(f"entries={len(i_arr)} profitable={y.sum():.0f} ({y.mean():.1%}) "
          f"net=${pnl_arr.sum():.0f}", flush=True)

    n_entry = len(i_arr)
    n_cv = int(n_entry * args.first)
    cut_i = n_cv  # holdout = sisa entry

    aucs = []
    prob_cv = np.zeros(n_cv)
    tr_folds = []
    for i in range(args.folds):
        e = int(n_cv * (i + 1) / args.folds)
        e2 = int(n_cv * (i + 2) / args.folds)
        if e >= n_cv or e2 > n_cv:
            break
        tr_folds.append((slice(0, e), slice(e, e2)))
    t0 = time.perf_counter()
    for tr, va in tr_folds:
        if va.start == va.stop:
            continue
        g = HistGBMLite()
        auc = g.fit(X[tr], y[tr], X[va], y[va])
        prob_cv[va] = g.predict(X[va])
        aucs.append(auc)
        print(f"fold tr={tr.start}:{tr.stop} va={va.start}:{va.stop} "
              f"auc={auc:.3f} t={(time.perf_counter()-t0):.0f}s", flush=True)
        t0 = time.perf_counter()

    g = HistGBMLite()
    g.fit(X[:n_cv], y[:n_cv], X[n_cv:], y[n_cv:])
    prob_hold = g.predict(X[n_cv:])

    def report(name, pnl, prob=None, th=0.0):
        if prob is None:
            use = np.ones(len(pnl), bool)
        else:
            use = prob >= th
        if use.sum() == 0:
            print(f"{name} th={th:.2f}: no trades")
            return
        r = pnl[use]
        print(f"{name} th={th:.2f}: n={use.sum():>4} win={(r>0).mean():.1%} "
              f"net=${r.sum():>7.0f} exp=${r.mean():.2f}/trade")

    print()
    print("== train period (CV) ==")
    report("baseline      ", pnl_arr[:n_cv])
    for th in (0.5, 0.55, 0.6, 0.65):
        report(f"filter th={th:.2f}", pnl_arr[:n_cv], prob_cv, th)

    print()
    print("== HOLD-OUT (tidak dipakai training) ==")
    report("baseline      ", pnl_arr[n_cv:])
    for th in (0.5, 0.55, 0.6, 0.65):
        report(f"filter th={th:.2f}", pnl_arr[n_cv:], prob_hold, th)

    if aucs:
        print(f"\nCV AUC mean={np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""ML regresi: prediksi return forward ternormalisasi ATR di M15 XAUUSDc.

Kenapa beda dari percobaan sebelumnya:
- Target = (close[t+fwd]-close[t]) / ATR[t]  -> ukuran gerakan TIDAK bergantung skala.
- Evaluasi sebagai TRADING SPREAD: trade hanya jika |prediksi| * ATR > margin
  biaya (spread*k). Inilah ukuran yang cocok dgn target "profitable".
- Fitur jauh lebih kaya: 12 Dasar + regime vol, jarak ke H/L 20/80, skew/kurt,
  autokorelasi, sesi jam, tendensi MA40x80 (persistence trend).

Evaluasi jujur:
- Walk-forward expanding folds di paruh awal, holdout paruh akhir.
- Trading: side = sign(pred), entry close[ts], exit close[ts+fwd], biaya = spread.

Contoh:
  python3 scripts/ml_predict_return.py --hold 12 --folds 4 --first 0.7
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
    at = atr(h, l, c)
    atrb = at / (c + 1e-9)
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

    # regime / konteks
    rv20 = np.zeros_like(c)          # realized vol 20 bar
    skew20 = np.zeros_like(c)
    kurt20 = np.zeros_like(c)
    ac1 = np.zeros_like(c)           # autokorelasi lag1 dari ret 20-bar
    dist_hi20 = np.zeros_like(c)     # jarak ke high 20 (>=0, <1)
    dist_lo20 = np.zeros_like(c)
    dist_hi80 = np.zeros_like(c)
    dist_lo80 = np.zeros_like(c)
    ma_state = np.zeros_like(c)      # +1 trend naik (40>80), -1 turun
    ma_age = np.zeros_like(c)        # umur regime (bar sejak cross)
    for i in range(20, len(c)):
        w = logr[i - 20 : i]
        rv20[i] = np.sqrt((w ** 2).sum())
        skew20[i] = float(((w - w.mean()) ** 3).mean() / (w.std() + 1e-12) ** 3)
        kurt20[i] = float(((w - w.mean()) ** 4).mean() / (w.std() + 1e-12) ** 4)
        ac1[i] = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if w[:-1].std() > 0 else 0.0
        hi20 = c[i - 20 : i].max()
        lo20 = c[i - 20 : i].min()
        dist_hi20[i] = (hi20 - c[i]) / (hi20 - lo20 + 1e-9)
        dist_lo20[i] = (c[i] - lo20) / (hi20 - lo20 + 1e-9)
    for i in range(80, len(c)):
        hi80 = c[i - 80 : i].max()
        lo80 = c[i - 80 : i].min()
        dist_hi80[i] = (hi80 - c[i]) / (hi80 - lo80 + 1e-9)
        dist_lo80[i] = (c[i] - lo80) / (hi80 - lo80 + 1e-9)
    s40 = sma(c, 40)
    s80 = sma(c, 80)
    d = s40 - s80
    sgn = np.sign(d)
    age = 0
    prev = 0.0
    for i in range(1, len(c)):
        if np.isfinite(d[i]):
            s = float(np.sign(d[i])) if abs(d[i]) > 1e-9 else prev
            if abs(s - prev) > 1e-9:
                age = 0 if prev != 0 else age + 1
            else:
                age += 1
            prev = s
            ma_state[i] = s
            ma_age[i] = age
    hoc = (ts % 86400) / 86400.0
    dow = (ts // 86400) % 7 / 7.0

    return np.column_stack([
        ret, logr, rng, body, vol_norm, atrb, rsi, mom5, mom20, z20,
        rv20, skew20, kurt20, ac1, dist_hi20, dist_lo20, dist_hi80, dist_lo80,
        ma_state, ma_age, hoc, dow,
    ]).astype(np.float32)


def folds(n: int, k: int):
    sizes = [n // (k + 1)] * (k + 1)
    for i in range(n % (k + 1)):
        sizes[i] += 1
    edges = np.cumsum([0] + sizes)
    return [(slice(0, edges[i]), slice(edges[i], edges[i + 1])) for i in range(1, k + 1)]


class HistGBMReg:
    """Histogram GBM regresi (squared loss). Mirip HistGBMLite tapi target real."""

    def __init__(self, nbins=32, max_depth=4, min_leaf=30, lr=0.05,
                 n_est=200, patience=20):
        self.nbins, self.max_depth, self.min_leaf = nbins, max_depth, min_leaf
        self.lr, self.n_est, self.patience = lr, n_est, patience
        self.bounds, self.trees, self.best_trees = None, [], []
        self.best_rmse = np.inf
        self.best_ep = 0

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
            grad = ytr - F
            tree = self._build(grad, Xb)
            pred = self._pred(tree, Xb)
            F += self.lr * pred
            Fv += self.lr * self._pred(tree, Xvb)
            rmse = float(np.sqrt(((yva - Fv) ** 2).mean()))
            if rmse < self.best_rmse:
                self.best_rmse, self.best_ep, no_improve = rmse, est + 1, 0
                self.best_trees = list(self.trees)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
            self.trees.append(tree)
        return self.best_rmse

    def predict(self, X):
        Xb = self._bin(X, self.bounds)
        F = np.zeros(Xb.shape[0])
        for t in self.best_trees:
            F += self.lr * self._pred(t, Xb)
        return F

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohlc", type=Path, default=ROOT / "data/raw/MT5_XAUUSD_M15_120d.csv")
    ap.add_argument("--hold", type=int, default=12)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--first", type=float, default=0.7)
    ap.add_argument("--spread", type=float, default=0.26)
    ap.add_argument("--th", type=str, default="0.0,0.5,1.0,1.5,2.0",
                    help="threshold |pred| (x ATR) utk filter entry")
    ap.add_argument("--len", type=int, default=48, help="window agregat")
    args = ap.parse_args()

    a = load_csv(args.ohlc)
    t, c = a[:, 0], a[:, 4]
    at = atr(a[:, 2], a[:, 3], c)
    F = feature_matrix(a)
    M15 = 900.0
    hold = args.hold
    spread = args.spread
    L = args.len

    closes = {int(x): v for x, v in zip(t, c)}

    starts = np.arange(L, len(c) - hold)
    ret_atr = (c[starts + hold] - c[starts]) / (at[starts] + 1e-9)
    ret_atr = np.nan_to_num(ret_atr, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.stack([F[i - L : i] for i in starts]).reshape(len(starts), -1)
    # agregat window: pakai feature per bar saja + mean/max tiap fitur window
    mean = X.reshape(len(starts), L, F.shape[1]).mean(axis=1)
    last = F[starts]
    Xf = np.concatenate([last, mean], axis=1).astype(np.float64)
    y = ret_atr.astype(np.float64)

    print(f"samples={len(y)} hold={hold} feat={Xf.shape[1]} "
          f"target |y|>0: {(np.abs(y)>0).mean():.1%}", flush=True)
    n_cv = int(len(y) * args.first)

    def backtest(pred_all, seg):
        s, e = seg
        out = []
        for i in range(s, e):
            pred = pred_all[i]
            side = 1 if pred >= 0 else -1
            t_in = int(t[starts[i]])
            t_ex = t_in + hold * M15
            if t_in not in closes or t_ex not in closes:
                continue
            mv = (closes[t_ex] - closes[t_in]) * side - spread
            out.append((mv, abs(at[starts[i]])))
        return out

    # ==== CV walk-forward ====
    print(f"\n== CV walk-forward (train:{n_cv}) ==")
    prob_cv = np.zeros(n_cv)
    t0 = time0 = __import__("time").perf_counter()
    for tr, va in folds(n_cv, args.folds):
        g = HistGBMReg()
        rmse = g.fit(Xf[tr], y[tr], Xf[va], y[va])
        prob_cv[va] = g.predict(Xf[va])
        print(f"fold {tr.start}-{va.stop}: rmse={rmse:.4f} "
              f"t={__import__('time').perf_counter()-t0:.0f}s", flush=True)
        t0 = __import__("time").perf_counter()

    # ==== holdout ====
    g = HistGBMReg()
    g.fit(Xf[:n_cv], y[:n_cv], Xf[n_cv:], y[n_cv:])
    pred_hold = g.predict(Xf[n_cv:])

    ths = [float(x) for x in args.th.split(",")]

    def report(label, preds, seg):
        s, e = seg
        base = []
        for j in range(s, e):
            k = j - s
            t_in = int(t[starts[j]])
            t_ex = t_in + hold * M15
            if t_in not in closes or t_ex not in closes:
                continue
            mv = (closes[t_ex] - closes[t_in]) * (1 if preds[k] >= 0 else -1) - spread
            base.append(mv)
        r0 = np.array(base)
        print(f"\n--- {label} ---")
        print(f"baseline all: n={len(r0):>5} win={(r0>0).mean():.1%} "
              f"net=${r0.sum():>8.0f} exp=${r0.mean():.2f}")
        for th in ths:
            sel = []
            for j in range(s, e):
                k = j - s
                t_in = int(t[starts[j]])
                t_ex = t_in + hold * M15
                if t_in not in closes or t_ex not in closes:
                    continue
                mv = (closes[t_ex] - closes[t_in]) * (1 if preds[k] >= 0 else -1) - spread
                if abs(preds[k]) * at[starts[j]] > th * spread:
                    sel.append(mv)
            if not sel:
                print(f"th={th:<4}: n=0")
                continue
            rr = np.array(sel)
            print(f"th={th:<4}: n={len(rr):>5} win={(rr>0).mean():.1%} "
                  f"net=${rr.sum():>8.0f} exp=${rr.mean():.2f}")

    report("CV OOS", prob_cv, (0, n_cv))
    report("HOLDOUT", pred_hold, (n_cv, len(y)))


if __name__ == "__main__":
    main()
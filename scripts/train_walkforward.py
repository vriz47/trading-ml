#!/usr/bin/env python3
"""Walk-forward CV + early stopping, evaluasi jujur tanpa diterbangkan.

Metode:
- Temporal expanding window: data dibelah jadi K segmen. Fold ke-k:
  train = segmen 0..k (masa lalu), val = segmen k+1 (masa depan, belum
  pernah dilihat model). Tidak ada shuffle lintas waktu.
- Tiap fold: training dgn patience (stop kalau val_acc tak membaik N epoch),
  simpan model terbaik per fold.
- Laporan: val_acc OOS per fold + rata-rata, dan GAP train-val (proxy
  overfit). Semakin kecil gap semakin sehat.
- Opsional --finetune-holdout: retrain model terbaik di seluruh block-0..k
  (tanpa fold terakhir) dan evaluasi di holdout terakhir (forward test).

Contoh:
  python3 scripts/train_walkforward.py \
    --dataset data/processed/MT5_XAUUSD_M15_120d_L64_F12_h1.npz \
    --folds 5 --epochs 60 --patience 12 --hidden 96 --layers 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")


class SequenceClassifier(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int, n_out: int = 3,
                 dropout: float = 0.3, kind: str = "lstm") -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1]
        return self.head(last)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             bs: int = 512) -> tuple[float, float, np.ndarray]:
    model.eval()
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits_all.append(model(X[i : i + bs]).numpy())
    logits = np.concatenate(logits_all)
    acc = (logits.argmax(1) == y.numpy()).mean()
    ce = float(nn.CrossEntropyLoss()(
        torch.from_numpy(logits), y).item())
    return acc, ce, logits


def make_folds(n: int, k: int) -> list[tuple[slice, slice]]:
    """Temporal expanding folds: fold-i train=[:cut_i], val=[cut_i, cut_{i+1}).
    cut_i dibagi sedemikian hingga tiap fold val ~n*k 1/(k+1)."""
    sizes = [n // (k + 1)] * (k + 1)
    for i in range(n % (k + 1)):
        sizes[i] += 1
    edges = np.cumsum([0] + sizes)
    folds = []
    for i in range(1, k + 1):
        folds.append((slice(0, edges[i]), slice(edges[i], edges[i + 1])))
    return folds


def train_fold(model: nn.Module, Xtr, ytr, Xva, yva, epochs: int,
               patience: int, lr: float, batch: int,
               fold_label: str = "", step_t0: float | None = None) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    best_acc = -1.0
    best_ce = float("inf")
    best_ep = 0
    best_sd = None
    since = 0
    ep_acc = []
    t0 = step_t0 or time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        va, vce, _ = evaluate(model, Xva, yva)
        ep_acc.append(va)
        elapsed = time.perf_counter() - t0
        eta = elapsed / ep * (epochs - ep)
        print(f"  ep {ep:3d}/{epochs} val_acc={va:.4f} best={best_acc:.4f} "
              f"elapsed={elapsed:.0f}s eta~{eta:.0f}s)", flush=True)
        if va > best_acc:
            best_acc, best_ce, best_ep = va, vce, ep
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_sd)
    return {
        "best_ep": best_ep, "best_val_acc": best_acc,
        "best_val_ce": best_ce, "last_ep": ep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--kind", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--holdout", action="store_true",
                    help="setelah CV, retrain di seluruh data & eval val (forward test)")
    args = ap.parse_args()

    d = np.load(args.dataset)
    X_all = torch.from_numpy(np.concatenate([d["X_train"], d["X_val"]], 0).astype(np.float32))
    y_all = torch.from_numpy(np.concatenate([d["y_cls_train"], d["y_cls_val"]], 0).astype(np.int64) + 1)
    L, n_feat = d["X_train"].shape[1], d["X_train"].shape[2]
    ts_all = np.concatenate([d["ts_train"], d["ts_val"]])
    n_cv = int(d["X_train"].shape[0])
    n_last = int(d["X_val"].shape[0])
    t_first = np.datetime64(int(ts_all[0]), "s")
    t_train_end = np.datetime64(int(ts_all[n_cv - 1]), "s")
    t_end = np.datetime64(int(ts_all[-1]), "s")

    print(f"rentang data: {t_first} .. {t_train_end} (train/cv) .. {t_end} (holdout)")
    print(f"CV window: {n_cv} seq -> {args.folds} fold; holdout terakhir: {n_last} seq")

    folds = make_folds(n_cv, args.folds)
    results = []
    for i, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        model = SequenceClassifier(n_feat, args.hidden, args.layers, kind=args.kind).to(DEVICE)
        r = train_fold(model, X_all[tr], y_all[tr], X_all[va], y_all[va],
                       args.epochs, args.patience, args.lr, args.batch)
        results.append({**r, "fold": i, "time": time.perf_counter() - t0})
        print(f"fold {i}: val_acc={r['best_val_acc']:.4f} ce={r['best_val_ce']:.4f} "
              f"best_ep={r['best_ep']} stop={r['last_ep']} "
              f"t={results[-1]['time']:.0f}s", flush=True)
        torch.save({"state_dict": model.state_dict(),
                    "cfg": {"hidden": args.hidden, "layers": args.layers,
                            "n_feat": n_feat, "n_out": 3, "kind": args.kind,
                            "L": int(L), "fwd": int(d["fwd"]), "fold": i},
                    "fold_result": r},
                   ROOT / "models" / f"wf_{args.dataset.stem}_fold{i}.pt")

    accs = [r["best_val_acc"] for r in results]
    print(f"\nCV OOS: mean_acc={np.mean(accs):.4f} std={np.std(accs):.4f} "
          f"min={np.min(accs):.4f} max={np.max(accs):.4f}")

    if args.holdout:
        t0 = time.perf_counter()
        model = SequenceClassifier(n_feat, args.hidden, args.layers, kind=args.kind).to(DEVICE)
        Xtr = X_all[: -n_last]
        ytr = y_all[: -n_last]
        Xte = X_all[-n_last:]
        yte = y_all[-n_last:]
        r = train_fold(model, Xtr, ytr, Xte, yte,
                       args.epochs, args.patience, args.lr, args.batch)
        va, ce, _ = evaluate(model, Xte, yte)
        acc_tr, _, _ = evaluate(model, Xtr, ytr)
        print(f"\nFORWARD TEST (holdout akhir {n_last} sample, TAK pernah dilihat):")
        print(f"  val_acc={va:.4f} ce={ce:.4f} | gap train-val={acc_tr - va:.4f} "
              f"| best_ep={r['best_ep']} t={time.perf_counter()-t0:.0f}s")
        out = args.out or ROOT / "models" / f"wf_{args.dataset.stem}_holdout.pt"
        torch.save({"state_dict": model.state_dict(),
                    "cfg": {"hidden": args.hidden, "layers": args.layers,
                            "n_feat": n_feat, "n_out": 3, "kind": args.kind,
                            "L": int(L), "fwd": int(d["fwd"]), "fold": "holdout"},
                    "fold_result": r, "val_acc": va, "train_acc": acc_tr},
                   out)
        print(f"  model -> {out}")
        results.append({"fold": "holdout", "best_val_acc": va})

    summary = ROOT / "models" / f"wf_{args.dataset.stem}_summary.json"
    summary.write_text(json.dumps({
        "dataset": str(args.dataset), "folds": args.folds, "epochs": args.epochs,
        "patience": args.patience, "hidden": args.hidden, "layers": args.layers,
        "lr": args.lr, "kind": args.kind, "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)), "results": results,
    }, indent=2, default=str))
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
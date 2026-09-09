#!/usr/bin/env python3
"""Training LSTM/GRU classifier utk prediksi arah harga.

Gunakan numpy torch CPU. Output: models/<name>.pt (state_dict + config).
Evaluasi: accuracy, directional accuracy, loss train/val.

Contoh:
  python3 scripts/train_lstm.py --dataset data/processed/*_L64_F12.npz --epochs 50
"""

from __future__ import annotations

import argparse
import json
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
                 dropout: float = 0.2, kind: str = "lstm") -> None:
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


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    d = np.load(path)
    return (d["X_train"], d["y_cls_train"], d["X_val"], d["y_cls_val"],
            int(d["L"]), int(d["features"]), int(d["fwd"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kind", choices=["lstm", "gru"], default="lstm")
    args = ap.parse_args()

    Xtr, ytr, Xva, yva, L, n_feat, fwd = load_npz(args.dataset)
    ytr = ytr + 1
    yva = yva + 1
    Xtr = torch.from_numpy(Xtr.astype(np.float32))
    ytr = torch.from_numpy(ytr.astype(np.int64))
    Xva = torch.from_numpy(Xva.astype(np.float32))
    yva = torch.from_numpy(yva.astype(np.int64))

    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)

    model = SequenceClassifier(n_feat, args.hidden, args.layers, n_out=3,
                               kind=args.kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    out = None
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, hit = 0, 0
        ep_t0 = time.perf_counter()
        for bi, (xb, yb) in enumerate(dl):
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += yb.numel()
            hit += (logits.argmax(1) == yb).sum().item()
            if bi % 5 == 0:
                bs = xb.shape[0]
                print(f"  batch {bi:3d} n={bs*dl.batch_size:5d} loss={loss.item():.4f}",
                      flush=True)
        sched.step()
        ep_time = time.perf_counter() - ep_t0

        model.eval()
        with torch.no_grad():
            vlogits = model(Xva)
            vloss = crit(vlogits, yva).item()
            vacc = (vlogits.argmax(1) == yva).float().mean().item()

        if vacc > best_acc:
            best_acc = vacc
            out = args.out or ROOT / "models" / "baseline_lstm.pt"
            ROOT.joinpath("models").mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "cfg": {"hidden": args.hidden, "layers": args.layers,
                                "n_feat": int(n_feat), "n_out": 3, "kind": args.kind,
                                "L": int(L), "fwd": int(fwd)}},
                       out)
        print(f"ep {ep:3d}/{args.epochs}  train_acc={hit/tot:.4f}  "
              f"val_loss={vloss:.4f}  val_acc={vacc:.4f}  ({ep_time:.1f}s/epoch)",
              flush=True)

    print(f"DONE best_val_acc={best_acc:.4f} {time.perf_counter()-t0:.0f}s "
          f"-> {out}")


if __name__ == "__main__":
    main()
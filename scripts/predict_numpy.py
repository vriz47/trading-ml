#!/usr/bin/env python3
"""Inference LSTM murni numpy (tanpa torch), + benchmark & cek dgn torch.

Cara pakai:
  python3 scripts/predict_numpy.py --dataset data/processed/*_L64_F12.npz
  python3 scripts/predict_numpy.py --dataset ... --bench 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_weights(path: Path) -> tuple[dict, dict]:
    d = np.load(path)
    cfg = {}
    w = {}
    for k in d.files:
        if k == "_cfg":
            cfg = {"n_feat": int(d[k][0]), "hidden": int(d[k][1]),
                   "layers": int(d[k][2]), "n_out": int(d[k][3]),
                   "L": int(d[k][4]), "fwd": int(d[k][5]),
                   "kind": "gru" if int(d[k][6]) else "lstm"}
        else:
            w[k] = d[k]
    return w, cfg


def lstm_forward(x: np.ndarray, w: dict) -> np.ndarray:
    """Run LSTM (1 layer) batch_first. Input: (B, L, n_feat) float32."""
    B, L, _ = x.shape
    H = w["rnn.bias_ih_l0"].shape[0] // 4
    b = w["rnn.bias_ih_l0"] + w["rnn.bias_hh_l0"]

    h = np.zeros((B, H), dtype=np.float32)
    c = np.zeros((B, H), dtype=np.float32)
    for t in range(L):
        gates = x[:, t] @ w["rnn.weight_ih_l0"].T + b + h @ w["rnn.weight_hh_l0"].T
        i = 1.0 / (1.0 + np.exp(-gates[:, :H]))
        f = 1.0 / (1.0 + np.exp(-gates[:, H:2 * H]))
        g = np.tanh(gates[:, 2 * H:3 * H])
        o = 1.0 / (1.0 + np.exp(-gates[:, 3 * H:]))
        c = f * c + i * g
        h = o * np.tanh(c)
    return h


def head_forward(z: np.ndarray, w: dict, eps: float = 1e-5) -> np.ndarray:
    m = z.mean(axis=1, keepdims=True)
    v = z.var(axis=1, keepdims=True)
    n = (z - m) / np.sqrt(v + eps) * w["head.0.weight"] + w["head.0.bias"]
    z = n @ w["head.1.weight"].T + w["head.1.bias"]
    z = np.maximum(z, 0.0)
    return z @ w["head.4.weight"].T + w["head.4.bias"]


def predict(x: np.ndarray, w: dict) -> np.ndarray:
    h = lstm_forward(x.astype(np.float32), w)
    logits = head_forward(h, w)
    return logits.argmax(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "models" / "baseline_lstm_weights.npz")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()

    w, cfg = load_weights(args.weights)
    d = np.load(args.dataset)
    Xva, yva = d["X_val"], d["y_cls_val"] + 1
    print(f"model  L={cfg['L']} fwd={cfg['fwd']} hidden={cfg['hidden']} "
          f"layers={cfg['layers']} n_feat={cfg['n_feat']}")

    logits_ref = lstm_forward(Xva[:8].astype(np.float32), w)
    logits_ref = head_forward(logits_ref, w)

    try:
        import torch  # noqa: PLC0415
        sys_path = str(ROOT / "scripts")
        import sys  # noqa: PLC0415
        sys.path.insert(0, sys_path)
        from train_lstm import SequenceClassifier  # noqa: PLC0415
        ckpt = torch.load(ROOT / "models" / "baseline_lstm.pt",
                          map_location="cpu", weights_only=False)
        model = SequenceClassifier(n_feat=cfg["n_feat"], hidden=cfg["hidden"],
                                   layers=cfg["layers"], n_out=cfg["n_out"],
                                   kind=cfg["kind"]).eval()
        model.load_state_dict(ckpt["state_dict"])
        with torch.no_grad():
            tor = model(torch.from_numpy(Xva[:8].astype(np.float32))).numpy()
        md = np.max(np.abs(logits_ref - tor))
        print(f"cocok torch? max_diff={md:.2e} ({'OK' if md < 1e-4 else 'SALAH'})")
    except Exception as e:
        print(f"skip cek torch: {type(e).__name__} {e}")

    yp = predict(Xva, w)
    acc = (yp == yva).mean()
    classes = np.array([(yp == yva)[yva == c].mean() if (yva == c).sum() else float("nan")
                        for c in [0, 1, 2]])
    print(f"val_acc={acc:.4f}  per-kls(down/flat/up)={classes.round(3)}")

    if args.bench:
        xb = Xva[: args.bench].astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = lstm_forward(xb, w)
        dt = (time.perf_counter() - t0) / 50 / args.bench * 1e6
        print(f"bench  {args.bench} sampel x 50x: {dt:.2f} us/sampel "
              f"({1e6 / dt:.0f} sampel/detik)")


if __name__ == "__main__":
    main()
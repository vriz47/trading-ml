#!/usr/bin/env python3
"""Dump weights dari checkpoint .pt ke .npz (supaya inference tanpa torch)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "scripts"
import sys

sys.path.insert(0, str(sys_path))
from train_lstm import SequenceClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "baseline_lstm.pt")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = SequenceClassifier(n_feat=cfg["n_feat"], hidden=cfg["hidden"],
                               layers=cfg["layers"], n_out=cfg["n_out"],
                               kind=cfg["kind"]).eval()
    model.load_state_dict(ckpt["state_dict"])

    sd = {k: v.numpy() for k, v in model.state_dict().items()}
    out = args.out or ROOT / "models" / "baseline_lstm_weights.npz"
    kind_codes = {"lstm": 0, "gru": 1}
    np.savez_compressed(out, **sd, _cfg=np.array([cfg["n_feat"], cfg["hidden"],
                                                  cfg["layers"], cfg["n_out"],
                                                  cfg["L"], cfg["fwd"],
                                                  kind_codes[cfg["kind"]]]))

    print(f"cfg: {cfg}")
    print(f"weights dikunci: {len(sd)} tensor -> {out}")
    for k, v in sd.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
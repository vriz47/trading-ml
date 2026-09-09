#!/usr/bin/env python3
"""Export trained LSTM ke ONNX, lalu benchmark inference.

Cek kedua provider:
  - CPUExecutionProvider (fallback)
  - NnapiExecutionProvider (GPU/NPU via driver vendor, tersedia di Termux)

Contoh:
  python3 scripts/export_onnx.py --model models/baseline_lstm.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_lstm import SequenceClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "baseline_lstm.pt")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    print(f"cfg: {cfg}")

    model = SequenceClassifier(**{k: cfg[k] for k in ("n_feat", "hidden", "layers")},
                               kind=cfg["kind"]).eval()
    model.load_state_dict(ckpt["state_dict"])

    L = int(cfg["L"])
    n_feat = int(cfg["n_feat"])
    dummy = torch.randn(1, L, n_feat)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(args.out or ROOT / "models" / "baseline_lstm.onnx"),
            input_names=["batch_seqs"], output_names=["logits"],
            opset_version=13, dynamic_axes={"batch_seqs": {0: "N"}},
            dynamo=False,
        )
    print(f"ONNX exported -> {args.out or ROOT / 'models' / 'baseline_lstm.onnx'}")

    import onnxruntime as ort
    path = str(args.out or ROOT / "models" / "baseline_lstm.onnx")
    bench_x = np.random.randn(32, L, n_feat).astype(np.float32)

    for prov in (["CPUExecutionProvider"],
                 ["NnapiExecutionProvider", "CPUExecutionProvider"]):
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(path, so, providers=prov)
            print(f"\nprovider aktif: {sess.get_providers()}")
        except Exception as e:
            print(f"\nprovider {prov} GAGAL: {e}")
            continue

        sess.run(None, {"batch_seqs": bench_x})  # warmup
        t_bench = []
        n = 200
        for _ in range(n):
            t0 = time.perf_counter()
            outs = sess.run(None, {"batch_seqs": bench_x})[0]
            t_bench.append((time.perf_counter() - t0) * 1000)
        t_bench.sort()
        print(f"  N=32 batch: median={t_bench[len(t_bench)//2]:.3f} ms  "
              f"p99={t_bench[int(n*0.99)]:.3f} ms  "
              f"min={t_bench[0]:.3f} ms  (np.argmax ok, shape={outs.shape})")


if __name__ == "__main__":
    main()
# trading-ml

ML pipeline untuk strategi trading MT5 — jalan di Termux (POCO F6).

## Arsitektur

```
training/    training model PyTorch (CPU), retrain berkala dari data MT5
inference/   export ONNX/TFLite + NNAPI/GPU delegate utk analisa real-time
models/      checkpoint .pt / .onnx / .tflite + metadata
data/raw/    hasil export get_rates dari Mt5-Termux-Probe-client (parquet)
data/processed/  feature engineering hasil training
experiments/     log eksperimen
config/          config strategi & model
```

## Stack

- numpy 2.4.4, scipy 1.18.1, pandas 3.0.5 (pkginstall)
- torch 2.11.0 (CPU, training)
- onnxruntime 1.29.0 (inference, provider NNAPI/GPU/NPU)
- tflite-runtime 2.20.0 (cadangan inference)

## Alur

1. `Mt5-Termux-Probe-client` connect ke broker, simpan OHLCV → `data/raw`
2. `training/` train model (LSTM/GRU) → `models/*.onnx`
3. `inference/` load ONNX runtime NNAPI → predict/sub-ms per bar

Pakai mirror cepat: `mirrors.bfsu.edu.cn` + pip parallel (`apt.conf.d/99parallel`).
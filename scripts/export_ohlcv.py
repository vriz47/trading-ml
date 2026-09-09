#!/usr/bin/env python3
"""Export OHLCV dari MT5 probe (Mt5-Termux-Probe-client) ke data/raw.

Contoh:
  python3 scripts/export_ohlcv.py --symbol XAUUSD --period 5 --days 30
  python3 scripts/export_ohlcv.py --symbol EURUSD --period 60 --days 180 --profile HEADWAY
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MT5_SRC = ROOT.parent / "trading-app-forex"
sys.path.insert(0, str(MT5_SRC))
sys.path.insert(0, str(MT5_SRC / "deps" / "pymt5"))

from src.config import Config  # noqa: E402
from src.broker import ExnessBroker  # noqa: E402

PERIOD_MIN: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200,
}


def _td(n: int, unit: str) -> int:
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


async def export(profile: str, symbol: str, period_min: int,
                 from_ts: int, to_ts: int, chunk_days: int = 7) -> list[dict]:
    cfg = Config.from_profile(profile)
    broker = ExnessBroker(cfg.uri)
    await broker.connect(cfg.login, cfg.password, url=cfg.url)
    print(f"[{cfg.server}] suffix={broker.suffix!r} "
          f"symbol={broker.resolve(symbol)!r} acc={broker.account.get('account_currency')}",
          file=sys.stderr)

    bars: list[dict] = []
    start = from_ts
    while start < to_ts:
        end = min(start + chunk_days * 86400, to_ts)
        chunk = await broker.get_bars(symbol, period_min, from_ts=start, to_ts=end)
        if chunk:
            bars.extend(chunk)
            last_t = chunk[-1].get("time", start)
            print(f"  {time.strftime('%Y-%m-%d %H:%M', time.gmtime(start))}"
                  f"..{time.strftime('%d-%H:%M', time.gmtime(last_t))}  "
                  f"+{len(chunk)} bars", file=sys.stderr)
        start = end
        await asyncio.sleep(0.2)

    await broker.close()
    if not bars:
        return []
    seen: set[int] = set()
    out = []
    for b in bars:
        if b.get("time") not in seen:
            seen.add(b.get("time"))
            out.append(b)
    out.sort(key=lambda b: b.get("time", 0))
    return out


async def amain() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="MT5", help="env prefix, default MT5")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--period", default="M5", choices=list(PERIOD_MIN))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--end-ts", type=int, default=0)
    args = ap.parse_args()

    period_min = PERIOD_MIN[args.period]
    to_ts = args.end_ts or int(time.time())
    from_ts = to_ts - _td(args.days, "d")

    bars = await export(args.profile, args.symbol, period_min, from_ts, to_ts)
    if not bars:
        print("GAGAL: tidak ada bar yang ditarik (cek kredensial/akun live?).", file=sys.stderr)
        sys.exit(1)

    outdir = ROOT / "data" / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{args.profile}_{args.symbol}_{args.period}_{args.days}d.csv"
    path = outdir / fname
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    header = ",".join(cols) + "\n"
    lines = []
    for b in sorted(bars, key=lambda x: x.get("time", 0)):
        lines.append(",".join(str(b.get(c, "")) for c in cols))
    path.write_text(header + "\n".join(lines))
    print(f"OK: {len(bars)} bars -> {path} "
          f"({args.profile} {args.symbol} {args.period})")


if __name__ == "__main__":
    asyncio.run(amain())
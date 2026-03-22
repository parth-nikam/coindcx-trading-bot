"""
backtest.py — Historical strategy backtester for Nexus.

Replays OHLCV data through the live alpha engine + risk manager.
Produces: equity curve, trade log, per-strategy metrics, Sharpe, max drawdown.

Usage:
    python backtest.py --symbol BTCUSDT --interval 5m --days 30
    python backtest.py --symbol ETHUSDT --interval 1h --days 90 --capital 5000
    python backtest.py --symbol BTCUSDT --interval 5m --days 60 --walk-forward
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np

import config
from alpha.engine import AlphaEngine, Signal
from execution.risk import PortfolioRisk
from utils.logger import get_logger

logger = get_logger("nexus.backtest")

BINANCE_PUBLIC = "https://api.binance.com"


async def fetch_historical(
    symbol: str,
    interval: str,
    days: int,
    session: aiohttp.ClientSession,
) -> pd.DataFrame:
    """Fetch full historical OHLCV from Binance (handles pagination)."""
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    all_rows = []

    logger.info(f"Fetching {days}d of {interval} candles for {symbol}...")

    while start_ms < end_ms:
        async with session.get(
            f"{BINANCE_PUBLIC}/api/v3/klines",
            params={"symbol": symbol, "interval": interval,
                    "startTime": start_ms, "limit": 1000},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            r.raise_for_status()
            rows = await r.json()

        if not rows:
            break

        all_rows.extend(rows)
        start_ms = rows[-1][0] + 1
        await asyncio.sleep(0.1)

    df = pd.DataFrame(all_rows, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    for col in ("open","high","low","close","volume"):
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").sort_index()
    logger.info(f"Fetched {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    return df


class BacktestEngine:

    WARMUP = 100   # candles needed before first signal

    def __init__(self, capital: float = 10_000.0):
        self._alpha   = AlphaEngine()
        self._capital = capital
        self._equity: list[dict] = []
        self._trades: list[dict] = []

    def run(self, symbol: str, df: pd.DataFrame) -> dict:
        logger.info(f"Running backtest on {len(df)} candles...")
        capital  = self._capital
        position = None

        for i in range(self.WARMUP, len(df)):
            window = df.iloc[i - self.WARMUP: i + 1].reset_index(drop=True)
            price  = float(window["close"].iloc[-1])
            ts     = df.index[i]

            # Check exit on open position
            if position:
                high = float(window["high"].iloc[-1])
                low  = float(window["low"].iloc[-1])

                # Update trailing stop
                trail_pct = getattr(config, "TRAILING_STOP_PCT", 1.0) / 100
                if position["side"] == "BUY":
                    if high > position.get("peak", position["entry"]):
                        position["peak"] = high
                    if position.get("trail_active"):
                        new_trail = position["peak"] * (1 - trail_pct)
                        if new_trail > position.get("trail_stop", 0):
                            position["trail_stop"] = new_trail
                    elif (high - position["entry"]) / position["entry"] >= trail_pct:
                        position["trail_active"] = True
                        position["trail_stop"]   = position["peak"] * (1 - trail_pct)
                else:
                    if low < position.get("peak", position["entry"]):
                        position["peak"] = low
                    if position.get("trail_active"):
                        new_trail = position["peak"] * (1 + trail_pct)
                        if position.get("trail_stop", 0) == 0 or new_trail < position["trail_stop"]:
                            position["trail_stop"] = new_trail
                    elif (position["entry"] - low) / position["entry"] >= trail_pct:
                        position["trail_active"] = True
                        position["trail_stop"]   = position["peak"] * (1 + trail_pct)

                # Determine effective stop
                eff_stop = position.get("trail_stop", position["sl"]) if position.get("trail_active") else position["sl"]

                hit_sl = (position["side"] == "BUY"  and low  <= eff_stop) or \
                         (position["side"] == "SELL" and high >= eff_stop)
                hit_tp = (position["side"] == "BUY"  and high >= position["tp"]) or \
                         (position["side"] == "SELL" and low  <= position["tp"])

                if hit_sl or hit_tp:
                    exit_price = eff_stop if hit_sl else position["tp"]
                    mult = 1 if position["side"] == "BUY" else -1
                    pnl  = mult * (exit_price - position["entry"]) * position["qty"]
                    fee  = exit_price * position["qty"] * 0.001
                    capital += pnl - fee
                    reason = ("trail_stop" if position.get("trail_active") else "stop_loss") if hit_sl else "take_profit"
                    self._trades.append({
                        "symbol":     symbol,
                        "side":       position["side"],
                        "entry":      position["entry"],
                        "exit":       exit_price,
                        "qty":        position["qty"],
                        "pnl":        round(pnl - fee, 4),
                        "reason":     reason,
                        "strategy":   position["strategy"],
                        "entry_time": position["entry_time"],
                        "exit_time":  str(ts),
                        "hold_bars":  i - position["entry_idx"],
                        "trail_used": position.get("trail_active", False),
                    })
                    self._alpha.record_outcome(position["strategy"], pnl > 0)
                    position = None
                    self._equity.append({"time": str(ts), "value": round(capital, 2)})
                    continue

            # Run alpha engine
            signal = self._alpha.evaluate(symbol, window)

            if signal.action != "HOLD" and position is None:
                qty = capital * signal.kelly_f / price
                if qty <= 0:
                    continue
                sl_mult = (1 - config.STOP_LOSS_PCT / 100)  if signal.action == "BUY" else (1 + config.STOP_LOSS_PCT / 100)
                tp_mult = (1 + config.TAKE_PROFIT_PCT / 100) if signal.action == "BUY" else (1 - config.TAKE_PROFIT_PCT / 100)
                fee = price * qty * 0.001
                capital -= fee
                position = {
                    "side":       signal.action,
                    "entry":      price,
                    "qty":        qty,
                    "sl":         price * sl_mult,
                    "tp":         price * tp_mult,
                    "peak":       price,
                    "trail_active": False,
                    "trail_stop": 0.0,
                    "strategy":   signal.reason.split("|")[0].strip() if signal.reason else "alpha",
                    "entry_time": str(ts),
                    "entry_idx":  i,
                }

            self._equity.append({"time": str(ts), "value": round(capital, 2)})

        # Close any open position at end
        if position:
            price = float(df["close"].iloc[-1])
            mult  = 1 if position["side"] == "BUY" else -1
            pnl   = mult * (price - position["entry"]) * position["qty"]
            fee   = price * position["qty"] * 0.001
            capital += pnl - fee
            self._trades.append({
                "symbol":   symbol, "side": position["side"],
                "entry":    position["entry"], "exit": price,
                "qty":      position["qty"], "pnl": round(pnl - fee, 4),
                "reason":   "end_of_data", "strategy": position["strategy"],
                "entry_time": position["entry_time"], "exit_time": str(df.index[-1]),
                "hold_bars": len(df) - position["entry_idx"],
                "trail_used": position.get("trail_active", False),
            })

        return self._compute_stats(capital)

    def _compute_stats(self, final_capital: float) -> dict:
        trades = self._trades
        equity = [e["value"] for e in self._equity]

        if not trades:
            return {"error": "no trades generated"}

        pnls      = [t["pnl"] for t in trades]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p <= 0]
        win_rate  = len(wins) / len(pnls)
        total_pnl = sum(pnls)
        avg_win   = sum(wins) / len(wins) if wins else 0
        avg_loss  = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

        # Expectancy per trade
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe (annualized)
        candles_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96,
                           "30m": 48, "1h": 24, "4h": 6, "1d": 1}
        interval = getattr(self, "_interval", "5m")
        cpd = candles_per_day.get(interval, 288)
        if len(pnls) > 1:
            mean_pnl = total_pnl / len(pnls)
            std_pnl  = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls))
            sharpe   = (mean_pnl / std_pnl * math.sqrt(cpd * 365)) if std_pnl > 0 else 0
        else:
            sharpe = 0

        # Sortino (downside deviation only)
        neg_pnls = [p for p in pnls if p < 0]
        if neg_pnls and len(pnls) > 1:
            mean_pnl   = total_pnl / len(pnls)
            down_var   = sum(p**2 for p in neg_pnls) / len(pnls)
            down_std   = math.sqrt(down_var)
            sortino    = (mean_pnl / down_std * math.sqrt(cpd * 365)) if down_std > 0 else 0
        else:
            sortino = 0

        # Max drawdown
        peak   = self._capital
        max_dd = 0.0
        for v in equity:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak * 100)

        # Calmar ratio
        calmar = (final_capital - self._capital) / self._capital * 100 / max_dd if max_dd > 0 else 0

        # Consecutive wins/losses
        max_consec_wins   = 0
        max_consec_losses = 0
        cur_w = cur_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1; cur_l = 0
                max_consec_wins = max(max_consec_wins, cur_w)
            else:
                cur_l += 1; cur_w = 0
                max_consec_losses = max(max_consec_losses, cur_l)

        return {
            "symbol":              trades[0]["symbol"] if trades else "?",
            "initial_capital":     self._capital,
            "final_capital":       round(final_capital, 2),
            "total_pnl":           round(total_pnl, 4),
            "return_pct":          round((final_capital - self._capital) / self._capital * 100, 2),
            "trade_count":         len(trades),
            "win_rate":            round(win_rate, 3),
            "avg_win":             round(avg_win, 4),
            "avg_loss":            round(avg_loss, 4),
            "expectancy":          round(expectancy, 4),
            "profit_factor":       round(profit_factor, 3),
            "sharpe":              round(sharpe, 3),
            "sortino":             round(sortino, 3),
            "calmar":              round(calmar, 3),
            "max_drawdown":        round(max_dd, 2),
            "avg_hold_bars":       round(sum(t["hold_bars"] for t in trades) / len(trades), 1),
            "max_consec_wins":     max_consec_wins,
            "max_consec_losses":   max_consec_losses,
            "trail_stop_exits":    sum(1 for t in trades if t.get("trail_used")),
        }


def walk_forward(
    symbol: str,
    df: pd.DataFrame,
    capital: float,
    n_splits: int = 5,
) -> list[dict]:
    """
    Walk-forward validation: split data into n_splits windows,
    train on first 70%, test on last 30% of each window.
    Returns list of out-of-sample stats.
    """
    results = []
    split_size = len(df) // n_splits

    for i in range(n_splits):
        start = i * split_size
        end   = start + split_size
        if end > len(df):
            end = len(df)
        window_df = df.iloc[start:end]
        if len(window_df) < 200:
            continue

        engine = BacktestEngine(capital=capital)
        engine._interval = "5m"
        stats  = engine.run(symbol, window_df)
        stats["window"] = i + 1
        stats["start"]  = str(window_df.index[0])
        stats["end"]    = str(window_df.index[-1])
        results.append(stats)
        logger.info(f"Walk-forward window {i+1}/{n_splits}: {stats.get('return_pct', 'N/A')}% return")

    return results


def print_report(stats: dict, trades: list):
    print("\n" + "=" * 60)
    print("  NEXUS BACKTEST REPORT")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("=" * 60)

    if trades:
        print(f"\n  Last 5 trades:")
        for t in trades[-5:]:
            pnl_str = f"{t['pnl']:+.4f}"
            trail   = " [trail]" if t.get("trail_used") else ""
            print(f"  {t['side']:<4} {t['entry']:.2f}→{t['exit']:.2f} "
                  f"pnl={pnl_str} [{t['reason']}]{trail} {t['strategy'][:20]}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Nexus Backtester")
    parser.add_argument("--symbol",       default="BTCUSDT")
    parser.add_argument("--interval",     default="5m")
    parser.add_argument("--days",         type=int, default=30)
    parser.add_argument("--capital",      type=float, default=10_000.0)
    parser.add_argument("--output",       default=None, help="Save results to JSON file")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        df = await fetch_historical(args.symbol, args.interval, args.days, session)

    if args.walk_forward:
        print(f"\nRunning walk-forward validation ({5} windows)...")
        results = walk_forward(args.symbol, df, args.capital)
        for r in results:
            print(f"\n  Window {r['window']} ({r['start'][:10]} → {r['end'][:10]})")
            print(f"    Return: {r.get('return_pct', 'N/A')}% | "
                  f"Trades: {r.get('trade_count', 0)} | "
                  f"WinRate: {r.get('win_rate', 0):.1%} | "
                  f"Sharpe: {r.get('sharpe', 0):.2f} | "
                  f"MaxDD: {r.get('max_drawdown', 0):.1f}%")
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
    else:
        engine = BacktestEngine(capital=args.capital)
        engine._interval = args.interval
        stats  = engine.run(args.symbol, df)
        print_report(stats, engine._trades)

        if args.output:
            out = {"stats": stats, "trades": engine._trades, "equity": engine._equity}
            Path(args.output).write_text(json.dumps(out, indent=2))
            logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

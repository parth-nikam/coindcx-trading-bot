"""
alpha/engine.py — Strategy orchestrator v5

Go-mirrored strategy mix:
  - trend_follow (0.35) — EMA200 + stack + MACD + RSI
  - momentum_burst (0.28) — large body + volume surge + EMA200
  - vol_breakout (0.22) — Donchian breakout + volume + EMA200
  - stoch_rsi (0.15) — K/D crossover with EMA200 filter

All strategies require EMA200 alignment — only trade with the long-term trend.
ADX gate: require ADX > 20 to avoid choppy markets.
MIN_AGREEING: at least 2 strategies must agree.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from alpha.strategies.trend_follow import TrendFollow
from alpha.strategies.momentum_burst import MomentumBurst
from alpha.strategies.vol_breakout import VolBreakout
from alpha.strategies.stoch_rsi import StochRSI
from alpha.strategies.base import Vote
from utils.logger import get_logger

logger = get_logger(__name__)

BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass
class Signal:
    symbol:    str
    action:    str          # BUY | SELL | HOLD
    score:     float        # 0.0 – 1.0 weighted confidence
    kelly_f:   float        # Kelly fraction for position sizing
    breakdown: dict         # per-strategy votes
    reason:    str = ""
    agreeing:  int = 0      # how many strategies agreed
    quality:   float = 0.0  # signal quality 0-1


@dataclass
class StrategyStats:
    """Rolling win rate tracker — last 50 trades."""
    _results: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def win_rate(self) -> float:
        if not self._results:
            return 0.5
        return sum(self._results) / len(self._results)

    @property
    def kelly_fraction(self) -> float:
        w = self.win_rate
        r = 2.0   # assume 2:1 reward/risk
        f = w - (1 - w) / r
        return max(0.0, min(0.25, f / 2))  # half-Kelly, capped 25%

    def record(self, won: bool):
        self._results.append(1 if won else 0)


class AlphaEngine:
    """
    Runs 4 Go-mirrored strategies, aggregates with fixed weights, returns Signal.
    """

    MIN_CANDLES = 210

    # Base weights — must sum to 1.0 (mirrors Go baseWeights)
    BASE_WEIGHTS = {
        "trend_follow":   0.35,
        "momentum_burst": 0.28,
        "vol_breakout":   0.22,
        "stoch_rsi":      0.15,
    }

    def __init__(self):
        self._strategies = [
            TrendFollow(),
            MomentumBurst(),
            VolBreakout(),
            StochRSI(),
        ]
        for s in self._strategies:
            s.weight = self.BASE_WEIGHTS[s.name]

        self._stats: dict[str, StrategyStats] = {
            s.name: StrategyStats() for s in self._strategies
        }
        self._last_votes: dict[str, dict] = {}

    def update_funding(self, rate: float):
        pass  # no funding strategy in this version

    def record_outcome(self, strategy_name: str, won: bool):
        if strategy_name in self._stats:
            self._stats[strategy_name].record(won)

    def evaluate(self, symbol: str, df: pd.DataFrame, ob: dict | None = None) -> Signal:
        if len(df) < self.MIN_CANDLES:
            return Signal(symbol, HOLD, 0.0, 0.0, {})

        votes: dict[str, Vote] = {}
        for strat in self._strategies:
            try:
                votes[strat.name] = strat.vote(df, ob)
            except Exception as e:
                logger.error(f"[{symbol}] {strat.name} error: {e}")
                votes[strat.name] = Vote(HOLD, 0.0, f"error: {e}")

        signal, score, breakdown, agreeing, quality = self._aggregate(votes)

        # ADX gate — require trending market (ADX > 20)
        if signal != HOLD:
            try:
                import ta as _ta
                adx_val = _ta.trend.ADXIndicator(
                    high=df["high"], low=df["low"], close=df["close"], window=14
                ).adx().iloc[-1]
                if not pd.isna(adx_val) and adx_val < 20:
                    logger.debug(f"[{symbol}] Signal {signal} blocked — ADX={adx_val:.1f} < 20")
                    signal = HOLD
                    score  = 0.0
                    quality = 0.0
            except Exception:
                pass

        # Confirmation filter — require MIN_AGREEING strategies
        min_agreeing = getattr(config, "MIN_AGREEING", 2)
        if signal != HOLD and agreeing < min_agreeing:
            logger.debug(f"[{symbol}] Signal {signal} blocked — only {agreeing} agreeing (need {min_agreeing})")
            signal = HOLD
            score  = 0.0
            quality = 0.0

        kelly = self._kelly_for_signal(signal, votes)

        reason = " | ".join(
            f"{k}={v.signal}({v.conf:.2f})"
            for k, v in votes.items()
            if v.signal != HOLD
        )

        logger.info(
            f"[{symbol}] {signal} score={score:.3f} kelly={kelly:.3f} "
            f"agree={agreeing} quality={quality:.3f} | {reason or 'all_hold'}"
        )

        self._last_votes[symbol] = {
            k: {"signal": v.signal, "conf": round(v.conf, 3), "reason": v.reason}
            for k, v in votes.items()
        }

        return Signal(symbol, signal, score, kelly, breakdown, reason, agreeing, quality)

    def _aggregate(self, votes: dict[str, Vote]) -> tuple[str, float, dict, int, float]:
        buy_score  = 0.0
        sell_score = 0.0
        buy_count  = 0
        sell_count = 0
        buy_conf_sum  = 0.0
        sell_conf_sum = 0.0
        breakdown  = {}

        total_weight = sum(s.weight for s in self._strategies)
        if total_weight == 0:
            total_weight = 1.0

        for strat in self._strategies:
            v = votes.get(strat.name, Vote(HOLD, 0.0))
            breakdown[strat.name] = {
                "signal": v.signal, "conf": v.conf,
                "reason": v.reason, "weight": round(strat.weight, 3)
            }
            norm_w = strat.weight / total_weight
            if v.signal == BUY:
                buy_score     += norm_w * v.conf
                buy_count     += 1
                buy_conf_sum  += v.conf
            elif v.signal == SELL:
                sell_score    += norm_w * v.conf
                sell_count    += 1
                sell_conf_sum += v.conf

        n = len(self._strategies)
        net = buy_score - sell_score

        if net > 0:
            avg_conf = buy_conf_sum / buy_count if buy_count else 0
            quality  = round(buy_score * (buy_count / n) * avg_conf, 4)
            if buy_score >= config.BUY_THRESHOLD:
                return BUY, buy_score, breakdown, buy_count, quality
        else:
            avg_conf = sell_conf_sum / sell_count if sell_count else 0
            quality  = round(sell_score * (sell_count / n) * avg_conf, 4)
            if sell_score >= config.SELL_THRESHOLD:
                return SELL, sell_score, breakdown, sell_count, quality

        best = buy_score if buy_score > sell_score else sell_score
        best_count = buy_count if buy_score > sell_score else sell_count
        return HOLD, best, breakdown, best_count, 0.0

    def _kelly_for_signal(self, signal: str, votes: dict[str, Vote]) -> float:
        if signal == HOLD:
            return 0.0
        fractions = [
            self._stats[s.name].kelly_fraction
            for s in self._strategies
            if votes.get(s.name, Vote(HOLD, 0.0)).signal == signal
        ]
        base = (
            max(config.TRADE_SIZE_PCT * 0.5, min(config.TRADE_SIZE_PCT, sum(fractions) / len(fractions)))
            if fractions else config.TRADE_SIZE_PCT
        )
        return round(min(base, config.TRADE_SIZE_PCT * 1.5), 4)

    @property
    def last_votes(self) -> dict:
        return self._last_votes

# VWAP Deviation — mean reversion against volume-weighted price anchor
# Edge: institutional algos defend VWAP. Deviations > 1.5 ATR snap back.
# v2: looser thresholds, added trend-following mode when price far from VWAP

import pandas as pd
from .base import BaseStrategy, Vote


class VWAPDeviation(BaseStrategy):
    name   = "vwap_dev"
    weight = 0.13

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]
        vwap  = self.vwap(df)
        atr   = self.atr(df, 14)
        rsi   = self.rsi(close, 14)
        e20   = self.ema(close, 20)

        p, v, a, r = close.iloc[-1], vwap.iloc[-1], atr.iloc[-1], rsi.iloc[-1]
        if a == 0 or pd.isna(v) or pd.isna(r):
            return Vote("HOLD", 0.0)

        dev = (v - p) / a  # positive = price below VWAP (oversold vs VWAP)
        trend_up = e20.iloc[-1] > e20.iloc[-5]  # short-term trend direction

        # Strong mean reversion signals only — require clear extremes
        if dev > 2.5 and r < 35:
            return Vote("BUY",  min(1.0, dev / 4), f"dev={dev:.2f}σ RSI={r:.0f} strong_rev")
        if dev < -2.5 and r > 65:
            return Vote("SELL", min(1.0, abs(dev) / 4), f"dev={dev:.2f}σ RSI={r:.0f} strong_rev")

        # Moderate mean reversion — tighter RSI requirement
        if dev > 2.0 and r < 38:
            return Vote("BUY",  min(0.75, dev / 4), f"dev={dev:.2f}σ RSI={r:.0f}")
        if dev < -2.0 and r > 62:
            return Vote("SELL", min(0.75, abs(dev) / 4), f"dev={dev:.2f}σ RSI={r:.0f}")

        return Vote("HOLD", 0.0, f"dev={dev:.2f}σ RSI={r:.0f}")

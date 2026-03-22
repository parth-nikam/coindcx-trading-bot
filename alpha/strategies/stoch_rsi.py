# Stochastic RSI — momentum oscillator with overbought/oversold + crossover
# Edge: StochRSI is more sensitive than plain RSI — catches turns earlier.
# v2: added mid-zone momentum signals, trend-aligned signals

import pandas as pd
import ta
from .base import BaseStrategy, Vote


class StochRSI(BaseStrategy):
    name   = "stoch_rsi"
    weight = 0.11

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]

        ind = ta.momentum.StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        k = ind.stochrsi_k()
        d = ind.stochrsi_d()

        if k.dropna().empty or d.dropna().empty:
            return Vote("HOLD", 0.0)

        k0, k1 = k.iloc[-1], k.iloc[-2]
        d0, d1 = d.iloc[-1], d.iloc[-2]

        if any(pd.isna(x) for x in [k0, k1, d0, d1]):
            return Vote("HOLD", 0.0)

        # Trend context from EMA
        e20 = self.ema(close, 20)
        e50 = self.ema(close, 50)
        trend_up = e20.iloc[-1] > e50.iloc[-1]

        # Bullish crossover in oversold zone
        if k1 < d1 and k0 > d0 and k0 < 0.30:
            conf = min(1.0, (0.30 - k0) * 3.5 + 0.5)
            if trend_up:
                conf = min(1.0, conf + 0.1)
            return Vote("BUY", conf, f"stoch_cross_up k={k0:.2f} d={d0:.2f}")

        # Bearish crossover in overbought zone
        if k1 > d1 and k0 < d0 and k0 > 0.70:
            conf = min(1.0, (k0 - 0.70) * 3.5 + 0.5)
            if not trend_up:
                conf = min(1.0, conf + 0.1)
            return Vote("SELL", conf, f"stoch_cross_dn k={k0:.2f} d={d0:.2f}")

        # Deep oversold / overbought without crossover
        if k0 < 0.10 and d0 < 0.15:
            return Vote("BUY",  0.50, f"stoch_oversold k={k0:.2f}")
        if k0 > 0.90 and d0 > 0.85:
            return Vote("SELL", 0.50, f"stoch_overbought k={k0:.2f}")

        # Mid-zone momentum: K rising from below 0.5 in uptrend
        if k0 > 0.45 and k1 < 0.45 and trend_up:
            return Vote("BUY",  0.40, f"stoch_mid_cross_up k={k0:.2f}")
        if k0 < 0.55 and k1 > 0.55 and not trend_up:
            return Vote("SELL", 0.40, f"stoch_mid_cross_dn k={k0:.2f}")

        return Vote("HOLD", 0.0, f"k={k0:.2f} d={d0:.2f}")

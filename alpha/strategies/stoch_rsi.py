# Stochastic RSI — K/D crossover with EMA200 trend filter (ported from Go)
# K crosses above D from oversold = BUY (only if above EMA200 or strong ADX)
# K crosses below D from overbought = SELL (only if below EMA200 or strong ADX)

import pandas as pd
import ta
from .base import BaseStrategy, Vote


class StochRSI(BaseStrategy):
    name   = "stoch_rsi"
    weight = 0.15

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

        rsi = self.rsi(close, 14).iloc[-1]
        adx_v, _, _ = self.adx(df, 14)
        adx = adx_v.iloc[-1]

        # EMA200 trend alignment
        e200 = self.ema(close, 200).iloc[-1]
        above_e200 = close.iloc[-1] > e200
        below_e200 = close.iloc[-1] < e200

        # K crossing above D from oversold zone (< 0.30)
        k_cross_up   = k0 > d0 and k1 <= d1
        # K crossing below D from overbought zone (> 0.70)
        k_cross_down = k0 < d0 and k1 >= d1

        # Bullish: K crosses up from oversold, RSI not overbought
        # Allow if above EMA200 OR ADX very strong (>30)
        trend_ok = above_e200 or adx > 30
        if k_cross_up and k1 < 0.30 and rsi < 68 and trend_ok:
            base_conf = 0.78 if adx > 25 else 0.65
            if not above_e200:
                base_conf -= 0.10  # reduce confidence if against EMA200
            conf = min(0.90, base_conf + (0.25 - k1) * 0.5)
            return Vote("BUY", conf, f"srsi_bull k={k0:.2f} d={d0:.2f} rsi={rsi:.0f} adx={adx:.0f}")

        # Bearish: K crosses down from overbought, RSI not oversold
        trend_ok_sell = below_e200 or adx > 30
        if k_cross_down and k1 > 0.70 and rsi > 32 and trend_ok_sell:
            base_conf = 0.78 if adx > 25 else 0.65
            if not below_e200:
                base_conf -= 0.10
            conf = min(0.90, base_conf + (k1 - 0.75) * 0.5)
            return Vote("SELL", conf, f"srsi_bear k={k0:.2f} d={d0:.2f} rsi={rsi:.0f} adx={adx:.0f}")

        return Vote("HOLD", 0.0, f"k={k0:.2f} d={d0:.2f} rsi={rsi:.0f}")

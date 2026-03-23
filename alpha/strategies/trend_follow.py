# TrendFollow — Multi-timeframe EMA trend following (ported from Go)
# Primary signal: EMA50/200 alignment + EMA8/21 stack + MACD + RSI confirmation.
# Only trades in the direction of the long-term trend (EMA200).

import pandas as pd
import numpy as np
from .base import BaseStrategy, Vote


class TrendFollow(BaseStrategy):
    name   = "trend_follow"
    weight = 0.35

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        if len(df) < 210:
            return Vote("HOLD", 0.0)

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        e8   = self.ema(close, 8)
        e21  = self.ema(close, 21)
        e50  = self.ema(close, 50)
        e200 = self.ema(close, 200)
        rsi  = self.rsi(close, 14)
        adx_v, dip, din = self.adx(df, 14)

        n = len(close) - 1
        p    = close.iloc[-1]
        r    = rsi.iloc[-1]
        adx  = adx_v.iloc[-1]
        dp   = dip.iloc[-1]
        dn   = din.iloc[-1]
        v8   = e8.iloc[-1]
        v21  = e21.iloc[-1]
        v50  = e50.iloc[-1]
        v200 = e200.iloc[-1]

        if any(pd.isna(x) for x in [r, adx, dp, dn, v8, v21, v50, v200]):
            return Vote("HOLD", 0.0)

        # ADX must show trending market
        if adx < 15:
            return Vote("HOLD", 0.0, f"adx_weak={adx:.0f}")

        # MACD momentum confirmation
        import ta
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        hist = macd_ind.macd_diff()
        if hist.dropna().empty:
            return Vote("HOLD", 0.0)
        h = hist.iloc[-1]

        # EMA stack alignment
        bull_stack = v8 > v21 and v21 > v50
        bear_stack = v8 < v21 and v21 < v50

        # Long-term trend: price vs EMA200
        above_e200 = p > v200
        below_e200 = p < v200

        # EMA200 slope over 20 bars
        v200_20 = e200.iloc[-21]
        e200_slope = (v200 - v200_20) / v200_20 if v200_20 > 0 else 0

        # EMA spread in ATR units
        atr_v = self.atr(df, 14).iloc[-1]
        spread = abs(v8 - v50) / atr_v if atr_v > 0 else 0

        # BULL: price above e200, e200 rising, bull stack, MACD positive, RSI 42-78
        if above_e200 and e200_slope > 0 and bull_stack and h > 0 and 42 <= r <= 78 and dp > dn:
            conf = min(0.92, 0.62 + (adx - 15) / 80 + min(0.12, spread * 0.04))
            if e200_slope > 0.0002:
                conf = min(0.92, conf + 0.05)
            return Vote("BUY", conf, f"tf_bull adx={adx:.0f} rsi={r:.0f} e200slope={e200_slope:.4f}")

        # BEAR: price below e200, e200 falling, bear stack, MACD negative, RSI 22-58
        if below_e200 and e200_slope < 0 and bear_stack and h < 0 and 22 <= r <= 58 and dn > dp:
            conf = min(0.92, 0.62 + (adx - 15) / 80 + min(0.12, spread * 0.04))
            if e200_slope < -0.0002:
                conf = min(0.92, conf + 0.05)
            return Vote("SELL", conf, f"tf_bear adx={adx:.0f} rsi={r:.0f} e200slope={e200_slope:.4f}")

        return Vote("HOLD", 0.0, f"aboveE200={above_e200} bullStack={bull_stack} adx={adx:.0f}")

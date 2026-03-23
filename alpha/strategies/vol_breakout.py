# VolatilityBreakout — Donchian channel breakout with multi-factor confirmation (ported from Go)
# Requires: 2-bar close confirmation + volume surge + ADX > 22 + MACD + EMA200 alignment.

import pandas as pd
import numpy as np
from .base import BaseStrategy, Vote


class VolBreakout(BaseStrategy):
    name   = "vol_breakout"
    weight = 0.22

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        if len(df) < 210:
            return Vote("HOLD", 0.0)

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        adx_v, dip, din = self.adx(df, 14)
        adx = adx_v.iloc[-1]
        if adx < 22:
            return Vote("HOLD", 0.0, f"adx_weak={adx:.0f}")

        # 20-bar Donchian channel (excluding last 2 bars for confirmation)
        period = 20
        dc_high = high.iloc[-(period + 2):-2].max()
        dc_low  = low.iloc[-(period + 2):-2].min()

        vol_avg   = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / max(vol_avg, 1e-9)

        rsi = self.rsi(close, 14).iloc[-1]
        dp  = dip.iloc[-1]
        dn  = din.iloc[-1]

        # MACD momentum
        import ta
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        hist = macd_ind.macd_diff()
        if hist.dropna().shape[0] < 2:
            return Vote("HOLD", 0.0)
        h0, h1 = hist.iloc[-1], hist.iloc[-2]
        macd_bull = h0 > 0 and h0 > h1
        macd_bear = h0 < 0 and h0 < h1

        # EMA200 alignment
        e200 = self.ema(close, 200).iloc[-1]
        above_e200 = close.iloc[-1] > e200
        below_e200 = close.iloc[-1] < e200

        price     = close.iloc[-1]
        prev_close = close.iloc[-2]

        # Breakout up: 2 consecutive closes above 20-bar high + volume + ADX + MACD + above e200
        if price > dc_high and prev_close > dc_high and vol_ratio > 2.0 and dp > dn and macd_bull and 52 < rsi < 82 and above_e200:
            conf = min(0.93, 0.68 + (adx - 22) / 70 + min(0.12, (vol_ratio - 2.0) * 0.06))
            return Vote("BUY", conf, f"vb_up adx={adx:.0f} vol={vol_ratio:.1f}x rsi={rsi:.0f}")

        # Breakout down: below e200
        if price < dc_low and prev_close < dc_low and vol_ratio > 2.0 and dn > dp and macd_bear and 18 < rsi < 48 and below_e200:
            conf = min(0.93, 0.68 + (adx - 22) / 70 + min(0.12, (vol_ratio - 2.0) * 0.06))
            return Vote("SELL", conf, f"vb_dn adx={adx:.0f} vol={vol_ratio:.1f}x rsi={rsi:.0f}")

        return Vote("HOLD", 0.0, f"dc=[{dc_low:.0f}-{dc_high:.0f}] adx={adx:.0f} vol={vol_ratio:.1f}x")

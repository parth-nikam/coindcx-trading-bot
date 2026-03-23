# MomentumBurst — large body candle with volume surge + trend alignment (ported from Go)
# Fires on genuine momentum candles that signal continuation.
# Requires ADX > 22 + EMA200 alignment + MACD confirmation.

import pandas as pd
import numpy as np
from .base import BaseStrategy, Vote


class MomentumBurst(BaseStrategy):
    name   = "momentum_burst"
    weight = 0.28

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        if len(df) < 210:
            return Vote("HOLD", 0.0)

        close  = df["close"]
        open_  = df["open"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        atr_s = self.atr(df, 14)
        atr_v = atr_s.iloc[-1]
        if atr_v == 0:
            return Vote("HOLD", 0.0)

        adx_v, _, _ = self.adx(df, 14)
        adx = adx_v.iloc[-1]
        if adx < 22:
            return Vote("HOLD", 0.0, f"adx_weak={adx:.0f}")

        # Volume average over 20 bars
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_now = volume.iloc[-1]
        vol_ratio = vol_now / max(vol_avg, 1e-9)

        rsi = self.rsi(close, 14).iloc[-1]

        # EMA trend direction
        e21 = self.ema(close, 21).iloc[-1]
        e50 = self.ema(close, 50).iloc[-1]
        e200 = self.ema(close, 200).iloc[-1]
        trend_up   = e21 > e50
        trend_down = e21 < e50
        above_e200 = close.iloc[-1] > e200
        below_e200 = close.iloc[-1] < e200

        # MACD confirmation
        import ta
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        hist = macd_ind.macd_diff()
        if hist.dropna().empty:
            return Vote("HOLD", 0.0)
        h = hist.iloc[-1]
        macd_bull = h > 0
        macd_bear = h < 0

        body = close.iloc[-1] - open_.iloc[-1]
        body_abs = abs(body)

        # Candle close position (close near top/bottom = strong)
        candle_range = high.iloc[-1] - low.iloc[-1]
        close_pos = (close.iloc[-1] - low.iloc[-1]) / candle_range if candle_range > 0 else 0.5

        # Strong burst: body > 2×ATR, volume > 2.5×, trend + MACD + e200 aligned
        if body > atr_v * 2.0 and vol_ratio > 2.5 and trend_up and macd_bull and above_e200 and rsi < 78 and close_pos > 0.75:
            conf = min(0.93, 0.68 + body_abs / atr_v * 0.03 + min(0.12, (vol_ratio - 2.5) * 0.06))
            return Vote("BUY", conf, f"burst_bull body/atr={body_abs/atr_v:.1f} vol={vol_ratio:.1f}x rsi={rsi:.0f}")

        if body < -atr_v * 2.0 and vol_ratio > 2.5 and trend_down and macd_bear and below_e200 and rsi > 22 and close_pos < 0.25:
            conf = min(0.93, 0.68 + body_abs / atr_v * 0.03 + min(0.12, (vol_ratio - 2.5) * 0.06))
            return Vote("SELL", conf, f"burst_bear body/atr={body_abs/atr_v:.1f} vol={vol_ratio:.1f}x rsi={rsi:.0f}")

        # Moderate burst: body > 1.8×ATR, volume > 2×
        if body > atr_v * 1.8 and vol_ratio > 2.0 and trend_up and macd_bull and above_e200 and rsi < 75 and close_pos > 0.70:
            return Vote("BUY", 0.65, f"burst_bull_mod body/atr={body_abs/atr_v:.1f} vol={vol_ratio:.1f}x")
        if body < -atr_v * 1.8 and vol_ratio > 2.0 and trend_down and macd_bear and below_e200 and rsi > 25 and close_pos < 0.30:
            return Vote("SELL", 0.65, f"burst_bear_mod body/atr={body_abs/atr_v:.1f} vol={vol_ratio:.1f}x")

        return Vote("HOLD", 0.0, f"body/atr={body_abs/atr_v:.2f} vol={vol_ratio:.1f}x adx={adx:.0f}")

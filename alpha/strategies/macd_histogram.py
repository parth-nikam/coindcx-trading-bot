# MACD Histogram — momentum acceleration + zero-line cross
# Edge: histogram slope change = momentum shift before price confirms.
# v2: added signal line cross as standalone signal, histogram divergence

import pandas as pd
import ta
from .base import BaseStrategy, Vote


class MACDHistogram(BaseStrategy):
    name   = "macd_hist"
    weight = 0.11

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]

        ind  = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd = ind.macd()
        sig  = ind.macd_signal()
        hist = ind.macd_diff()

        if hist.dropna().shape[0] < 5:
            return Vote("HOLD", 0.0)

        h0, h1, h2 = hist.iloc[-1], hist.iloc[-2], hist.iloc[-3]
        m0, s0     = macd.iloc[-1], sig.iloc[-1]
        m1, s1     = macd.iloc[-2], sig.iloc[-2]

        if any(pd.isna(x) for x in [h0, h1, h2, m0, s0]):
            return Vote("HOLD", 0.0)

        # MACD line zero-cross (strongest signal)
        macd_cross_up   = m1 < 0 < m0
        macd_cross_down = m1 > 0 > m0

        # Signal line cross
        sig_cross_up   = m1 < s1 and m0 > s0
        sig_cross_down = m1 > s1 and m0 < s0

        # Histogram momentum
        hist_turning_up   = h2 < h1 and h0 > h1  # any turn up
        hist_turning_down = h2 > h1 and h0 < h1  # any turn down
        hist_accel_up     = h0 > h1 > h2 and h0 > 0  # accelerating positive
        hist_accel_down   = h0 < h1 < h2 and h0 < 0  # accelerating negative

        if macd_cross_up:
            return Vote("BUY",  0.85, f"macd_zero_cross_up h={h0:.4f}")
        if macd_cross_down:
            return Vote("SELL", 0.85, f"macd_zero_cross_dn h={h0:.4f}")

        if sig_cross_up and m0 < 0:
            return Vote("BUY",  0.70, f"sig_cross_up below_zero h={h0:.4f}")
        if sig_cross_down and m0 > 0:
            return Vote("SELL", 0.70, f"sig_cross_dn above_zero h={h0:.4f}")

        if hist_accel_up and sig_cross_up:
            return Vote("BUY",  0.65, f"hist_accel+sig h={h0:.4f}")
        if hist_accel_down and sig_cross_down:
            return Vote("SELL", 0.65, f"hist_accel+sig h={h0:.4f}")

        return Vote("HOLD", 0.0, f"macd={m0:.4f} sig={s0:.4f}")

# Regime Filter — ADX-gated trend/range mode switcher
# Edge: different strategies work in different regimes. This routes correctly.
# v2: smoother ADX thresholds, added squeeze detection, better range signals

import pandas as pd
import ta
from .base import BaseStrategy, Vote


class RegimeFilter(BaseStrategy):
    name   = "regime"
    weight = 0.12

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]
        adx_v, dip, din = self.adx(df, 14)
        rsi   = self.rsi(close, 14)
        bb    = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
        e20   = self.ema(close, 20)
        e50   = self.ema(close, 50)

        a   = adx_v.iloc[-1]
        r   = rsi.iloc[-1]
        p   = close.iloc[-1]
        bbu = bb.bollinger_hband().iloc[-1]
        bbl = bb.bollinger_lband().iloc[-1]
        bbm = bb.bollinger_mavg().iloc[-1]
        d20 = e20.iloc[-1]
        d50 = e50.iloc[-1]
        dp  = dip.iloc[-1]
        dn  = din.iloc[-1]

        if any(pd.isna(x) for x in [a, r, bbu, bbl]):
            return Vote("HOLD", 0.0)

        bb_width = (bbu - bbl) / bbm if bbm > 0 else 0

        if a > 22:
            # Trending regime — follow DI crossover
            if dp > dn and d20 > d50:
                conf = min(1.0, a / 45)
                return Vote("BUY",  conf, f"trend adx={a:.0f} +DI={dp:.0f}")
            if dn > dp and d20 < d50:
                conf = min(1.0, a / 45)
                return Vote("SELL", conf, f"trend adx={a:.0f} -DI={dn:.0f}")
            # Trend but DI not aligned — weak signal
            if dp > dn:
                return Vote("BUY",  0.35, f"trend_weak adx={a:.0f}")
            if dn > dp:
                return Vote("SELL", 0.35, f"trend_weak adx={a:.0f}")

        if a < 22:
            # Range regime — fade extremes
            if p <= bbl * 1.003 and r < 38:
                return Vote("BUY",  0.72, f"range bbl={bbl:.0f} rsi={r:.0f}")
            if p >= bbu * 0.997 and r > 62:
                return Vote("SELL", 0.72, f"range bbu={bbu:.0f} rsi={r:.0f}")

            # Approaching extremes
            if p < bbm and r < 42:
                return Vote("BUY",  0.40, f"range_mid_low rsi={r:.0f}")
            if p > bbm and r > 58:
                return Vote("SELL", 0.40, f"range_mid_high rsi={r:.0f}")

        return Vote("HOLD", 0.0, f"transitioning adx={a:.0f}")

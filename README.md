# Nexus Trading System

Production-grade async crypto trading bot — Binance, 9-strategy alpha engine, Kelly sizing, FastAPI dashboard.

## Architecture

```
nexus/
├── bot.py                  # Entry point — async event loop
├── config.py               # All configuration + startup validation
├── backtest.py             # Historical backtester
│
├── alpha/
│   ├── engine.py           # Strategy orchestrator (voting, Kelly, weight adaptation)
│   └── strategies/
│       ├── vwap_deviation.py    # Mean reversion vs VWAP anchor
│       ├── ema_ribbon.py        # 4-EMA trend stack
│       ├── ttm_squeeze.py       # Bollinger/Keltner coil breakout
│       ├── rsi_divergence.py    # Regular + hidden RSI divergence
│       ├── regime_filter.py     # ADX-gated trend/range switcher
│       ├── microstructure.py    # Order book imbalance
│       ├── funding_rate.py      # Perpetual funding carry signal
│       ├── stoch_rsi.py         # Stochastic RSI crossover
│       └── macd_histogram.py    # MACD histogram momentum
│
├── exchange/
│   ├── base.py             # Abstract exchange interface
│   ├── paper.py            # Paper trading engine (real Binance prices)
│   └── binance.py          # Live Binance REST + WebSocket adapter
│
├── execution/
│   ├── risk.py             # Portfolio risk manager (SL/TP, drawdown, Kelly)
│   └── router.py           # Smart order router (limit/market, vol-adjusted sizing)
│
├── dashboard/
│   ├── app.py              # FastAPI app
│   └── templates/index.html # TradingView charts + live dashboard
│
└── utils/
    ├── logger.py
    ├── retry.py            # Exponential backoff decorator
    └── circuit_breaker.py  # Per-symbol circuit breaker
```

## Quick Start

```bash
cd nexus
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Paper trading (default)
venv/bin/python3 bot.py

# Dashboard → http://localhost:8080
```

## Live Trading

```bash
# Add keys to .env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=false
PAPER_TRADING=false

venv/bin/python3 bot.py
```

## Backtesting

```bash
# 30 days of BTCUSDT 5m candles
venv/bin/python3 backtest.py --symbol BTCUSDT --interval 5m --days 30

# Save results to JSON
venv/bin/python3 backtest.py --symbol ETHUSDT --days 14 --output results.json
```

## Alpha Engine

9 strategies vote independently. Weighted scores aggregated — signal fires when:
1. Weighted score > threshold (0.35)
2. At least 2 strategies agree (confirmation filter)
3. No circuit breaker open for that symbol

Weights adapt every 10 minutes based on recent win rates (80% base, 20% performance).

| Strategy | Base Weight | Edge |
|---|---|---|
| VWAP Deviation | 14% | Mean reversion to institutional anchor |
| EMA Ribbon | 14% | 4-EMA trend stack alignment |
| TTM Squeeze | 13% | Bollinger/Keltner coil breakout |
| RSI Divergence | 11% | Momentum exhaustion before reversal |
| Regime Filter | 11% | ADX-gated trend/range mode |
| Stoch RSI | 10% | K/D crossover in extreme zones |
| Microstructure | 9% | Order book imbalance |
| Funding Rate | 9% | Fade crowded perpetual positions |
| MACD Histogram | 9% | Momentum acceleration + zero-cross |

## Risk Controls

- Stop loss: 1.5% from entry
- Take profit: 3.0% from entry
- Max daily loss: 5% → auto-halt
- Max drawdown: 15% → auto-halt
- Max concurrent positions: 3
- Order timeout: 5 min (stale limits auto-cancelled)
- Circuit breaker: 5 consecutive errors → 5-min cooldown
- Volatility-adjusted sizing: high-vol regime scales position down

## Configuration

All parameters in `config.py`. Key settings:

```python
SYMBOLS         = ["BTCUSDT", "ETHUSDT"]
PAPER_TRADING   = True
INITIAL_CAPITAL = 10_000
STOP_LOSS_PCT   = 1.5
TAKE_PROFIT_PCT = 3.0
BUY_THRESHOLD   = 0.35
```

"""
dashboard/app.py — FastAPI dashboard for Nexus Trading System.

REST:
  GET  /              — HTML dashboard
  GET  /api/status    — portfolio snapshot
  GET  /api/trades    — trade history
  GET  /api/positions — open positions
  GET  /api/candles/{symbol} — OHLCV for chart
  POST /api/control   — force BUY/SELL/HALT

WebSocket:
  WS /ws  — pushes every second: prices, pnl, strategy scores, equity curve point
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from bot import NexusBot

app = FastAPI(title="Nexus", version="1.0.0")

_bot: "NexusBot | None" = None
_ws_clients: list[WebSocket] = []

# Rolling equity curve — list of {time, value}
_equity_history: list[dict] = []
_last_equity_ts: float = 0


def attach_bot(bot: "NexusBot"):
    global _bot
    _bot = bot


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text()


@app.get("/api/status")
async def status():
    if not _bot:
        return {"error": "bot not running"}
    summary = _bot._risk.summary()
    pv = await _bot._exchange.portfolio_value()
    return {
        "portfolio_value": round(pv, 2),
        "capital":         summary["capital"],
        "total_pnl":       summary["total_pnl"],
        "trade_count":     summary["trade_count"],
        "win_rate":        summary["win_rate"],
        "open_count":      summary["open_count"],
        "drawdown_pct":    summary["drawdown_pct"],
        "cycle":           _bot._cycle,
        "running":         _bot._running,
    }


@app.get("/api/trades")
async def trades():
    if not _bot:
        return []
    return _bot._exchange.trade_history()


@app.get("/api/positions")
async def positions():
    if not _bot:
        return {}
    return {
        sym: {
            "side":           pos.side,
            "quantity":       pos.quantity,
            "entry_price":    pos.entry_price,
            "current_price":  pos.current_price,
            "unrealized_pnl": pos.unrealized_pnl,
        }
        for sym, pos in _bot._risk.open_positions.items()
    }


@app.get("/api/candles/{symbol}")
async def candles(symbol: str, interval: str = "5m", limit: int = 100):
    if not _bot:
        return []
    data = await _bot._exchange.get_candles(symbol.upper(), interval, limit)
    return data


@app.get("/api/equity")
async def equity():
    return _equity_history[-200:]


@app.post("/api/control")
async def control(body: dict):
    if not _bot:
        return {"error": "bot not running"}
    action = body.get("action", "").lower()
    symbol = body.get("symbol", "BTCUSDT").upper()

    if action == "halt":
        _bot.stop()
        return {"ok": True, "action": "halt"}

    if action in ("buy", "sell"):
        from alpha.engine import Signal
        sig = Signal(
            symbol=symbol, action=action.upper(),
            score=1.0, kelly_f=0.05, breakdown={}, reason="manual_override",
        )
        asyncio.create_task(_bot._router.process(sig))
        return {"ok": True, "action": action, "symbol": symbol}

    return {"error": f"unknown action: {action}"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
            if not _bot:
                continue

            summary = _bot._risk.summary()
            pv = await _bot._exchange.portfolio_value()

            # Append equity point every 5s
            global _last_equity_ts
            now = int(time.time())
            if now - _last_equity_ts >= 5:
                _equity_history.append({"time": now, "value": round(pv, 2)})
                _last_equity_ts = now

            # Live prices for all symbols
            prices = {}
            for sym in ["BTCUSDT", "ETHUSDT"]:
                try:
                    prices[sym] = await _bot._exchange.get_ticker(sym)
                except Exception:
                    pass

            # Last strategy votes from alpha engine
            votes = {}
            try:
                for sym in ["BTCUSDT", "ETHUSDT"]:
                    last = getattr(_bot._alpha, "_last_votes", {}).get(sym, {})
                    if last:
                        votes = last
                        break
            except Exception:
                pass

            await ws.send_text(json.dumps({
                "portfolio_value": round(pv, 2),
                "total_pnl":       summary["total_pnl"],
                "win_rate":        summary["win_rate"],
                "drawdown_pct":    summary["drawdown_pct"],
                "trade_count":     summary["trade_count"],
                "open_count":      summary["open_count"],
                "cycle":           _bot._cycle,
                "prices":          prices,
                "equity_point":    {"time": now, "value": round(pv, 2)},
                "votes":           votes,
            }))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)

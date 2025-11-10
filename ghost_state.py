import json
import os
import tempfile
from typing import Any

STATE_PATH = os.getenv("GHOST_STATE_PATH", "ghost_state.json")


def _default_trading_state() -> dict[str, Any]:
    return {
        "cash": {"stock": 1000.0, "crypto": 1000.0},
        "cash_balance": 2000.0,
        "positions": [],
        "goals": [],
        "goal_locked": False,
        "security": {
            "max_trade_usd": 1000.0,
            "max_daily_loss_usd": 0.0,
            "slippage_bps": 0.0,
            "stop_loss_pct": 0.0,
            "trailing_stop_pct": 0.0,
            "whale_min_usd": 0.0,
        },
        "ai_settings": {
            "model": "gpt-mini",
            "advisor_strategies": {
                "momentum": True,
                "mean_reversion": True,
                "onchain_flows": False,
            },
            "feature_weights": {
                "technical": 0.4,
                "fundamental": 0.3,
                "sentiment": 0.2,
                "macro": 0.1,
            },
            "backtest_window_days": 180,
        },
        "last_advisor_refresh": None,
        "portfolio_value": 2000.0,
        "ghost_score": 0,
        "ledger": [],
    }


def default_state() -> dict[str, Any]:
    return {
        "vip_coins": ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC"],
        "watchlist": {
            "stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
            "crypto": ["BTC", "ETH", "XRP", "SOL", "ADA"],
        },
        "wallets": [
            {"name": "MetaMask", "addresses": []},
            {"name": "Coinbase", "addresses": []},
            {"name": "Trust", "addresses": []},
        ],
        "goals": [],
        "trading_state": _default_trading_state(),
        "universe": {
            "stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
            "crypto": ["BTC", "ETH", "XRP", "SOL", "ADA"],
        },
    }


_STATE: dict[str, Any] | None = None


def _atomic_write(path: str, data: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False, encoding="utf-8") as tf:
        tmp = tf.name
        tf.write(data)
    os.replace(tmp, path)


def load() -> dict[str, Any]:
    global _STATE
    if _STATE is not None:
        return _STATE
    try:
        if os.path.isfile(STATE_PATH):
            with open(STATE_PATH, encoding="utf-8") as f:
                _STATE = json.load(f)
        else:
            _STATE = default_state()
            save()
    except Exception:
        _STATE = default_state()
        save()
    return _STATE


def save() -> None:
    global _STATE
    if _STATE is None:
        _STATE = default_state()
    _atomic_write(STATE_PATH, json.dumps(_STATE, ensure_ascii=False, indent=2))


def get_state() -> dict[str, Any]:
    return load()


def set_state(new_state: dict[str, Any]) -> dict[str, Any]:
    global _STATE
    _STATE = dict(new_state)
    save()
    return _STATE


def update_runtime(trading_state: dict[str, Any], universe: dict[str, list[str]]):
    st = load()
    st["trading_state"] = trading_state
    st["universe"] = {"stocks": universe.get("stocks", []), "crypto": universe.get("crypto", [])}
    # Keep watchlist aligned with universe for simplicity
    st["watchlist"] = {
        "stocks": universe.get("stocks", []),
        "crypto": universe.get("crypto", []),
    }
    save()


def reset() -> dict[str, Any]:
    return set_state(default_state())

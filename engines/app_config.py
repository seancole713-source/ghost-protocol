"""App configuration and global state for Ghost Protocol Wolf.

Extracted from wolf_app.py (Step 12 cleanup). Contains:
- Optional feature-flag imports (Stage 1-5, Watchlist, etc.)
- Logging setup
- All module-level constants, caches, and worker threads
- atexit registrations

Imported by wolf_app.py via `from engines.app_config import *`
so that routes using `import wolf_app as _wa; _wa.SOME_GLOBAL` continue to work.
"""
import asyncio
import atexit
import json
import logging
import math
import os
import queue as _queue
import random
import sqlite3
import sys
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:
    Retry = None  # type: ignore

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore

from core.concurrency import AsyncRateLimiter

from wolf_helpers import (
    _configure_logging,
    _ensure_ai_storage,
    _init_security_tables,
    _set_mode_gauge,
    _set_hold_gauge,
    _stop_autosave_worker,
    _stop_alert_worker,
    _persist_save,
    _stop_schedule_worker,
    _classify_symbol_category,
    _tg_send_chat_message,
)

# ── IP Allowlisting (defined here so middleware.py can access via _ac) ────
_allowlist_str = os.getenv("IP_ALLOWLIST", "").strip()
IP_ALLOWLIST: set = (
    set(ip.strip() for ip in _allowlist_str.split(",") if ip.strip())
    if _allowlist_str else set()
)
IP_ALLOWLIST_ENABLED: bool = len(IP_ALLOWLIST) > 0





# Ghost Hunter Phase 1 imports
try:
    from core.feature_diagnostics import diagnose_features, build_confidence_with_diagnostics
    GHOST_HUNTER_ENABLED = True
except Exception as e:
    GHOST_HUNTER_ENABLED = False
    print(f"Ghost Hunter Phase 1 disabled: {e}")

try:
    from core.research_blueprint import build_research_snapshot  # type: ignore

    RESEARCH_BLUEPRINT_ON = True
except Exception:
    RESEARCH_BLUEPRINT_ON = False

# Import portfolio persistence layer
try:
    from core.portfolio_persistence import get_portfolio_store

    PORTFOLIO_PERSISTENCE_ENABLED = True
except Exception:
    PORTFOLIO_PERSISTENCE_ENABLED = False

# Optional ChatGPT price provider
try:
    from chatgpt_price_provider import ChatGPTStockPriceProvider  # type: ignore

    CHATGPT_PROVIDER_IMPORT = True
except Exception:
    ChatGPTStockPriceProvider = None
    CHATGPT_PROVIDER_IMPORT = False
    # print(f"[GHOST INIT] ChatGPT provider import failed: {e}")

# Stage 1: Context Awareness imports
try:
    from core.stage1_integration import (
        get_enhanced_context,
        get_symbol_context,
        initialize_stage1,
    )

    STAGE1_ENABLED = True
except Exception as e:
    STAGE1_ENABLED = False
    print(f"Stage 1 Context Awareness disabled: {e}")

# Stage 2: Self-Evaluation System imports
try:
    from core.accuracy_tracker import get_accuracy_report, get_accuracy_tracker
    from core.learning_loop import (
        get_learning_loop,
        get_learning_stats,
        run_learning_cycle,
    )

    STAGE2_ENABLED = True
except Exception as e:
    STAGE2_ENABLED = False
    print(f"Stage 2 Self-Evaluation System disabled: {e}")

# Scheduled Predictions System imports
try:
    import core.scheduled_predictions as scheduled_predictions

    SCHEDULED_PREDICTIONS_ENABLED = True
except Exception as e:
    SCHEDULED_PREDICTIONS_ENABLED = False
    print(f"Scheduled Predictions System disabled: {e}")

# Stage 3: Continuous Improvement System imports
try:
    from core.ensemble_forecaster import get_ensemble_forecaster
    from core.regime_detector import get_regime_detector
    from core.risk_engine import get_risk_engine

    STAGE3_ENABLED = True
except Exception as e:
    STAGE3_ENABLED = False
    print(f"Stage 3 Continuous Improvement System disabled: {e}")

# Stage 4: Portfolio Optimization & Advanced Strategies imports
try:
    from core.backtester import get_backtester
    from core.hedging_engine import get_hedging_engine
    from core.portfolio_manager import get_portfolio_manager
    from core.strategy_tester import get_strategy_tester

    STAGE4_ENABLED = True
except Exception as e:
    STAGE4_ENABLED = False
    print(f"Stage 4 Portfolio Optimization disabled: {e}")

# Stage 5: Advanced Execution & Order Management imports
try:
    from core.execution_analytics import get_execution_analytics
    from core.execution_risk import get_execution_risk
    from core.order_manager import OrderSide, OrderType, TimeInForce, get_order_manager
    from core.smart_router import get_smart_router

    STAGE5_ENABLED = True
except Exception as e:
    STAGE5_ENABLED = False
    print(f"Stage 5 Advanced Execution disabled: {e}")

# Watchlist Manager import
try:
    from core.watchlist_manager import get_watchlist_manager

    WATCHLIST_ENABLED = True
except Exception as e:
    WATCHLIST_ENABLED = False
    print(f"Watchlist Manager disabled: {e}")
_configure_logging()
LOGGER = logging.getLogger("ghost")

# Suppress noisy yfinance "symbol may be delisted" warnings
# These fire even when Polygon fallback succeeds, creating noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ============================================================================
# PREDICTION GATES (Step 3, Mar 13 2026)
# Kill switch + confidence floor + rate limit — applied before every INSERT
# into ghost_predictions. Only blocks NEW predictions; never touches history.
# ============================================================================
_PREDICTION_GATE_KILL_SWITCH_MIN_TRADES = 10
_PREDICTION_GATE_KILL_SWITCH_MIN_WINRATE = 35.0   # percent
_PREDICTION_GATE_CONFIDENCE_FLOOR = 0.45           # was 0.55 — too high, base_confidence starts at 0.50-0.52
_PREDICTION_GATE_MAX_PER_DAY = 2




# OpenTelemetry (optional)
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "0").lower() in ("1", "true", "yes")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "ghost-wolf")
_OTEL_TRACER = None
if OTEL_ENABLED:
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import (  # type: ignore
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: OTEL_SERVICE_NAME}))
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _OTEL_TRACER = trace.get_tracer(OTEL_SERVICE_NAME)
    except Exception as e:
        LOGGER.warning(f"Failed to initialize OpenTelemetry tracer: {e}", exc_info=True)
        _OTEL_TRACER = None

# Serve prebuilt UI if present (ui_dist)
_RETRAIN_STATUS = {"running": False, "last_result": None, "started_at": None}



# Global variable to track paper trade re-evaluation status
_REEVALUATION_STATUS = {"running": False, "last_result": None, "started_at": None}











# Minimal cockpit route to serve the existing HTML template without a template engine.
# This helps evidence collectors and manual checks access the cockpit when UI bundles
# are not mounted or when running in minimal deployment environments.










# Load secrets from secrets.env if API keys not already in environment
_secrets_file = os.path.join(os.path.dirname(__file__), "secrets.env")
if os.path.exists(_secrets_file) and (
    not os.getenv("POLYGON_API_KEY") or not os.getenv("ALPHAVANTAGE_API_KEY")
):
    try:
        with open(_secrets_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _value = _line.split("=", 1)
                    _value = _value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if _key in (
                        "POLYGON_API_KEY",
                        "ALPHAVANTAGE_API_KEY",
                        "ALPHA_VANTAGE_API_KEY",
                        "GHOST_API_TOKEN",
                    ) and not os.getenv(_key):
                        os.environ[_key] = _value
    except Exception:
        pass  # Continue if secrets.env unavailable

# Env/config
WOLF = "WOLF"
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY", "")
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")

# Multi-symbol prediction lists
# UNLIMITED WATCHLIST: Ghost can track thousands of symbols simultaneously
# Default includes 100+ stocks + 50+ crypto for comprehensive market coverage
# For custom watchlists, set STOCK_SYMBOLS / CRYPTO_SYMBOLS environment variables
# For on-demand predictions of ANY symbol, use /api/predictions/run?symbol=SYMBOL

DEFAULT_STOCK_SYMBOLS = [
    # Mega Cap Tech (FAANG+)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA",
    # Major Tech
    "ORCL", "CRM", "ADBE", "NFLX", "INTC", "AMD", "CSCO", "IBM", "QCOM", "TXN", "AVGO",
    "MU", "MRVL", "KLAC", "LRCX", "AMAT", "ASML", "TSM", "SNPS", "CDNS", "ARM",
    # AI & Cloud Leaders
    "PLTR", "SNOW", "DDOG", "MDB", "NET", "ZS", "CRWD", "PANW", "FTNT", "OKTA",
    "TEAM", "WDAY", "NOW", "SPLK", "VEEV", "HUBS", "TTD", "BILL", "DOCN", "PATH",
    # Semiconductors Extended
    "ON", "SWKS", "QRVO", "NXPI", "MCHP", "ADI", "MPWR", "WOLF", "SIMO", "SMCI",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "USB", "PNC", "TFC", "COF", "AXP",
    "V", "MA", "PYPL", "SQ", "COIN", "HOOD", "AFRM", "UPST", "SOFI",
    # Healthcare & Biotech
    "UNH", "JNJ", "PFE", "ABBV", "TMO", "ABT", "MRK", "LLY", "AMGN", "GILD", "BMY", "CVS",
    "MRNA", "BNTX", "REGN", "VRTX", "BIIB", "ISRG", "DXCM", "ZTS", "GEHC",
    # Consumer Discretionary
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "DIS", "BKNG", "ABNB", "EBAY", "ETSY",
    "LULU", "ROST", "TJX", "DHI", "LEN", "PHM", "ORLY", "AZO", "TSCO",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "PM", "MDLZ", "CL", "KHC", "GIS", "KMB",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY", "HAL", "DVN",
    # EV & Clean Energy
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "CHPT", "BLNK", "PLUG", "FCEL", "ENVX",
    "ENPH", "SEDG", "FSLR", "RUN", "NOVA", "ARRY", "STEM",
    # Industrials
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "RTX", "MMM", "DE", "UNP", "FDX", "DAL", "UAL",
    # Materials
    "LIN", "APD", "FCX", "NEM", "CTVA", "DD", "DOW", "PPG", "NUE", "CLF", "X", "AA",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "DLR", "O", "VICI",
    # Communication Services
    "CMCSA", "VZ", "T", "TMUS", "CHTR", "PARA", "WBD", "FOX", "NWSA",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG",
    # Market Indices & ETFs
    "SPY", "QQQ", "DIA", "IWM", "ARKK", "ARKG", "SOXL", "TQQQ", "SQQQ", "VXX",
    # High Momentum/Meme/Volatility
    "GME", "AMC", "BBBY", "KOSS", "EXPR", "BB", "NOK", "CLOV", "WISH", "SPCE",
    "SNAP", "PINS", "UBER", "LYFT", "DASH", "RBLX", "U", "ROKU", "SPOT", "SHOP",
    # Edge Symbols — proven 81.4% WR (added Feb 9, 2026)
    "BMBL", "XPO", "ITRI",
    # Space & Defense
    "RKLB", "ASTR", "ASTS", "RDW", "LUNR", "MNTS",
    # Biotech Momentum
    "ARWR", "SRPT", "EXAS", "RARE", "BMRN", "ALNY", "NTLA", "CRSP", "EDIT", "BEAM",
    # Recent IPOs & SPACs
    "IONQ", "QBTS", "RGTI", "QUBT",  # Quantum computing
    "GRAB", "SE", "MELI", "NU",  # EM Fintech
    # Storage & Data (today's movers category)
    "WDC", "STX", "NTAP", "PSTG",
]

DEFAULT_CRYPTO_SYMBOLS = [
    # === MAJOR LAYER 1s (Top 50 by Market Cap) ===
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
    "SHIB", "LTC", "TRX", "ATOM", "ETC", "XLM", "BCH", "ALGO",
    "VET", "ICP", "HBAR", "FIL", "APT", "SUI", "SEI", "TIA", "NEAR",
    "FTM", "KAVA", "ROSE", "CELO", "ZIL", "EOS", "NEO", "IOTA",
    "EGLD", "XTZ", "STX", "CORE", "CFX", "ASTR", "CANTO",
    # Removed: XDC, KAS, MINA, XMR, TON, INJ, OSMO, STRK, FUN, MYST, TNT
    # (all crypto providers fail - Binance 451/geo-blocked, CoinGecko 429/unsupported, Coinbase 404)
    
    # === LAYER 2s & SCALING ===
    "ARB", "OP", "MANTA", "METIS", "IMX", "BOBA", "ZK", "SCROLL",
    "LINEA", "BASE", "ZKSYNC", "POLYGON", "BLAST", "MODE",
    
    # === SOLANA ECOSYSTEM (Hot in 2024-2026) ===
    "RAY", "JTO", "JUP", "PYTH", "ORCA", "MNGO", "SRM", "STEP", "COPE", "FIDA",
    "TULIP", "SLND", "MARINADE", "MSOL", "JITO",
    
    # === DEFI PROTOCOLS (Blue Chips) ===
    "UNI", "LINK", "AAVE", "MKR", "SNX", "COMP", "CRV", "SUSHI", "YFI", "LDO",
    "RPL", "BAL", "1INCH", "DYDX", "GMX", "GNS", "PENDLE", "CVX", "FXS", "FRAX",
    "RAI", "LQTY", "SPELL", "ALCX", "RUNE", "THOR",
    
    # === AI & DATA TOKENS (Hottest Narrative) ===
    "FET", "AGIX", "OCEAN", "RNDR", "TAO", "ARKM", "AIOZ", "OLAS", "NMR", "RLC",
    "PRIME", "AKT", "CLORE", "NOSANA", "GRASS", "IO", "ATH", "VIRTUAL", "AI16Z",
    
    # === MEME COINS (High Volatility = High Opportunity) ===
    "PEPE", "WIF", "BONK", "FLOKI", "BABYDOGE", "ELON", "SAMO", "LADYS", "TURBO", "MEME",
    "POPCAT", "MYRO", "SLERF", "BOME", "WEN", "BRETT", "DEGEN", "TOSHI", "MOCHI",
    "NEIRO", "GOAT", "PNUT", "ACT", "CHILLGUY", "MOODENG", "SPX", "GIGA", "MOG",
    "SNEK", "ANDY", "TROLL", "WOLF", "HPOS10I", "BITCOIN", "ANALOS",
    
    # === NFT & GAMING ===
    "SAND", "MANA", "AXS", "GALA", "ENJ", "CHZ", "FLOW", "APE", "ILV", "MAGIC",
    "PIXEL", "PRIME", "IMX", "GODS", "SUPER", "YGG", "MC", "ALICE", "HERO",
    "PYR", "VOXEL", "HIGH", "GHST", "REVV", "TOWER", "NAKA",
    
    # === PRIVACY COINS ===
    "ZEC", "SCRT", "BEAM", "XVG", "ARRR", "FIRO", "ZEN", "PIVX", "GRIN",
    
    # === STORAGE & INFRASTRUCTURE ===
    "AR", "STORJ", "GRT", "ANKR", "POKT", "LPT", "AIOZ", "THETA", "TFUEL", "HNT",
    "MOBILE", "IOT", "AKT", "GLM", "SC",
    
    # === ORACLE & DATA ===
    "LINK", "BAND", "TRB", "DIA", "API3", "UMA", "PYTH",
    
    # === EXCHANGE TOKENS ===
    "CRO", "OKB", "HT", "LEO", "KCS", "GT", "MX", "BGB", "WOO", "DYDX",
    
    # === DEPIN (Decentralized Physical Infrastructure) ===
    "HNT", "MOBILE", "IOT", "IOTX", "DIMO", "WIFI", "HONEY", "ARKM",
    
    # === REAL WORLD ASSETS (RWA) ===
    "ONDO", "MPL", "CFG", "POLS", "CPOOL", "GFI", "MAPLE", "TRU", "PROPC",
    
    # === LIQUID STAKING ===
    "LDO", "RPL", "FXS", "ANKR", "SD", "SWISE", "OETH", "SWETH",
    "STETH", "RETH", "CBETH", "MSOL", "BSOL", "JITOSOL",
    
    # === PERPS & DERIVATIVES ===
    "DYDX", "GMX", "GNS", "KWENTA", "LYRA", "PREMIA", "DOPEX", "HEGIC", "OPYN",
    "SNX", "PERP", "MCB", "VELA", "HMX", "VERTEX",
    
    # === BASE ECOSYSTEM (Coinbase L2) ===
    "BRETT", "DEGEN", "TOSHI", "MOCHI", "WELL", "AERO", "EXTRA",
    
    # === COSMOS ECOSYSTEM ===
    "ATOM", "TIA", "SEI", "KAVA", "SCRT", "JUNO", "EVMOS", "STARS",
    "KUJI", "NTRN", "DYM", "SAGA",
    
    # === POLKADOT ECOSYSTEM ===
    "DOT", "KSM", "ASTR", "GLMR", "ACA", "PARA", "CLV", "NODL", "AZERO",
    
    # === STABLECOINS & WRAPPED (for reference/pairs) ===
    "WBTC", "STETH", "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX",
    "LUSD", "SUSD", "MIM", "USDD", "FDUSD",
    
    # === LEGACY TOP 100-200 ===
    "THETA", "ONE", "HIVE", "ICX", "QTUM", "WAVES",
    "KDA", "FLUX", "ERG", "RVN", "DGB", "SYS", "NAV",
    "OMG", "SNT", "ANT", "MLN", "REP", "LOOM", "BAT",
    "ZRX", "DENT", "CVC", "GNO", "DNT",
    "QNT", "MASK", "ENS", "SSV", "BLUR", "ID", "HOOK", "EDU",
]

# Load from environment or use defaults
STOCK_SYMBOLS = os.getenv("STOCK_SYMBOLS", ",".join(DEFAULT_STOCK_SYMBOLS)).split(",")
CRYPTO_SYMBOLS = os.getenv("CRYPTO_SYMBOLS", ",".join(DEFAULT_CRYPTO_SYMBOLS)).split(",")

# VIP COINS — Ghost Protocol Special Tracking (Presale/Meme Coins)
# These are user's priority coins for strike prep and presale awareness
VIP_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP"]  # Reverted: presale coins unavailable on exchanges

# Multi-symbol prediction health tracking
_LAST_MULTI_PREDICTION_TIME: float | None = None
_LAST_MULTI_PREDICTION_COUNTS: dict[str, int] = {"stocks": 0, "crypto": 0, "vip": 0}
_LAST_MULTI_PREDICTION_RESULT: dict[str, Any] | None = None  # Cache full result
_MULTI_PREDICTION_CACHE_TTL = 30  # Reduced cache TTL for fresher predictions at scale
_LAST_TELEGRAM_SEND_TIME: float | None = None
_LAST_TELEGRAM_STATUS: str = "never_run"
_LAST_TELEGRAM_ERROR: str | None = None

# In-memory predictions store (wires /api/predict/run → /api/cockpit)
# Maps symbol → {prediction_id, run_at, confidence, direction, horizon_h, symbol}
# Structure: flat dict where symbol is the key
# Use _classify_symbol_category() to determine if stocks/crypto/vip
_LATEST_PREDICTIONS: dict[str, dict[str, Any]] = {}
# BUG FIX (Jan 6, 2026): Add thread lock for race condition protection
import threading
_LATEST_PREDICTIONS_LOCK = threading.Lock()

# ============================================================
# NOTIFICATION LOOP STATUS TRACKING (for /debug/notification-loop-status)
# ============================================================
_NOTIFICATION_LOOP_STATUS = {
    "running": False,
    "started_at": None,
    "loop_count": 0,
    "last_top10_date": None,
    "last_top10_send_time": None,
    "last_top10_success": None,
    "last_check_time": None,
    "current_central_time": None,
    "predictions_count": 0,
}

# Ghost Hunter V2: UNLIMITED symbol tracking across all markets
# Auto-expands to track ANY liquid symbol with available price feeds
# NO ARTIFICIAL LIMITS - scales to thousands of symbols
HUNTER_STOCK_SYMBOLS = DEFAULT_STOCK_SYMBOLS  # Use full expanded list

# Crypto: All liquid coins on major exchanges with reliable price feeds
# Includes DeFi, Layer 1/2, NFT, Meme coins, and emerging tokens
HUNTER_CRYPTO_SYMBOLS = DEFAULT_CRYPTO_SYMBOLS  # Use full expanded list

# Keep crypto-vs-stock routing consistent across the repo (providers/pillars/etc.).
# We register the runtime crypto universe here to avoid circular imports.
try:
    from core.asset_classification import register_crypto_symbols as _register_crypto_symbols

    _register_crypto_symbols(CRYPTO_SYMBOLS)
    _register_crypto_symbols(HUNTER_CRYPTO_SYMBOLS)
except Exception as e:
    LOGGER.warning(f"crypto_symbol_registration_failed: {e}")


# ChatGPT Price Provider (for watchlist stocks)
try:
    CHATGPT_PRICE_PROVIDER = ChatGPTStockPriceProvider()
    print("[GHOST INIT] ChatGPT Price Provider: ENABLED")
except Exception as e:
    CHATGPT_PRICE_PROVIDER = None
    print(f"[GHOST INIT] ChatGPT Price Provider: DISABLED ({e})")

# Debug: Log API key status at module load time
import sys

print(
    f"[GHOST INIT] ALPHAVANTAGE_KEY: {f'SET (len={len(ALPHAVANTAGE_KEY)})' if ALPHAVANTAGE_KEY else 'MISSING'}",
    file=sys.stderr,
)
print(
    f"[GHOST INIT] POLYGON_KEY: {f'SET (len={len(POLYGON_KEY)})' if POLYGON_KEY else 'MISSING'}",
    file=sys.stderr,
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TICK_INTERVAL_S = int(os.getenv("TICK_INTERVAL_S", "5"))
PRICE_TTL_S = int(os.getenv("PRICE_TTL_S", "60"))  # Increased from 30s for better caching
# Increased TTL during market hours to avoid rate limits (was 5s, now 60s)
# This prevents hammering APIs and getting 429 errors
PRICE_TTL_OPEN_S = int(os.getenv("PRICE_TTL_OPEN_S", "60"))
NEWS_TTL_S = int(os.getenv("NEWS_TTL_S", "300"))
REUTERS_FEEDS_ON = int(os.getenv("REUTERS_FEEDS_ON", "0"))
REUTERS_FEEDS = os.getenv(
    "REUTERS_FEEDS",
    "https://feeds.reuters.com/reuters/businessNews,https://feeds.reuters.com/reuters/technologyNews",
)
# Optional Reuters filtering and manual feeds
REUTERS_SYMBOLS = [
    s.strip().upper() for s in os.getenv("REUTERS_SYMBOLS", "").split(",") if s.strip()
]
REUTERS_KEYWORDS = [
    s.strip().lower() for s in os.getenv("REUTERS_KEYWORDS", "").split(",") if s.strip()
]
NEWS_MANUAL_FEEDS = [u.strip() for u in os.getenv("NEWS_MANUAL_FEEDS", "").split(",") if u.strip()]
NEWS_WHITELIST = [
    h.strip().lower() for h in os.getenv("NEWS_WHITELIST", "").split(",") if h.strip()
]
NEWS_MAX_AGE_MIN = int(os.getenv("NEWS_MAX_AGE_MIN", "0") or "0")

# Price anomaly guardrails
PRICE_ANOMALY_X = float(os.getenv("PRICE_ANOMALY_X", "5"))
PRICE_ANOMALY_NEWS_WINDOW_MIN = int(os.getenv("PRICE_ANOMALY_NEWS_WINDOW_MIN", "60"))
# Pause forecast when anomaly detected (manual override is always paused)
FORECAST_PAUSE_ON_ANOMALY = int(os.getenv("FORECAST_PAUSE_ON_ANOMALY", "1"))
# Focus mode: restrict UI and actions to WOLF-only by default
FOCUS_WOLF_ONLY = os.getenv("FOCUS_WOLF_ONLY", "0").lower() in ("1", "true", "yes")
HTTP_POOL_ENABLED = os.getenv("HTTP_POOL_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)
HTTP_POOL_SIZE = int(os.getenv("HTTP_POOL_SIZE", "20"))  # Increased from 10 to 20 for yfinance concurrency
HTTP_POOL_RETRIES = int(os.getenv("HTTP_POOL_RETRIES", "2"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "8"))

# Coinbase Pro configuration (used in data_collector.py for RSI/trend)
COINBASE_PRO_ENABLED = os.getenv("COINBASE_PRO_ENABLED", "1").lower() in ("1", "true", "yes")
COINBASE_PRO_TIMEOUT_S = float(os.getenv("COINBASE_PRO_TIMEOUT_S", "5.0"))
COINBASE_PRO_BASE_URL = os.getenv("COINBASE_PRO_BASE_URL", "https://api.exchange.coinbase.com")

# Cache TTL settings for high-traffic endpoints
HUNTER_FEED_CACHE_TTL = int(os.getenv("HUNTER_FEED_CACHE_TTL", "30"))  # Default: 30s
WATCHLIST_CACHE_TTL = int(os.getenv("WATCHLIST_CACHE_TTL", "120"))  # Default: 120s (increased for performance)
VIP_SNAPSHOT_CACHE_TTL = int(os.getenv("VIP_SNAPSHOT_CACHE_TTL", "30"))  # Default: 30s
MACRO_BRAIN_ON = os.getenv("MACRO_BRAIN_ON", "0").lower() in ("1", "true", "yes")
MACRO_TICKERS = os.getenv("MACRO_TICKERS", "SMH,SOXX,QQQ").split(",")
MACRO_LOOKBACK_DAYS = int(os.getenv("MACRO_LOOKBACK_DAYS", "20"))

# Optional provider-order tweak: try Yahoo HTTP first during constrained environments
PRICE_YAHOO_FIRST = os.getenv("PRICE_YAHOO_FIRST", "0").lower() in ("1", "true", "yes")
# Looser max deviation during market hours (defaults to same as PRICE_MAX_DEVIATION if unset)
PRICE_MAX_DEVIATION_OPEN = float(
    os.getenv("PRICE_MAX_DEVIATION_OPEN", os.getenv("PRICE_MAX_DEVIATION", "0.5"))
)
# If we only have prev_close cached, respect TTL and avoid provider calls
PRICE_PREV_ONLY_RESPECT_TTL = os.getenv("PRICE_PREV_ONLY_RESPECT_TTL", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Reorder to prioritize AlphaVantage (most reliable) over Yahoo/yfinance (rate limited)
_DEFAULT_PROVIDER_ORDER = ("alphavantage", "polygon", "yfinance", "yahoo")
_stock_source_env = os.getenv("STOCK_PRICE_SOURCE", ",".join(_DEFAULT_PROVIDER_ORDER))
STOCK_PRICE_SOURCE = [
    token for token in (piece.strip().lower() for piece in _stock_source_env.split(",")) if token
]
if not STOCK_PRICE_SOURCE:
    STOCK_PRICE_SOURCE = list(_DEFAULT_PROVIDER_ORDER)

PRICE_STRICT_LIVE = os.getenv("PRICE_STRICT_LIVE", "0").lower() in ("1", "true", "yes")
try:
    DATA_FRESHNESS_SEC = int(os.getenv("DATA_FRESHNESS_SEC", str(PRICE_TTL_S)))
except Exception:
    DATA_FRESHNESS_SEC = PRICE_TTL_S
PRICE_PROVIDER_TIMEOUT_S = float(os.getenv("PRICE_PROVIDER_TIMEOUT", "6"))

# Per-symbol provider blacklist (exclude misbehaving sources from consensus)
# Acceptance: never surface polygon as provider for WOLF if it disagrees
# TEMPORARY FIX: Allow polygon since AlphaVantage rate limited and Yahoo blocked
PROVIDER_BLOCKLIST: dict[str, set[str]] = {
    "WOLF": set(),  # Removed {"polygon"} - it's the only working provider after rate limits
}

# Add near other globals (after PROVIDER_BLOCKLIST)
try:
    PROVIDER_BACKOFF  # type: ignore
except NameError:
    PROVIDER_BACKOFF: dict[
        str, dict[str, float]
    ] = {}  # {provider: {"until": epoch, "failures": n}}

# Delisted/restructured symbols registry (corporate actions)
# Tracks bankruptcy, reverse splits, spinoffs, etc.
DELISTED_SYMBOLS: dict[str, dict[str, Any]] = {
    "WOLF": {
        "status": "restructured",  # restructured|delisted|suspended
        "date": "2025-10-01",
        "reverse_split_ratio": 120,  # 120:1 reverse split on bankruptcy exit
        "note": "Emerged from Chapter 11 bankruptcy Oct 2025",
        "untradable": False,  # Can still trade post-restructuring
        "banner": "⚠️ WOLF underwent 120:1 reverse split in bankruptcy exit (Oct 2025)",
        "shareholders_diluted": True,  # Original shareholders received 1:120 ratio
    }
}


# --- Corporate Actions API ---------------------------------------------------




# Last price diagnostic to inform snapshot flags/banners
PRICE_DIAG: dict[str, Any] = {
    "anomaly": False,
    "reason": "",
    "quorum_ok": True,
    "provider_spread": None,  # relative spread across providers
    "providers": [],  # [(name, price)]
    "last_fetch_provider": None,  # provider used for last successful fetch
    "last_fetch_latency_ms": None,  # latency of last fetch
    "last_good_price_ts": None,  # timestamp of last successful price fetch
    "fallback_reason": None,  # reason for fallback if applicable
}

# ── Lightweight prediction/feedback state ─────────────────────────────────────
from collections import deque as _deque

PRED_FEEDBACK: _deque[dict[str, Any]] = _deque(maxlen=200)
PRED_CALLS_TOTAL = 0
PRED_LAST_TS = 0.0

# Tunables for simple cone forecast
PRED_SIGMA_DAILY = float(os.getenv("PRED_SIGMA_DAILY", "0.06"))  # ~6% daily vol default
PRED_Z = float(os.getenv("PRED_Z", "1.0"))  # 1-sigma band
PRED_STEP_H = int(os.getenv("PRED_STEP_H", "2"))  # 2h resolution

# Research integration (news + filings) — enabled by default
PRED_USE_NEWS = os.getenv("PRED_USE_NEWS", "1") not in ("0", "false", "False", "no")
PRED_USE_FILINGS = os.getenv("PRED_USE_FILINGS", "1") not in ("0", "false", "False", "no")
FILINGS_TTL_S = int(os.getenv("FILINGS_TTL_S", "600"))  # cache SEC filings signal 10 minutes

# Simple in-memory cache for filings signal
FILINGS_CACHE: dict[str, dict[str, Any]] = {"ts": 0.0, "data": None}
# Runtime tunable forecast params (added for two-line overlay)
FORECAST_STEP_S = int(os.getenv("FORECAST_STEP_S", str(2 * 3600)))  # 2h default = 7200s
FORECAST_HORIZON_S = int(os.getenv("FORECAST_HORIZON_S", str(48 * 3600)))  # 48h default = 172800s
FORECAST_GRID_PATH = "data/forecast_WOLF.json"
FORECAST_MAX_AGE_S = 24 * 3600  # Regenerate if >24h old
















# ============================================================================
# TWO-LINE OVERLAY SYSTEM: Ghost vs Live Forecast
# ============================================================================










# ============================================================================
# End Two-Line Overlay System
# ============================================================================




from core.ai_memory import AIMemory, get_memory

# ── AI Brain: persistent memory + preview/training stubs ────────────────────
AI_MEMORY_READ_AUTH = int(os.getenv("AI_MEMORY_READ_AUTH", "0"))
_AI_MEMORY_AUTH_REQUIRED = bool(AI_MEMORY_READ_AUTH)

AI_DATA_DIR = os.getenv("AI_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
AI_LEGACY_DB_PATH = os.getenv("AI_DB_PATH", os.path.join(AI_DATA_DIR, "ghost_ai.db"))
AI_MEMORY_DB_PATH = os.getenv("AI_MEMORY_DB_PATH", os.path.join(AI_DATA_DIR, "ai_memory.db"))
# Default to "none" to avoid chromadb/faiss dependency issues
AI_MEMORY_VECTOR_STORE = os.getenv("AI_MEMORY_VECTOR_STORE", os.getenv("AI_VECTOR_STORE", "none"))




_ensure_ai_storage()

# Initialize persistent AI memory (with graceful fallback)
AI_MEMORY_STORE: AIMemory | None = None
try:
    # Use "none" for vector store initially to avoid chromadb/faiss deps
    vector_store = AI_MEMORY_VECTOR_STORE.lower()
    if vector_store not in ["chromadb", "faiss", "none"]:
        vector_store = "none"
    AI_MEMORY_STORE = get_memory(AI_MEMORY_DB_PATH, vector_store)
    LOGGER.info("ai_memory_initialized", extra={"db": AI_MEMORY_DB_PATH, "vector": vector_store})
except Exception as _ai_err:  # pragma: no cover - defensive guard
    LOGGER.exception("ai_memory_init_failed", extra={"error": str(_ai_err)})
    AI_MEMORY_STORE = None

AI_MEMORY_RING: deque[dict[str, Any]] = deque(maxlen=1000)






















DEFAULT_QTY = float(os.getenv("WOLF_QTY", "0") or 0)
DEFAULT_AVG = float(os.getenv("WOLF_AVG_COST", "0") or 0)

# Optional persistence for WOLF position
WOLF_PERSIST_MODE = (
    os.getenv("WOLF_PERSIST_MODE", "auto").strip().lower()
)  # none|file|redis|sqlite|auto






# Background task: compute and persist forecast error metrics (learning)


# Background task: auto-record actual prices for each forecast


# Background task: auto-record forecasts to SQLite


# ══════════════════════════════════════════════════════════════════════════════
# 48-HOUR FORECAST MODULE (Spec-Compliant)
# ══════════════════════════════════════════════════════════════════════════════












# Background task: Auto-generate forecasts every hour


# ══════════════════════════════════════════════════════════════════════════════
# 48H FORECAST API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
















WOLF_STATE_FILE = os.getenv("WOLF_STATE_FILE", "data/wolf_state.json")
REDIS_URL = os.getenv("REDIS_URL", "")
WOLF_SQLITE_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
WOLF_AUTOSAVE_S = int(os.getenv("WOLF_AUTOSAVE_S", "0"))  # 0 disables periodic autosave
SQLITE_FALLBACK = False

# Global REDIS client - lazy initialized on first use
REDIS = None


# ---------------------------------------------------------------------------
# Background live price updater
# ---------------------------------------------------------------------------
PRICE_AUTO_REFRESH_S = int(
    os.getenv("PRICE_AUTO_REFRESH_S", "7")
)  # cadence for attempted refreshes
_LAST_BG_PRICE_TS: float | None = None




# ---------------------------------------------------------------------------
# Background movers scanner
# ---------------------------------------------------------------------------



# If WOLF_SQLITE_PATH target is not writable/creatable (common in local dev where /data is not permitted),
# fall back to a workspace-local ./data/wolf.db to avoid sqlite OperationalError.
try:
    _fallback_needed = False
    try:
        _ensure_dir_for_file(WOLF_SQLITE_PATH)
        _test_path = WOLF_SQLITE_PATH + ".touch"
        with open(_test_path, "wb") as _f:
            _f.write(b"")
        os.remove(_test_path)
    except Exception:
        _fallback_needed = True
    if _fallback_needed:
        old = WOLF_SQLITE_PATH
        WOLF_SQLITE_PATH = os.path.join(os.getcwd(), "data", "wolf.db")
        _ensure_dir_for_file(WOLF_SQLITE_PATH)
        SQLITE_FALLBACK = True
        try:
            LOGGER.warning(
                "sqlite_path_fallback",
                extra={"component": "persist", "from": old, "to": WOLF_SQLITE_PATH},
            )
        except Exception:
            pass
except Exception as e:
    LOGGER.warning(f"sqlite_path_setup_failed: {e}")




# Initialize security tables on startup
try:
    _init_security_tables()
except Exception as e:
    logging.getLogger("ghost").error(f"Security tables initialization failed: {e}", exc_info=True)


# In-memory state (single WOLF position) and caches
STATE: dict[str, Any] = {
    "qty": DEFAULT_QTY,
    "avg_cost": round(DEFAULT_AVG, 2) if DEFAULT_AVG else 0.0,
    # UI compatibility state
    "active": True,
    "mode": "live",  # live|sim
    # Cash balance (unallocated) in account currency
    "cash": 0.0,
}

PRICE_CACHE: dict[str, dict[str, Any]] = {}  # symbol -> {price, prev_close, provider, ts}
NEWS_CACHE: dict[str, Any] = {"items": [], "ts": 0.0}

# --- Forecast overlay storage (MVP in-memory, move to SQLite in Phase 2) -----------
FORECAST_STORE: dict[
    str, dict[str, Any]
] = {}  # forecast_id -> {symbol, as_of, hours, path_mid, path_lo, path_hi}
FORECAST_ACTUALS: dict[str, list[dict[str, Any]]] = {}  # forecast_id -> [{t, p, provider}, ...]

# --- Manual price override (global) -------------------------------------------------
# Allows temporarily overriding the displayed price for a symbol with a TTL.
# Provider label will be reported as "manual" when active.
PRICE_OVERRIDE: dict[str, Any] = {"symbol": None, "price": None, "until": 0.0}




# Lightweight in-memory event ring (used by /logs/recent and SSE /events)
EVENTS: deque[dict] = deque(maxlen=500)
DIAG_COLLAPSE_DUPES: bool = True
_EVENT_SEQ = 0
_EVENT_LAST_TS: dict[tuple[str, str], float] = {}

# UI preferences (timezone and clock format)
GHOST_TZ = os.getenv("GHOST_TZ", "America/Chicago").strip() or "America/Chicago"
try:
    _h24_env = os.getenv("GHOST_CLOCK_24H", "0").strip().lower()
    GHOST_CLOCK_24H = _h24_env in ("1", "true", "yes", "on")
except Exception:
    GHOST_CLOCK_24H = False




# Alerts/dedupe state
ALERT_STATE: dict[str, Any] = {
    "last_signal": None,  # e.g., {"action":"BUY","price":x,"ts":...}
    "last_sent_ts": 0.0,
    "last_sent_ts_buy": 0.0,
    "last_sent_ts_sell": 0.0,
    "hold_override": False,
    "trailing_high": None,
    "trailing_low": None,
    "last_vol": None,
    "vol_ts": 0.0,
}

ALERT_THROTTLE_S = int(os.getenv("ALERT_THROTTLE_S", "60"))
ALERT_THROTTLE_BUY_S = int(os.getenv("ALERT_THROTTLE_BUY_S", str(ALERT_THROTTLE_S)))
ALERT_THROTTLE_SELL_S = int(os.getenv("ALERT_THROTTLE_SELL_S", str(ALERT_THROTTLE_S)))
ALERT_BUY_PCT = float(
    os.getenv("ALERT_BUY_PCT", "0.99")
)  # fixed: BUY if price < avg_cost * ALERT_BUY_PCT
ALERT_SELL_PCT = float(
    os.getenv("ALERT_SELL_PCT", "1.01")
)  # fixed: SELL if price > avg_cost * ALERT_SELL_PCT

# Alert modes and volatility gating
ALERT_MODE = os.getenv("ALERT_MODE", "fixed").strip().lower()  # fixed|band|trailing
# Scheduled market open/close status cards
SCHEDULE_OPEN_CLOSE = int(os.getenv("ALERT_SCHEDULE_OPEN_CLOSE", "0"))  # 1 to enable
SCHEDULE_WINDOW_S = int(
    os.getenv("ALERT_SCHEDULE_WINDOW_S", "300")
)  # fire within +/- this many seconds
BAND_PCT = float(os.getenv("BAND_PCT", "0.02"))  # band mode: +/- around avg
TRAIL_SELL_PCT = float(os.getenv("TRAIL_SELL_PCT", "0.05"))  # trailing: drop from trailing_high
TRAIL_BUY_PCT = float(os.getenv("TRAIL_BUY_PCT", "0.05"))  # trailing: rise from trailing_low

VOL_GATE = int(os.getenv("VOL_GATE", "0"))  # 1 to enable gating by volatility
VOL_LOOKBACK_DAYS = int(os.getenv("VOL_LOOKBACK_DAYS", "20"))
VOL_K = float(os.getenv("VOL_K", "1.0"))
VOL_TTL_S = int(os.getenv("VOL_TTL_S", "600"))

TELEGRAM_HEARTBEAT_ON_START = int(os.getenv("TELEGRAM_HEARTBEAT_ON_START", "0"))
PROTECT_ALERTS_TEST = int(os.getenv("PROTECT_ALERTS_TEST", "0"))

# Optional AI advisor (LLM) — disabled by default (standardized on AGENTS_ENABLED/AGENT_MODEL)
AGENTS_ENABLED = int(os.getenv("AGENTS_ENABLED", os.getenv("AI_ON", "0")))
AI_ON = AGENTS_ENABLED  # Backward-compat alias
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()  # ollama|openai
AGENT_MODEL = os.getenv("AGENT_MODEL", os.getenv("AI_MODEL", "llama3.1:8b")).strip()
AI_MODEL = AGENT_MODEL  # Backward-compat alias
AI_TIMEOUT_S = int(os.getenv("AI_TIMEOUT_S", "10"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = (os.getenv("OPENAI_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()

ALERT_WEBHOOK_URLS: list[str] = [
    u.strip() for u in os.getenv("ALERT_WEBHOOK_URLS", "").split(",") if u.strip()
]
SLACK_WEBHOOK_URLS: list[str] = [
    u.strip() for u in os.getenv("SLACK_WEBHOOK_URLS", "").split(",") if u.strip()
]
# Runtime-configurable alert templates
ALERT_CONFIG = {
    "signal_template": os.getenv("ALERT_SIGNAL_TEMPLATE", "").strip() or None,
    "status_template": os.getenv("ALERT_STATUS_TEMPLATE", "").strip() or None,
}

# ── News sentiment fusion (env-toggled; defaults off to avoid regressions) ─────────────
NEWS_SENTIMENT_ON = int(os.getenv("NEWS_SENTIMENT_ON", "1"))
FINBERT_ON = int(os.getenv("FINBERT_ON", "0"))
NEWS_LOOKBACK_MIN = int(os.getenv("NEWS_LOOKBACK_MIN", "30"))  # FIX (Step 8): was 240 → 30 min to weight recent news
NEWS_DECAY_HALF_MIN = int(os.getenv("NEWS_DECAY_HALF_MIN", "180"))
SENT_ALPHA = float(os.getenv("SENT_ALPHA", "0.7"))  # weight for price_signal
SENT_BETA = float(os.getenv("SENT_BETA", "0.3"))  # weight for news_score
FUSE_DECISION_ON = int(os.getenv("FUSE_DECISION_ON", "0"))
FUSE_GAMMA_MACRO = float(os.getenv("FUSE_GAMMA_MACRO", "0.2"))  # extra weight for macro pressure
MODULE_WEIGHTING_ON = int(os.getenv("MODULE_WEIGHTING_ON", "1"))
FUSE_T_BUY = float(os.getenv("FUSE_T_BUY", "0.15"))
FUSE_T_SELL = float(os.getenv("FUSE_T_SELL", "-0.15"))

_FINBERT_PIPE = None  # lazy-loaded sentiment pipeline
_NEWS_SENT_CACHE: dict[
    str, dict[str, Any]
] = {}  # id -> {"sent": float, "engine": str, "ts": float}






_BEARISH = {
    "downgrade": -0.5,
    "plunge": -0.6,
    "fall": -0.4,
    "slump": -0.5,
    "cut": -0.2,
    "miss": -0.4,
    "bear": -0.3,
    "delay": -0.2,
    "loss": -0.5,
    "lawsuit": -0.5,
}
_BULLISH = {
    "upgrade": 0.5,
    "surge": 0.6,
    "rise": 0.4,
    "beat": 0.4,
    "raise": 0.2,
    "bull": 0.3,
    "win": 0.4,
    "profit": 0.5,
    "record": 0.3,
    "contract": 0.2,
}














# ── Minimal templating helper and formatters ─────────────────────────────────────












# Provider breaker config
PROVIDER_FAIL_THRESHOLD = int(os.getenv("PROVIDER_FAIL_THRESHOLD", "3"))
PROVIDER_BACKOFF_S = int(os.getenv("PROVIDER_BACKOFF_S", "30"))
PROVIDER_BACKOFF_MAX_S = int(os.getenv("PROVIDER_BACKOFF_MAX_S", "300"))

# Rate limit + backoff tracking for data providers
PROVIDER_BACKOFF: dict[str, dict[str, float]] = {  # provider -> {last_429, backoff_until, failures}
    # example: "yahoo": {"last_429": 0.0, "backoff_until": 0.0, "failures": 0}
}

_PROVIDER_LIMITERS: dict[str, AsyncRateLimiter] = {
    "polygon": AsyncRateLimiter(rate=100, per=60.0),  # Scaled for unlimited symbols
    "polygon_intraday": AsyncRateLimiter(rate=100, per=60.0),
    "alphavantage": AsyncRateLimiter(rate=75, per=60.0),  # Premium tier assumed
    "yahoo": AsyncRateLimiter(rate=60, per=60.0),  # Aggressive but sustainable
    "yfinance": AsyncRateLimiter(rate=30, per=60.0),  # Increased from 4
}

BACKOFF_BASE_S = 30.0
BACKOFF_MAX_S = 600.0








# ── Metrics (reload-safe) ───────────────────────────────────────────────────────────────
_H_SNAPSHOT_BUILD = None
_C_SNAPSHOT_FAIL = None
_G_UP = None
# Alert metrics
_C_ALERT_SENT = None
_C_ALERT_THROTTLED = None
_G_ALERT_HOLD = None
_G_ALERT_MODE = None
# Provider metrics
_H_PROVIDER_FETCH = None
_C_PROVIDER_FETCH = None
_H_TG_SEND = None
_C_TG_SEND = None
_G_ALERT_QUEUE_LEN = None
_C_ALERT_RETRIES = None
_C_RATE_LIMIT_DROPS = None
_G_RATE_LIMIT_TOKENS = None
_G_FINAL_SCORE = None
_G_WHY_NOW_COUNT = None
_C_LLM_CALLS = None
_C_LLM_DECISIONS = None
_G_LLM_CONFIDENCE = None
_C_HTTP_POOL_USED = None
_C_HTTP_DIRECT_USED = None
_C_AI_MEMORY_REQ = None
_H_AI_MEMORY_LAT = None

# Crypto metrics (initialized to None)
_C_CRYPTO_PRICE_FETCH = None
_C_CRYPTO_PREDICT_DURATION = None
_G_CRYPTO_PREDICTION_MAPE = None
_G_SENTIMENT_SCORE = None
_G_MACRO_CONFIDENCE = None










# ── HTTP session pooling (optional) ───────────────────────────────────────────────────
_HTTP_SESSIONS: dict[str, requests.Session] = {}




# ── Forecast overlay persistence and APIs ──────────────────────────────────────────────
# Runtime toggles
OVERLAY_ENABLED = int(os.getenv("OVERLAY_ENABLED", "1"))
OVERLAY_DT_MINUTES = int(os.getenv("OVERLAY_DT_MINUTES", "60"))
LEARNING_ENABLED = int(os.getenv("LEARNING_ENABLED", "1"))
BAND_WIDEN_FACTOR = float(os.getenv("BAND_WIDEN_FACTOR", "1.0"))


























# ══════════════════════════════════════════════════════════════════════════════
# Ghost Prediction Endpoints
# ══════════════════════════════════════════════════════════════════════════════

from services import predictor
try:
    from services import outcome_reconciler
except ImportError:
    outcome_reconciler = None  # v1 removed — using outcome_reconciler_v2




















# =============================================================================
# RESEARCH MODULE ENDPOINTS
# =============================================================================

# NOTE: Batch route MUST come before {symbol} route to prevent "batch" being captured as symbol












# =============================================================================
# GHOST BRAIN INTELLIGENCE ENDPOINTS
# =============================================================================















# =============================================================================
# OPUS BRAIN ENDPOINTS - CLAUDE AI POWERED INTELLIGENCE
# =============================================================================











# =============================================================================
# STOCK ENGINE API (Jan 26, 2026)
# Dedicated endpoint for the new stock-specific prediction engine
# =============================================================================











































# =============================================================================
# AUTO-CALIBRATION ENDPOINTS
# =============================================================================















# ============================================================================
# MOMENTUM TRACKER ENDPOINTS - Track prediction confidence trends
# ============================================================================









# ===========================
# CASCADING PREDICTIONS API
# ===========================































# ============================================================================
# V3 COCKPIT ENDPOINTS - For cockpit_v3.html UI
# ============================================================================





# Watchlist enriched cache (30s TTL) — prevents thundering herd from UI polling
_WATCHLIST_ENRICHED_CACHE: dict = {}
_WATCHLIST_ENRICHED_CACHE_AT: float = 0.0
_WATCHLIST_ENRICHED_CACHE_TTL: float = 30.0  # 30 seconds
_WATCHLIST_ENRICHED_LOCK: asyncio.Lock | None = None








# ──────────────────────────────────────────────────────────────────────────────
# V4 ENDPOINTS — Extracted to routes/ modules (Step 10)
# /api/v4/picks  → routes/picks.py
# /api/v4/history → routes/history.py
# ──────────────────────────────────────────────────────────────────────────────

# /api/v4/history → extracted to routes/history.py (Step 10)
# /api/v3/heartbeat/status → extracted to routes/heartbeat.py (Step 10)


# Alias for /api/v3/watchlist/user (compatibility with personal watchlist router)


# VIP snapshot cache (30s TTL - reduced from 5min due to timeout issues)
_VIP_SNAPSHOT_CACHE = {"data": None, "timestamp": 0, "ttl": 30}







# ============================================================================
# DATA-ENHANCED PREDICTION ENDPOINTS
# ============================================================================





























# ============================================================================
#  OPTION A OBSERVABILITY ENDPOINTS
# ============================================================================









# ══════════════════════════════════════════════════════════════
# INTEGRITY AUDIT — Self-healing health check system
# ══════════════════════════════════════════════════════════════









# ═══════════════════════════════════════════════════════════
# v5 COCKPIT ENDPOINTS — Market Ticker + AI Brain + Financials
# ═══════════════════════════════════════════════════════════

# ── Market Index Price Cache (60s TTL, multi-provider fallback) ──
_INDEX_CACHE: dict = {}  # {symbol: {"price": float, "prev": float, "ts": float}}
_INDEX_CACHE_TTL = 60  # seconds












# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM INVENTORY — extracted to routes/subsystems.py (Step 10)
# /api/v4/subsystems → routes/subsystems.py
# ═══════════════════════════════════════════════════════════════════════════════


















































# ============================================================================
# END V3 COCKPIT ENDPOINTS
# ============================================================================
















# =============================================================================
# CRYPTO PREDICTION API
# =============================================================================

# Lazy-load crypto module to avoid hard dependency
_crypto_engine = None
_crypto_provider = None














# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO FEATURE PARITY - NEW ENDPOINTS
# Added: Oct 14, 2025 - Bring crypto to full parity with stock Ghost
# ═══════════════════════════════════════════════════════════════════════════
















# ═══════════════════════════════════════════════════════════════════════════════
# AI ADVISOR - Autonomous market scanner + recommendations
# ═══════════════════════════════════════════════════════════════════════════════






















# ═══════════════════════════════════════════════════════════════════════════════
# TO THE MOON: ADVANCED SYSTEMS (Tier 2 + Tier 3)
# ═══════════════════════════════════════════════════════════════════════════════
























# Initialize gauges after definitions
_set_mode_gauge()
_set_hold_gauge()


# ── Provider Circuit Breaker ──────────────────────────────────────────────────────────
_PROVIDER_BREAKERS: dict[str, dict[str, Any]] = {
    name: {
        "state": "closed",  # closed|open|half-open
        "failures": 0,
        "backoff_factor": 0,
        "open_until_ts": 0.0,
    }
    for name in ("alphavantage", "polygon", "yfinance")
}




























































# ── Market hours helpers (NYSE basic approximation; holidays not modeled) ─────
_TZ_NY = ZoneInfo("America/New_York") if ZoneInfo else None














# Autosave worker
_AUTOSAVE_WORKER: threading.Thread | None = None
_AUTOSAVE_STOP = threading.Event()








# ── Control endpoints: master save and engine reset ──────────────────────────




atexit.register(_stop_autosave_worker)
















# Legacy format function (keep for backward compatibility)








_ALERT_QUEUE: "_queue.Queue[dict]" = _queue.Queue(maxsize=1000)
_ALERT_WORKER: threading.Thread | None = None
_ALERT_STOP = threading.Event()










atexit.register(_stop_alert_worker)
atexit.register(_persist_save)
try:
    atexit.register(lambda: _stop_schedule_worker())
except Exception:
    pass


# Request logging middleware (structured)


LOG_SAMPLE_RATE = float(os.getenv("LOG_SAMPLE_RATE", "1.0"))
LOG_SKIP_PATHS = [
    p.strip()
    for p in os.getenv(
        "LOG_SKIP_PATHS",
        "/assets,/static,/img,/favicon.ico,/metrics,/api/cockpit/stream,/events",
    ).split(",")
    if p.strip()
]




# ── Scheduled market open/close announcer ─────────────────────────────────────────────
_SCHED_WORKER: threading.Thread | None = None
_SCHED_STOP = threading.Event()
_SCHED_LAST_OPEN_DAY: str | None = None
_SCHED_LAST_CLOSE_DAY: str | None = None








# ── Ghost Prediction Outcome Reconciler ──────────────────────────────────────────────
_RECONCILER_WORKER: threading.Thread | None = None
_RECONCILER_STOP = threading.Event()








# ── Live Accuracy Tracking Worker ────────────────────────────────────────────────
_ACCURACY_TRACKER: threading.Thread | None = None
_ACCURACY_STOP = threading.Event()












# Optional admin IP allowlist for write operations (POST/PUT/PATCH/DELETE)
ADMIN_IP_ALLOWLIST = [
    p.strip() for p in os.getenv("ADMIN_IP_ALLOWLIST", "").split(",") if p.strip()
]

# Simple in-memory idempotency cache for /api/alerts/dispatch
_IDEMPOTENCY_TTL_S = int(os.getenv("IDEMPOTENCY_TTL_S", "300"))
_IDEMP_CACHE: dict[str, dict[str, Any]] = {}
_IDEMP_CACHE_TS: dict[str, float] = {}







# Per-action throttle for /alerts/status and merge-guard
STATUS_THROTTLE_S = int(os.getenv("STATUS_THROTTLE_S", "30"))
STATUS_MERGE_TTL_S = int(os.getenv("STATUS_MERGE_TTL_S", "60"))
_STATUS_LAST_TS: float = 0.0
_STATUS_LAST_HASH: str | None = None


# ── Advisory orders (SQLite-backed) ───────────────────────────────────────────
ORDERS_TABLE = "orders"
























# --- Memory MCP Integration Endpoints -------------------------------------

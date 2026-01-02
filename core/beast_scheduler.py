"""
Ghost Beast Scheduler
Periodic market predictions and alerts for stocks and crypto
"""

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Will be set by wolf_app.py
REDIS_CLIENT = None
LOGGER = None
GET_PRICE_FUNC = None  # func(symbol, market) -> (price, prev_close, provider, after_hours)
RUN_PREDICTION_FUNC = None  # func(symbol, market, horizon) -> prediction dict
TELEGRAM_ALERTS_MODULE = None  # telegram_alerts module

# Scheduler state
_SCHEDULER_THREAD: threading.Thread | None = None
_SCHEDULER_STOP = threading.Event()

# Watch lists - EXPANDED WITH 300+ HIGHEST MOMENTUM SYMBOLS FROM CASH APP
STOCK_SYMBOLS = [
    # Tech Giants & FAANG
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AMZN",
    
    # CHINA ADRs (Move together - theme trades)
    "BABA", "JD", "PDD", "XPEV", "LI",  # IQ, YMM, TAL, TME, NIO already added below
    
    # BTC MINERS (Follow Bitcoin - correlated trades)
    "MARA", "CLSK", "BITF", "HIVE",  # RIOT, HUT already added below
    
    # AI THEME STOCKS
    "AI", "BBAI", "UPST",  # SOUN, PLTR already added below
    
    # CANNABIS MEGA MOVERS (Today's +40-70% gainers!)
    "TLRY", "CGC", "ACB", "SNDL", "TCNNF", "YCBD", "WEED",
    
    # TOP GAINERS - 52 WEEK (500%+ returns)
    "QMMM", "RGC", "TAWNF", "ABVX", "DSVSF", "CDTX", "TMC", "CELC", "TERN", "SNDK",
    "ONDS", "QBTS", "OKLO", "BMNR", "COGT", "EOSE", "OLMA", "FNMA", "ZBIO", "SATS",
    "PL", "FMCC", "GRAL", "BE", "VSAT",
    
    # TODAY'S TOP GAINERS (+10%+)
    "YOU", "RIVN", "LULU", "RH", "VERA", "SHAK", "CCC", "IMVT", "GE", "CMG",
    "VSCO", "MOS", "CYTK", "CAVA", "ENVA", "SMG", "GLPI", "OSCR", "PLMR", "LIN",
    "COKE", "AJG", "ROIV", "LFST", "ATAT", "PACS", "ERIE", "WLTH", "RYM",
    
    # TODAY'S TOP LOSERS (High Volume)
    "FRMI", "SEI", "MOD", "ALAB", "RCUS", "LITE", "ECG", "NTSK", "CLS", "FLNC",
    "RMBS", "AAOI", "SMR", "FN", "COHR", "CIEN", "VRT", "ASTS", "HPP", "CRWV",
    "PLAB", "SANM", "HUT", "TTMI", "LGN", "PRIM", "LEU", "GLW", "GLXY", "NXT",
    "SMTC", "LBRT", "VAL", "ANET", "APH", "STRL", "VIAV", "UEC", "DQ", "STX",
    "Q", "MKSI", "FIX", "BTDR", "SBET", "MIR", "IE", "SXI", "NVMI", "UUUU",
    "NVT", "IOT", "CEG", "RBRK", "MU", "TSEM", "SEZL", "WDC", "MTSI", "FIG",
    "U", "LOGI", "SIMO", "ZIM", "USAR", "WCC", "FSLR", "WRBY", "CAMT", "IPGP",
    "PWR", "APP", "VIA", "PONY", "AMKR", "CRCL", "AAON", "ESI", "DELL", "CIFR",
    "QS", "BLTE", "AMBA", "AG", "POWL", "JBL", "TEL", "WULF", "SITM", "IRM",
    "FLY", "HP",
    
    # User's Original Watchlist
    "TAL", "ARCT", "TME", "HIMS", "PFE", "RDFN", "BILL", "XPO",
    
    # User's Cash App Top Holdings
    "SHOP", "AVGO", "BAC", "V", "ADBE", "ABNB",
    
    # Original Tracked
    "WOLF",
    
    # High Volume Movers
    "SPY", "QQQ", "GME", "AMC", "PLTR", "SOFI", "LCID", "NIO", "LRCX",
    
    # Financial/Industrial
    "GS", "JPM", "MA", "C", "WFC", "MS",
    
    # Communication/Media
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    
    # Energy/Utilities  
    "XOM", "CVX", "NEE", "DUK", "SO", "D",
    
    # Retail/Consumer
    "WMT", "TGT", "COST", "HD", "LOW", "NKE", "SBUX",
    
    # Healthcare/Pharma
    "UNH", "JNJ", "ABT", "LLY", "MRK", "ABBV", "TMO",
    
    # Transportation/Logistics
    "UBER", "LYFT", "FDX", "UPS", "DAL", "UAL",
    
    # Social/Emerging
    "SNAP", "PINS", "DASH", "SQ", "RBLX", "HOOD",
    
    # Real Estate
    "VICI", "O", "PLD", "AMT", "CCI",
    
    # Semiconductors/Hardware
    "AMD", "INTC", "QCOM", "TSM",
    
    # Cloud/SaaS
    "CRM", "ORCL", "NOW", "SNOW", "DDOG", "NET",
    
    # Biotech/Genomics
    "GILD", "VRTX", "BIIB", "REGN", "ILMN",
    
    # Industrial/Defense
    "BA", "LMT", "RTX", "HON", "GD",
    
    # Financial Services
    "BLK", "SCHW", "AXP", "USB", "PNC",
    
    # Additional High Volume Stocks
    "MMYT", "EVCM", "DNLI", "BLLN", "CAE", "SLNO", "LNTH", "EWTX", "NAVN", "MTN", "MLYS",
    
    # COMPLETE CASH APP STOCK UNIVERSE (A-Z)
    # A-B
    "ABCL", "ACMR", "AA", "AEO", "APGE", "MT", "ARWR", "BEAM", "BBIO", "BTSG", "BRKR", "BMBL",
    # C-D
    "CSIQ", "CPRI", "CAH", "CVNA", "CNX", "DOCN", "DLTR", "DUOL", "DY",
    # E-G
    "EPAM", "FSLY", "FLNC", "YMM", "GKOS", "GMED", "GH", "HAL", "HAE", "HL",
    # I-J
    "INSP", "IQ", "ITRI", "JBHT", "JAZZ", "FROG",
    # K-L
    "KSS", "LMND", "LAC", "LUMN",
    # M-O
    "MRVL", "KSS", "NTRA", "NOK", "ONTO",
    # P-R
    "PAAS", "PLTK", "PLUG", "RGEN", "RIOT",
    # S-T
    "SOUN", "SCCO", "STLD", "SHOO", "SYM", "TNDM", "TDC", "TEVA", "TGTX", "DJT", "TWLO",
    # U-W
    "PATH", "UPWK", "VCYT", "VFC", "COCO", "WAT", "WFRD",
    # User's Additional Watchlist
    "SLV", "AMAT", "WBD", "ASML", "CAT", "XBI"
]

CRYPTO_SYMBOLS = [
    # Top 10 by Market Cap (from user's Cash App list)
    "BTC", "ETH", "USDT", "XRP", "BNB", "USDC", "SOL", "DOGE", "ADA", "TRX",
    
    # Top DeFi & Layer 1/2 (11-30)
    "WSTETH", "BCH", "WBTC", "WBETH", "WETH", "LINK", "WEETH", "LEO", "XLM", "ZEC",
    "XMR", "USDE", "LTC", "CBBTC", "BTCB", "SUI", "AVAX", "DAI", "HBAR", "SHIB",
    
    # Emerging/High Growth (31-50)
    "MNT", "TON", "PYUSD", "WLFI", "CRO", "SUSDE", "UNI", "DOT", "TAO", "AAVE",
    "CC", "RAIN", "BGB", "OKB", "USDF", "ASTER", "NEAR", "ETC", "M", "ENA",
    
    # VIP Microcaps (Ghost originals)
    "PEPE", "WEPE", "LILPEPE", "DORKL", "SLOTH", "APC",
    
    # Meme/Trending
    "FLOKI", "BONK", "WIF", "TRUMP",
    
    # Staking/Wrapped Variants
    "STETH", "JITOSOL", "RETH", "BNSOL", "SOLVBTC",
    
    # Altcoins with volume
    "FIL", "ATOM", "VET", "ALGO", "ICP", "QNT", "FLR", "XDC",
    
    # Governance/Exchange Tokens
    "KCS", "GT", "HYPE"
]

# Timezone
CHICAGO_TZ = ZoneInfo("America/Chicago")


def _get_chicago_time():
    """Get current time in Chicago timezone"""
    return datetime.now(CHICAGO_TZ)


def _is_market_day(dt):
    """Check if it's a weekday (Mon-Fri)"""
    return dt.weekday() <= 4


def _send_prediction_alert(symbol: str, market: str, horizon: str):
    """
    Generate and send a prediction alert for a symbol

    Args:
        symbol: Trading symbol
        market: "stock" or "crypto"
        horizon: "SHORT" or "LONG"
    """
    if not GET_PRICE_FUNC or not RUN_PREDICTION_FUNC or not TELEGRAM_ALERTS_MODULE:
        if LOGGER:
            LOGGER.warning(f"Missing functions for {symbol} alert")
        return

    try:
        # Get current price
        price_result = GET_PRICE_FUNC(symbol, market)
        if not price_result:
            if LOGGER:
                LOGGER.warning(f"No price data for {symbol}")
            return

        price, prev_close, provider, after_hours = price_result

        # Check if price is stale (prev close fallback during closed market)
        if after_hours and market == "stock":
            if LOGGER:
                LOGGER.info(f"Skipping {symbol} alert - market closed, prev close only")
            return

        # Run prediction
        prediction = RUN_PREDICTION_FUNC(symbol, market, horizon)
        if not prediction:
            if LOGGER:
                LOGGER.warning(f"No prediction for {symbol}")
            return

        # Build price metadata
        price_meta = {
            "price": price,
            "prev_close": prev_close,
            "provider": provider,
            "after_hours": after_hours,
        }

        # Send alert via telegram_alerts module
        success = TELEGRAM_ALERTS_MODULE.send_alert(
            symbol=symbol,
            market=market,
            horizon_bucket=horizon,
            prediction=prediction,
            price_meta=price_meta,
        )

        if success:
            if LOGGER:
                LOGGER.info(f"✅ Sent {horizon} alert for {symbol}")
        else:
            if LOGGER:
                LOGGER.info(f"⏭️  Skipped {horizon} alert for {symbol} (duplicate)")

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Error sending {symbol} alert: {e}")


def _run_stock_predictions(horizon: str):
    """Run predictions for all stock symbols"""
    for symbol in STOCK_SYMBOLS:
        _send_prediction_alert(symbol, "stock", horizon)
        time.sleep(0.5)  # Rate limit


def _run_crypto_predictions(horizon: str):
    """Run predictions for all crypto symbols"""
    import os
    exclude_symbols = set(s.strip().upper() for s in os.getenv("GHOST_EXCLUDE_SYMBOLS", "").split(",") if s.strip())
    
    for symbol in CRYPTO_SYMBOLS:
        if symbol.upper() in exclude_symbols:
            if LOGGER:
                LOGGER.debug(f"⏭️  Skipping {symbol} (in GHOST_EXCLUDE_SYMBOLS)")
            continue
        _send_prediction_alert(symbol, "crypto", horizon)
        time.sleep(0.5)  # Rate limit


def _check_schedule():
    """
    Check if it's time to run predictions

    Stock schedule (CT):
    - 07:55: Pre-market (SHORT + LONG)
    - 09:35: Market open (SHORT + LONG)
    - 12:00: Midday (SHORT)
    - 15:10: Close (SHORT + LONG)

    Crypto schedule (CT):
    - Every 2 hours on the hour (00:00, 02:00, 04:00, etc.)
    - Run SHORT + LONG
    """
    now = _get_chicago_time()
    hour = now.hour
    minute = now.minute

    # Stock schedules (weekdays only)
    if _is_market_day(now):
        # Pre-market: 07:55
        if hour == 7 and minute == 55:
            if LOGGER:
                LOGGER.info("📊 Running pre-market stock predictions")
            _run_stock_predictions("SHORT")
            _run_stock_predictions("LONG")
            time.sleep(60)  # Skip next minute
            return

        # Market open: 09:35
        if hour == 9 and minute == 35:
            if LOGGER:
                LOGGER.info("📊 Running market open stock predictions")
            _run_stock_predictions("SHORT")
            _run_stock_predictions("LONG")
            time.sleep(60)
            return

        # Midday: 12:00
        if hour == 12 and minute == 0:
            if LOGGER:
                LOGGER.info("📊 Running midday stock predictions")
            _run_stock_predictions("SHORT")
            time.sleep(60)
            return

        # Close: 15:10
        if hour == 15 and minute == 10:
            if LOGGER:
                LOGGER.info("📊 Running market close stock predictions")
            _run_stock_predictions("SHORT")
            _run_stock_predictions("LONG")
            time.sleep(60)
            return

    # Crypto schedule: Every 2 hours on the hour
    if minute == 0 and hour % 2 == 0:
        if LOGGER:
            LOGGER.info(f"₿ Running {hour:02d}:00 crypto predictions")
        _run_crypto_predictions("SHORT")
        _run_crypto_predictions("LONG")
        time.sleep(60)
        return


def _scheduler_loop():
    """Main scheduler loop"""
    if LOGGER:
        LOGGER.info("🚀 Beast scheduler started")

    while not _SCHEDULER_STOP.is_set():
        try:
            _check_schedule()
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"Scheduler loop error: {e}")
            time.sleep(60)


def start_beast_scheduler():
    """Start the beast scheduler"""
    global _SCHEDULER_THREAD

    if _SCHEDULER_THREAD is None or not _SCHEDULER_THREAD.is_alive():
        _SCHEDULER_STOP.clear()
        _SCHEDULER_THREAD = threading.Thread(
            target=_scheduler_loop, name="beast-scheduler", daemon=True
        )
        _SCHEDULER_THREAD.start()

        if LOGGER:
            LOGGER.info("✅ Beast scheduler started")

        print("[BEAST SCHEDULER] Started - Stock: 07:55, 09:35, 12:00, 15:10 CT | Crypto: Every 2h")


def stop_beast_scheduler():
    """Stop the beast scheduler"""
    try:
        _SCHEDULER_STOP.set()
        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
            _SCHEDULER_THREAD.join(timeout=2.0)

        if LOGGER:
            LOGGER.info("⏹️  Beast scheduler stopped")
    except Exception:
        pass


def trigger_manual_prediction(symbol: str, market: str, horizon: str = "SHORT") -> bool:
    """
    Manually trigger a prediction alert (for testing)

    Args:
        symbol: Trading symbol
        market: "stock" or "crypto"
        horizon: "SHORT" or "LONG"

    Returns:
        True if alert was sent successfully
    """
    try:
        _send_prediction_alert(symbol, market, horizon)
        return True
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Manual prediction error: {e}")
        return False

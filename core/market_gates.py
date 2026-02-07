"""
Market Gates Module - Quick Wins for Better Predictions
=======================================================
Implements 4 key filters to improve Ghost's BUY/SELL accuracy:

1. REGIME FILTER - Is the market bullish/bearish?
2. VIX GATE - Are people scared?  
3. CONFIRMATION COUNTER - Do multiple signals agree?
4. LOSER ANALYSIS - Query patterns in failed BUYs

Author: Ghost Protocol Team
Created: January 2026
"""

import os
import time
import logging
import sqlite3
from typing import Any, Optional
from datetime import datetime, timedelta

LOGGER = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (Can be overridden via environment variables)
# ============================================================================

# Regime Filter Settings
REGIME_FILTER_ENABLED = os.getenv("REGIME_FILTER", "1") == "1"
SPY_MA_PERIOD = int(os.getenv("SPY_MA_PERIOD", "20"))  # 20-day MA
BTC_TREND_DAYS = int(os.getenv("BTC_TREND_DAYS", "7"))  # 7-day trend
BTC_TREND_THRESHOLD = float(os.getenv("BTC_TREND_THRESHOLD", "-5.0"))  # -5% = bearish

# VIX Gate Settings
VIX_GATE_ENABLED = os.getenv("VIX_GATE", "1") == "1"
VIX_PANIC_THRESHOLD = float(os.getenv("VIX_PANIC", "30"))  # No BUYs above this
VIX_FEAR_THRESHOLD = float(os.getenv("VIX_FEAR", "25"))   # Half confidence above this
VIX_CAUTION_THRESHOLD = float(os.getenv("VIX_CAUTION", "20"))  # Reduced confidence

# Confirmation Counter Settings
MIN_CONFIRMATIONS_HIGH = int(os.getenv("MIN_CONF_HIGH", "4"))  # For high-confidence BUY
MIN_CONFIRMATIONS_LOW = int(os.getenv("MIN_CONF_LOW", "3"))   # For low-confidence BUY


# ============================================================================
# 1. REGIME FILTER - "Is the market bullish?"
# ============================================================================

class RegimeFilter:
    """
    Checks if the overall market is in a bullish or bearish regime.
    
    Logic:
    - SPY above 20-day MA = Bullish
    - SPY below 20-day MA = Bearish (block BUY signals)
    - BTC down >5% over 7 days = Crypto bearish (block crypto BUYs)
    """
    
    def __init__(self):
        self.last_check = 0
        self.cache_ttl = 300  # 5 minutes cache
        self._spy_above_ma = None
        self._btc_7d_trend = None
        self._regime = None
    
    async def get_spy_regime(self) -> dict[str, Any]:
        """
        Get SPY position vs 20-day MA.
        
        Returns:
            {
                "above_20ma": True/False,
                "spy_price": 450.50,
                "spy_ma20": 448.00,
                "regime": "bull"/"bear"/"unknown"
            }
        """
        try:
            # Try Polygon first
            import aiohttp
            polygon_key = os.getenv("POLYGON_API_KEY")
            
            if polygon_key:
                async with aiohttp.ClientSession() as session:
                    # Get SPY quote
                    quote_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?adjusted=true&apiKey={polygon_key}"
                    async with session.get(quote_url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("results"):
                                spy_price = data["results"][0].get("c", 0)  # close
                                
                                # Get 20-day bars for MA calculation
                                end = datetime.now().strftime("%Y-%m-%d")
                                start = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
                                
                                bars_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start}/{end}?adjusted=true&apiKey={polygon_key}"
                                async with session.get(bars_url, timeout=10) as bars_resp:
                                    if bars_resp.status == 200:
                                        bars_data = await bars_resp.json()
                                        results = bars_data.get("results", [])
                                        
                                        if len(results) >= SPY_MA_PERIOD:
                                            # Calculate 20-day MA
                                            closes = [r["c"] for r in results[-SPY_MA_PERIOD:]]
                                            spy_ma20 = sum(closes) / len(closes)
                                            
                                            above_ma = spy_price > spy_ma20
                                            regime = "bull" if above_ma else "bear"
                                            
                                            self._spy_above_ma = above_ma
                                            self._regime = regime
                                            
                                            LOGGER.info(
                                                f"📊 SPY Regime: {regime.upper()} "
                                                f"(${spy_price:.2f} vs MA20 ${spy_ma20:.2f})"
                                            )
                                            
                                            return {
                                                "above_20ma": above_ma,
                                                "spy_price": spy_price,
                                                "spy_ma20": round(spy_ma20, 2),
                                                "regime": regime
                                            }
            
            # Fallback: Use cached or assume neutral
            LOGGER.warning("Unable to fetch SPY data - using neutral regime")
            return {
                "above_20ma": True,  # Don't block by default
                "spy_price": 0,
                "spy_ma20": 0,
                "regime": "unknown"
            }
            
        except Exception as e:
            LOGGER.error(f"Regime filter SPY check failed: {e}")
            return {"above_20ma": True, "regime": "unknown", "error": str(e)}
    
    async def get_btc_trend(self) -> dict[str, Any]:
        """
        Get BTC 7-day trend for crypto regime.
        
        Returns:
            {
                "trend_7d_pct": -5.2,
                "btc_price": 43000,
                "btc_price_7d_ago": 45000,
                "crypto_regime": "bull"/"bear"
            }
        """
        # Try CoinGecko first
        try:
            from core.crypto.crypto_providers import CoinGeckoProvider
            
            provider = CoinGeckoProvider()
            history = provider.get_historical("BTC", days=BTC_TREND_DAYS + 1)
            
            if history and len(history) >= 2:
                price_now = history[-1]["price"]
                price_7d_ago = history[0]["price"]
                trend_pct = ((price_now - price_7d_ago) / price_7d_ago) * 100
                
                crypto_regime = "bear" if trend_pct < BTC_TREND_THRESHOLD else "bull"
                self._btc_7d_trend = trend_pct
                
                LOGGER.info(
                    f"₿ BTC Trend: {trend_pct:+.1f}% over {BTC_TREND_DAYS}d "
                    f"→ Crypto regime: {crypto_regime.upper()}"
                )
                
                return {
                    "trend_7d_pct": round(trend_pct, 2),
                    "btc_price": price_now,
                    "btc_price_7d_ago": price_7d_ago,
                    "crypto_regime": crypto_regime
                }
                
            LOGGER.warning("CoinGecko returned insufficient BTC history, trying Binance...")
            
        except Exception as e:
            LOGGER.warning(f"CoinGecko BTC trend failed: {e}, trying Binance...")
        
        # Fallback to Binance
        try:
            from core.providers.binance_ohlcv import get_binance_ohlcv
            
            # Get 7 days of daily candles
            bars = get_binance_ohlcv("BTC", interval="1d", limit=BTC_TREND_DAYS + 1)
            
            if bars and len(bars) >= 2:
                price_now = bars[-1]["close"]
                price_7d_ago = bars[0]["close"]
                trend_pct = ((price_now - price_7d_ago) / price_7d_ago) * 100
                
                crypto_regime = "bear" if trend_pct < BTC_TREND_THRESHOLD else "bull"
                self._btc_7d_trend = trend_pct
                
                LOGGER.info(
                    f"₿ BTC Trend (Binance): {trend_pct:+.1f}% over {BTC_TREND_DAYS}d "
                    f"→ Crypto regime: {crypto_regime.upper()}"
                )
                
                return {
                    "trend_7d_pct": round(trend_pct, 2),
                    "btc_price": price_now,
                    "btc_price_7d_ago": price_7d_ago,
                    "crypto_regime": crypto_regime,
                    "source": "binance"
                }
                
        except Exception as e:
            LOGGER.error(f"Binance BTC trend also failed: {e}")
        
        return {"crypto_regime": "unknown", "trend_7d_pct": 0}
    
    async def should_allow_buy(self, asset_type: str = "stock") -> tuple[bool, str]:
        """
        Main check: Should we allow BUY signals?
        
        Args:
            asset_type: "stock" or "crypto"
        
        Returns:
            (allow: bool, reason: str)
        """
        if not REGIME_FILTER_ENABLED:
            return True, "Regime filter disabled"
        
        if asset_type == "crypto":
            btc_data = await self.get_btc_trend()
            if btc_data.get("crypto_regime") == "bear":
                return False, f"BTC down {btc_data.get('trend_7d_pct', 0):.1f}% - blocking crypto BUYs"
            return True, "Crypto regime: OK"
        
        else:  # stock
            spy_data = await self.get_spy_regime()
            if spy_data.get("regime") == "bear":
                return False, f"SPY below 20MA (${spy_data.get('spy_price', 0):.2f} < ${spy_data.get('spy_ma20', 0):.2f})"
            return True, "Stock regime: OK"


# ============================================================================
# 2. VIX GATE - "Are people scared?"
# ============================================================================

class VIXGate:
    """
    Adjusts BUY confidence based on VIX level.
    
    Logic:
    - VIX > 30: PANIC - Block ALL BUY signals (multiplier = 0)
    - VIX > 25: FEAR - Half confidence (multiplier = 0.5)
    - VIX > 20: CAUTION - Reduced confidence (multiplier = 0.75)
    - VIX <= 20: NORMAL - Full confidence (multiplier = 1.0)
    """
    
    def __init__(self):
        self.last_vix = None
        self.last_check = 0
        self.cache_ttl = 300  # 5 minutes
    
    async def get_current_vix(self) -> float:
        """Fetch current VIX level with multiple fallbacks."""
        try:
            # Check cache
            if self.last_vix and (time.time() - self.last_check) < self.cache_ttl:
                return self.last_vix
            
            # Try 1: Use world_context VIX (which has its own fallbacks)
            try:
                from core.world_context import get_world_context
                ctx = get_world_context()
                if ctx and ctx.get("vix"):
                    vix = ctx["vix"]
                    if vix and vix > 0:
                        self.last_vix = vix
                        self.last_check = time.time()
                        LOGGER.info(f"VIX from world_context: {vix:.1f}")
                        return vix
            except Exception as e:
                LOGGER.debug(f"world_context VIX failed: {e}")
            
            # Try 2: Fear & Greed as VIX proxy (Extreme Fear=30+, Fear=25-30, etc.)
            try:
                from core.pattern_intelligence.fear_greed import get_fear_greed_index
                fg = get_fear_greed_index()
                if fg and fg.get("value"):
                    fg_value = int(fg["value"])
                    # Convert Fear & Greed (0-100) to VIX-like (10-40)
                    # FG 0 (Extreme Fear) → VIX 35
                    # FG 25 (Fear) → VIX 25
                    # FG 50 (Neutral) → VIX 18
                    # FG 75 (Greed) → VIX 14
                    # FG 100 (Extreme Greed) → VIX 10
                    vix_estimate = 35 - (fg_value * 0.25)
                    self.last_vix = vix_estimate
                    self.last_check = time.time()
                    LOGGER.info(f"VIX estimated from Fear&Greed ({fg_value}): {vix_estimate:.1f}")
                    return vix_estimate
            except Exception as e:
                LOGGER.debug(f"Fear&Greed VIX proxy failed: {e}")
            
            # Try 3: Direct Polygon VIX
            import aiohttp
            polygon_key = os.getenv("POLYGON_API_KEY")
            
            if polygon_key:
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev?adjusted=true&apiKey={polygon_key}"
                    
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("results"):
                                vix = data["results"][0].get("c", 20)
                                self.last_vix = vix
                                self.last_check = time.time()
                                return vix
            
            # Default to neutral
            LOGGER.warning("All VIX sources failed, using default 20.0")
            return 20.0
            
        except Exception as e:
            LOGGER.error(f"VIX fetch failed: {e}")
            return 20.0  # Assume normal
    
    async def get_buy_confidence_multiplier(self) -> tuple[float, str]:
        """
        Get confidence multiplier for BUY signals based on VIX.
        
        Returns:
            (multiplier: 0.0-1.0, reason: str)
        """
        if not VIX_GATE_ENABLED:
            return 1.0, "VIX gate disabled"
        
        vix = await self.get_current_vix()
        
        if vix > VIX_PANIC_THRESHOLD:
            LOGGER.warning(f"🚨 VIX @ {vix:.1f} - PANIC MODE - Blocking BUYs")
            return 0.0, f"VIX {vix:.1f} > {VIX_PANIC_THRESHOLD} (PANIC)"
        
        elif vix > VIX_FEAR_THRESHOLD:
            LOGGER.warning(f"⚠️ VIX @ {vix:.1f} - FEAR - Halving BUY confidence")
            return 0.5, f"VIX {vix:.1f} > {VIX_FEAR_THRESHOLD} (FEAR)"
        
        elif vix > VIX_CAUTION_THRESHOLD:
            LOGGER.info(f"📊 VIX @ {vix:.1f} - CAUTION - Reducing BUY confidence")
            return 0.75, f"VIX {vix:.1f} > {VIX_CAUTION_THRESHOLD} (CAUTION)"
        
        else:
            LOGGER.debug(f"✅ VIX @ {vix:.1f} - Normal - Full confidence")
            return 1.0, f"VIX {vix:.1f} - Normal"


# ============================================================================
# 3. CONFIRMATION COUNTER - "Do multiple signals agree?"
# ============================================================================

class ConfirmationCounter:
    """
    Requires multiple bullish/bearish signals before issuing BUY/SELL.
    
    Confirmations for BUY:
    - RSI < 30 (oversold)
    - MACD crossover bullish
    - Price near support
    - SPY above 20MA
    - VIX < 20
    - No negative news in 6h
    
    Minimum 4 = HIGH confidence BUY
    Minimum 3 = LOW confidence BUY
    Less than 3 = NO signal
    """
    
    async def count_buy_confirmations(
        self,
        metrics: dict[str, Any],
        spy_above_ma: bool = True,
        vix_level: float = 20.0,
        has_negative_news: bool = False
    ) -> tuple[int, list[str]]:
        """
        Count bullish confirmations.
        
        Returns:
            (count: int, reasons: list[str])
        """
        confirmations = 0
        reasons = []
        
        # 1. RSI Oversold (< 30)
        rsi = metrics.get("rsi_14") or metrics.get("rsi") or 50
        if rsi is not None and rsi < 30:
            confirmations += 1
            reasons.append(f"RSI oversold ({rsi:.0f})")
        
        # 2. MACD Crossover bullish (histogram turning positive)
        macd_hist = metrics.get("macd_histogram") or 0
        macd_prev = metrics.get("macd_histogram_prev") or macd_hist
        if macd_hist is not None and macd_hist > 0 and macd_prev is not None and macd_prev <= 0:
            confirmations += 1
            reasons.append("MACD bullish crossover")
        elif macd_hist is not None and macd_hist > 0:
            confirmations += 0.5  # Partial credit
            reasons.append("MACD positive")
        
        # 3. Price near support (using Bollinger lower band)
        bb_lower = metrics.get("bb_lower") or 0
        price = metrics.get("current_price") or metrics.get("price") or 0
        if bb_lower > 0 and price > 0:
            distance_to_bb = (price - bb_lower) / price
            if distance_to_bb < 0.02:  # Within 2% of lower band
                confirmations += 1
                reasons.append("Price near support (BB lower)")
        
        # 4. SPY above 20MA (market regime)
        if spy_above_ma:
            confirmations += 1
            reasons.append("SPY above 20MA (bull market)")
        
        # 5. VIX low (< 20)
        if vix_level is not None and vix_level < VIX_CAUTION_THRESHOLD:
            confirmations += 1
            reasons.append(f"VIX low ({vix_level:.1f})")
        elif vix_level is not None and vix_level < VIX_FEAR_THRESHOLD:
            confirmations += 0.5
            reasons.append(f"VIX moderate ({vix_level:.1f})")
        
        # 6. No negative news
        if not has_negative_news:
            confirmations += 1
            reasons.append("No negative news")
        
        # 7. Momentum positive
        momentum = metrics.get("momentum_7d") or metrics.get("momentum") or 0
        if momentum is not None and momentum > 0.02:  # 2% positive momentum
            confirmations += 1
            reasons.append(f"Positive momentum ({momentum:.1%})")
        
        # 8. Volume confirmation
        volume_trend = metrics.get("volume_trend") or 1.0
        if volume_trend is not None and volume_trend > 1.2:  # 20% above average
            confirmations += 0.5
            reasons.append("High volume")
        
        return int(confirmations), reasons
    
    async def count_sell_confirmations(
        self,
        metrics: dict[str, Any],
        spy_above_ma: bool = True,
        vix_level: float = 20.0,
        has_negative_news: bool = False
    ) -> tuple[int, list[str]]:
        """
        Count bearish confirmations.
        
        Returns:
            (count: int, reasons: list[str])
        """
        confirmations = 0
        reasons = []
        
        # 1. RSI Overbought (> 70)
        rsi = metrics.get("rsi_14") or metrics.get("rsi") or 50
        if rsi is not None and rsi > 70:
            confirmations += 1
            reasons.append(f"RSI overbought ({rsi:.0f})")
        
        # 2. MACD Crossover bearish
        macd_hist = metrics.get("macd_histogram") or 0
        if macd_hist is not None and macd_hist < 0:
            confirmations += 1
            reasons.append("MACD bearish")
        
        # 3. Price near resistance (BB upper band)
        bb_upper = metrics.get("bb_upper") or 0
        price = metrics.get("current_price") or metrics.get("price") or 0
        if bb_upper > 0 and price > 0:
            distance_to_bb = (bb_upper - price) / price
            if distance_to_bb < 0.02:
                confirmations += 1
                reasons.append("Price near resistance (BB upper)")
        
        # 4. SPY below 20MA (bear market)
        if not spy_above_ma:
            confirmations += 1
            reasons.append("SPY below 20MA (bear market)")
        
        # 5. VIX high (> 25) - fear = downside likely
        if vix_level is not None and vix_level > VIX_FEAR_THRESHOLD:
            confirmations += 1
            reasons.append(f"VIX high ({vix_level:.1f})")
        
        # 6. Negative news
        if has_negative_news:
            confirmations += 1
            reasons.append("Negative news detected")
        
        # 7. Momentum negative
        momentum = metrics.get("momentum_7d") or metrics.get("momentum") or 0
        if momentum is not None and momentum < -0.02:
            confirmations += 1
            reasons.append(f"Negative momentum ({momentum:.1%})")
        
        return int(confirmations), reasons
    
    def get_signal_quality(self, confirmations: int) -> str:
        """Convert confirmation count to signal quality."""
        if confirmations is None:
            return "SKIP"
        if confirmations >= MIN_CONFIRMATIONS_HIGH:
            return "HIGH"
        elif confirmations >= MIN_CONFIRMATIONS_LOW:
            return "LOW"
        else:
            return "SKIP"


# ============================================================================
# 4. LOSER ANALYSIS - Query patterns in failed trades
# ============================================================================

class LoserAnalyzer:
    """
    Analyzes patterns in losing BUY trades to find what's going wrong.
    """
    
    def __init__(self, db_path: str = "data/ghost_predictions.db"):
        self.db_path = db_path
    
    def get_losing_buys(self, limit: int = 50) -> list[dict]:
        """
        Query losing BUY trades from paper_trades_v2.
        
        Returns list of:
            {symbol, confidence, entry_price, pnl_pct, created_at}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Query losing BUY trades
            c.execute("""
                SELECT 
                    symbol,
                    confidence,
                    entry_price,
                    pnl_pct,
                    created_at,
                    direction
                FROM paper_trades_v2
                WHERE direction IN ('UP', 'LONG', 'BUY')
                  AND outcome = 'LOSS'
                ORDER BY pnl_pct ASC
                LIMIT ?
            """, (limit,))
            
            rows = c.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            LOGGER.error(f"Failed to query losing trades: {e}")
            return []
    
    def analyze_patterns(self) -> dict[str, Any]:
        """
        Analyze patterns in losing BUY trades.
        
        Returns:
            {
                "worst_symbols": [...],
                "avg_confidence": 0.75,
                "time_patterns": {...},
                "recommendations": [...]
            }
        """
        losers = self.get_losing_buys(100)
        
        if not losers:
            return {"error": "No losing trades found", "recommendations": []}
        
        # Analyze by symbol
        symbol_losses = {}
        for trade in losers:
            sym = trade["symbol"]
            if sym not in symbol_losses:
                symbol_losses[sym] = []
            symbol_losses[sym].append(trade["pnl_pct"])
        
        # Sort by total loss
        worst_symbols = sorted(
            [(sym, len(losses), sum(losses)) for sym, losses in symbol_losses.items()],
            key=lambda x: x[2]  # Sort by total loss
        )[:10]
        
        # Average confidence of losers
        confidences = [t["confidence"] for t in losers if t["confidence"]]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        # Time patterns (hour of day)
        hour_losses = {}
        for trade in losers:
            try:
                ts = trade["created_at"]
                if isinstance(ts, (int, float)):
                    hour = datetime.fromtimestamp(ts).hour
                else:
                    hour = datetime.fromisoformat(str(ts)).hour
                hour_losses[hour] = hour_losses.get(hour, 0) + 1
            except Exception:
                pass
        
        # Build recommendations
        recommendations = []
        
        if avg_conf > 0.7:
            recommendations.append(
                "High confidence BUYs are failing - confidence may be miscalibrated"
            )
        
        for sym, count, total_loss in worst_symbols[:3]:
            recommendations.append(
                f"Consider blacklisting {sym}: {count} losses totaling {total_loss:.1f}%"
            )
        
        if hour_losses:
            worst_hour = max(hour_losses, key=hour_losses.get)
            recommendations.append(
                f"Most losses at hour {worst_hour}:00 - consider time filter"
            )
        
        return {
            "total_losing_buys": len(losers),
            "worst_symbols": worst_symbols,
            "avg_confidence": round(avg_conf, 2),
            "time_patterns": hour_losses,
            "recommendations": recommendations
        }


# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================

async def apply_market_gates(
    direction: str,
    confidence: float,
    metrics: dict[str, Any],
    asset_type: str = "crypto",
    symbol: str = "UNKNOWN"
) -> tuple[str, float, dict[str, Any]]:
    """
    Apply all market gates to a prediction.
    
    This is the main function to call before finalizing a prediction.
    
    Args:
        direction: "UP", "DOWN", or "FLAT"
        confidence: Original confidence (0-1)
        metrics: Technical indicators dict
        asset_type: "stock" or "crypto"
        symbol: Trading symbol for logging
    
    Returns:
        (final_direction, final_confidence, gate_info)
    """
    gate_info = {
        "regime_filter": {"applied": False},
        "vix_gate": {"applied": False},
        "confirmations": {"count": 0, "reasons": []},
        "original_confidence": confidence,
        "gates_passed": True,
        "symbol": symbol
    }
    
    # Initialize gates
    regime_filter = RegimeFilter()
    vix_gate = VIXGate()
    confirmation_counter = ConfirmationCounter()
    
    # ========================================
    # GATE 1: Regime Filter (SYMMETRIC - checks both UP and DOWN)
    # ========================================
    if direction == "UP":
        allow_buy, regime_reason = await regime_filter.should_allow_buy(asset_type)
        gate_info["regime_filter"] = {
            "applied": True,
            "allowed": allow_buy,
            "reason": regime_reason
        }
        
        if not allow_buy:
            LOGGER.warning(f"🚫 REGIME FILTER: Penalizing BUY for {symbol} - {regime_reason}")
            gate_info["gates_passed"] = False
            # Don't flatten — reduce confidence heavily instead
            confidence = confidence * 0.60  # 40% penalty for buying in bear market
    elif direction == "DOWN":
        # Symmetric: also check regime for SELL signals
        # In strong uptrends (SPY > 20MA, BTC trending up), penalize DOWN signals
        allow_buy, regime_reason = await regime_filter.should_allow_buy(asset_type)
        gate_info["regime_filter"] = {
            "applied": True,
            "allowed": True,  # Don't hard-block DOWN, but reduce confidence
            "reason": f"SELL in {'bullish' if allow_buy else 'bearish'} regime"
        }
        if allow_buy:
            # Regime is bullish (would allow buys) — selling into strength is risky
            old_conf = confidence
            confidence = confidence * 0.85  # 15% penalty for selling in uptrend
            LOGGER.info(f"[REGIME-GATE] {symbol} SELL penalized in bullish regime ({old_conf:.0%} → {confidence:.0%})")
    
    # ========================================
    # GATE 2: VIX Gate (SYMMETRIC - affects both UP and DOWN)
    # ========================================
    vix_multiplier, vix_reason = await vix_gate.get_buy_confidence_multiplier()
    vix_level = vix_gate.last_vix or 20.0
    
    gate_info["vix_gate"] = {
        "applied": True,
        "multiplier": vix_multiplier,
        "reason": vix_reason,
        "vix_level": vix_level
    }
    
    if direction == "UP":
        if vix_multiplier == 0:
            LOGGER.warning(f"🚫 VIX GATE: Heavy penalty for BUY {symbol} - VIX at {vix_level:.1f} (PANIC)")
            gate_info["gates_passed"] = False
            # Don't flatten — reduce confidence heavily instead
            confidence = confidence * 0.40  # 60% penalty for buying in panic
        elif vix_multiplier < 1.0:
            old_conf = confidence
            confidence = confidence * vix_multiplier
            LOGGER.info(f"[VIX-GATE] Reduced UP confidence for {symbol} - VIX {vix_level:.1f} ({old_conf:.0%} → {confidence:.0%})")
    elif direction == "DOWN":
        # Symmetric: LOW VIX (calm) should penalize SELL signals
        # If VIX is calm (< 15), market is complacent — shorting calm markets is risky
        if vix_level < 15:
            old_conf = confidence
            confidence = confidence * 0.85
            LOGGER.info(f"[VIX-GATE] Reduced DOWN confidence for {symbol} - VIX calm at {vix_level:.1f} ({old_conf:.0%} → {confidence:.0%})")
        # High VIX should also moderate DOWN (don't short panic bottoms)
        elif vix_level > 35:
            old_conf = confidence
            confidence = confidence * 0.90
            LOGGER.info(f"[VIX-GATE] Reduced DOWN confidence for {symbol} - VIX extreme at {vix_level:.1f} ({old_conf:.0%} → {confidence:.0%})")
    
    LOGGER.info(f"📊 VIX GATE: {symbol} {direction} confidence at {confidence:.0%} ({vix_reason})")
    
    # ========================================
    # GATE 3: Confirmation Counter
    # ========================================
    try:
        spy_data = await regime_filter.get_spy_regime()
        spy_above_ma = spy_data.get("above_20ma", True)
        vix_level = vix_gate.last_vix or 20.0
    except Exception:
        spy_above_ma = True
        vix_level = 20.0
    
    # Check for negative news
    has_negative_news = False
    try:
        from core.news_sentiment import fetch_news_sentiment
        news_data = fetch_news_sentiment("BTC", limit=3)
        if news_data.get("ok") and news_data.get("sentiment_score", 0) < -0.3:
            has_negative_news = True
    except Exception:
        pass
    
    if direction == "UP":
        conf_count, conf_reasons = await confirmation_counter.count_buy_confirmations(
            metrics, spy_above_ma, vix_level, has_negative_news
        )
        signal_quality = confirmation_counter.get_signal_quality(conf_count)
        
        gate_info["confirmations"] = {
            "count": conf_count,
            "reasons": conf_reasons,
            "quality": signal_quality
        }
        
        if signal_quality == "SKIP":
            LOGGER.warning(f"⚠️ CONFIRMATIONS: Only {conf_count} for {symbol} - low quality BUY signal")
            gate_info["gates_passed"] = False
            # Don't flatten — penalize confidence instead
            confidence = confidence * 0.65  # 35% penalty for unconfirmed signal
        elif signal_quality == "LOW":
            old_conf = confidence
            confidence = confidence * 0.8
            LOGGER.info(f"[CONFIRM-GATE] {symbol} BUY LOW quality ({conf_count} confirmations) ({old_conf:.0%} → {confidence:.0%})")
        else:
            LOGGER.info(f"[CONFIRM-GATE] {symbol} BUY HIGH quality ({conf_count} confirmations)")
    
    elif direction == "DOWN":
        # SYMMETRIC: DOWN signals also need confirmation quality checks
        conf_count, conf_reasons = await confirmation_counter.count_sell_confirmations(
            metrics, spy_above_ma, vix_level, has_negative_news
        )
        signal_quality = confirmation_counter.get_signal_quality(conf_count)
        
        gate_info["confirmations"] = {
            "count": conf_count,
            "reasons": conf_reasons,
            "quality": signal_quality
        }
        
        if signal_quality == "SKIP":
            LOGGER.warning(f"⚠️ CONFIRMATIONS: Only {conf_count} for {symbol} - low quality SELL signal")
            gate_info["gates_passed"] = False
            # Don't flatten — penalize confidence instead
            confidence = confidence * 0.65  # 35% penalty for unconfirmed signal
        elif signal_quality == "LOW":
            old_conf = confidence
            confidence = confidence * 0.8
            LOGGER.info(f"[CONFIRM-GATE] {symbol} SELL LOW quality ({conf_count} confirmations) ({old_conf:.0%} → {confidence:.0%})")
        else:
            LOGGER.info(f"[CONFIRM-GATE] {symbol} SELL HIGH quality ({conf_count} confirmations)")
    
    # ========================================
    # FINAL OUTPUT
    # ========================================
    gate_info["final_confidence"] = confidence
    
    if gate_info["gates_passed"]:
        LOGGER.info(
            f"✅ [GATES-SUMMARY] {symbol} PASSED all gates: {direction} @ {confidence:.0%} "
            f"(Confirmations: {gate_info.get('confirmations', {}).get('count', '?')})"
        )
    else:
        LOGGER.info(
            f"⚠️ [GATES-SUMMARY] {symbol} PENALIZED: {direction} {gate_info['original_confidence']:.0%} → {confidence:.0%} "
            f"(Confirmations: {gate_info.get('confirmations', {}).get('count', '?')})"
        )
    
    return direction, confidence, gate_info


# ============================================================================
# API ENDPOINTS (To be registered in wolf_app.py)
# ============================================================================

def get_market_gates_status() -> dict[str, Any]:
    """Get current status of all market gates."""
    return {
        "regime_filter": {
            "enabled": REGIME_FILTER_ENABLED,
            "spy_ma_period": SPY_MA_PERIOD,
            "btc_trend_days": BTC_TREND_DAYS,
            "btc_trend_threshold": BTC_TREND_THRESHOLD
        },
        "vix_gate": {
            "enabled": VIX_GATE_ENABLED,
            "panic_threshold": VIX_PANIC_THRESHOLD,
            "fear_threshold": VIX_FEAR_THRESHOLD,
            "caution_threshold": VIX_CAUTION_THRESHOLD
        },
        "confirmations": {
            "min_high": MIN_CONFIRMATIONS_HIGH,
            "min_low": MIN_CONFIRMATIONS_LOW
        }
    }


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_gates():
        print("Testing Market Gates...")
        
        # Test regime filter
        rf = RegimeFilter()
        spy = await rf.get_spy_regime()
        print(f"SPY Regime: {spy}")
        
        btc = await rf.get_btc_trend()
        print(f"BTC Trend: {btc}")
        
        # Test VIX gate
        vg = VIXGate()
        vix = await vg.get_current_vix()
        mult, reason = await vg.get_buy_confidence_multiplier()
        print(f"VIX: {vix}, Multiplier: {mult}, Reason: {reason}")
        
        # Test full pipeline
        metrics = {"rsi": 25, "momentum_7d": 0.05, "macd_histogram": 0.1}
        direction, conf, info = await apply_market_gates("UP", 0.75, metrics, "crypto")
        print(f"Final: {direction} @ {conf:.0%}")
        print(f"Gate Info: {info}")
    
    asyncio.run(test_gates())

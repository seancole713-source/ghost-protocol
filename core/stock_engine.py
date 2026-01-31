#!/usr/bin/env python3
"""
🏛️ GHOST STOCK ENGINE - Stock-Specific Prediction Model
========================================================

Why stocks need a different model than crypto:
- Stocks move 2-3x slower than crypto
- Stocks are affected by market hours, earnings, Fed
- RSI 30 (crypto oversold) rarely happens in stocks
- 48h horizon is too long for stocks (24h better)
- 6% target is unrealistic (2% is achievable)

This engine applies the Ghost blueprint that WORKS for crypto,
but with stock-tuned parameters:

CRYPTO ENGINE          | STOCK ENGINE
-----------------------|--------------------
48h horizon            | 24h horizon
6% target              | 2% target
RSI 30/70              | RSI 35/65
3 confirmations        | 4 confirmations
VIX < 25               | VIX < 20
BTC trend gate         | SPY regime gate
24/7 trading           | Market hours only
No earnings            | Earnings blackout

Target: 40-50% win rate (up from 4.5%)
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

# Pattern tracking for accuracy measurement
from core.pattern_tracker import record_pattern_detection

# Internal imports - lazy loaded to avoid circular imports
_wolf_app_loaded = False

LOGGER = logging.getLogger("ghost.stock_engine")

# ============================================================================
# STOCK ENGINE CONFIGURATION
# ============================================================================

@dataclass
class StockConfig:
    """Stock-specific prediction parameters (tuned for slower-moving assets)"""
    
    # Prediction horizon (vs 48h for crypto)
    horizon_hours: int = 24
    
    # Target move percentage (vs 6% for crypto)
    target_pct: float = 2.0
    
    # RSI thresholds (RELAXED from 35/65 to 40/60 for better signal generation)
    # Stocks rarely hit 35/65 - we were getting too many HOLDs
    rsi_oversold: float = 40.0
    rsi_overbought: float = 60.0
    
    # Confirmation requirements (reduced from 4 to 3 to allow more predictions)
    min_confirmations: int = 3
    
    # VIX threshold (relaxed from 20 to 22 - 20 was too strict)
    vix_max: float = 22.0
    
    # SPY regime requirement
    require_spy_bull: bool = True
    spy_ma_period: int = 20
    
    # Market hours
    market_hours_only: bool = True
    
    # Earnings blackout
    earnings_blackout_days: int = 7
    
    # Position sizing
    max_position_pct: float = 5.0  # Max 5% of portfolio per stock
    
    # Stop loss / Take profit
    stop_loss_pct: float = 1.0  # Tighter than crypto
    take_profit_pct: float = 2.5  # Smaller targets
    
    # Multi-timeframe requirements
    require_mtf_alignment: bool = True
    mtf_min_agree: int = 2  # At least 2 of 3 timeframes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_hours": self.horizon_hours,
            "target_pct": self.target_pct,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "min_confirmations": self.min_confirmations,
            "vix_max": self.vix_max,
            "require_spy_bull": self.require_spy_bull,
            "market_hours_only": self.market_hours_only,
            "earnings_blackout_days": self.earnings_blackout_days,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "require_mtf_alignment": self.require_mtf_alignment,
        }


# Default configuration
STOCK_CONFIG = StockConfig()

# Stock whitelist (high-liquidity, predictable stocks)
STOCK_WHITELIST = {
    # Tech giants (most predictable)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    
    # Finance
    "JPM", "BAC", "GS", "MS",
    
    # Consumer
    "TSLA", "DIS", "NKE", "SBUX",
    
    # Healthcare
    "JNJ", "PFE", "UNH",
    
    # Energy
    "XOM", "CVX",
    
    # Industrial
    "CAT", "BA", "GE",
}


@dataclass
class StockPrediction:
    """Stock prediction result"""
    symbol: str
    direction: str  # "UP", "DOWN", "HOLD"
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    horizon_hours: int
    confirmations: int
    gates_passed: List[str]
    gates_failed: List[str]
    reasons: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_actionable(self) -> bool:
        """True if prediction should generate a signal"""
        return (
            self.direction in ("UP", "DOWN") and
            self.confidence >= 0.6 and
            len(self.gates_failed) == 0 and
            self.confirmations >= STOCK_CONFIG.min_confirmations
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "entry_price": round(self.entry_price, 2),
            "target_price": round(self.target_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "horizon_hours": self.horizon_hours,
            "confirmations": self.confirmations,
            "min_confirmations": STOCK_CONFIG.min_confirmations,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "reasons": self.reasons,
            "is_actionable": self.is_actionable,
            "timestamp": self.timestamp.isoformat(),
        }


class StockEngine:
    """
    Stock-specific prediction engine.
    
    Uses the same ensemble approach as crypto (LSTM + XGBoost + Transformer)
    but with stock-tuned parameters and additional gates.
    """
    
    def __init__(self, config: StockConfig = None):
        self.config = config or STOCK_CONFIG
        self._initialized = False
        self._vix_cache: Tuple[float, float] = (0, 0)  # (value, timestamp)
        self._spy_cache: Tuple[Dict, float] = ({}, 0)  # (data, timestamp)
    
    async def initialize(self):
        """Initialize the engine (load models, etc.)"""
        if self._initialized:
            return
        
        LOGGER.info("🏛️ Initializing Stock Engine...")
        
        # Import gate modules
        try:
            from core.stock_gates import run_all_stock_gates
            from core.sector_momentum import sector_momentum_gate, analyze_sector_momentum
            from core.economic_calendar import economic_calendar_gate
            self._gates_available = True
            LOGGER.info("✅ Stock gates loaded")
        except ImportError as e:
            LOGGER.warning(f"⚠️ Stock gates not fully available: {e}")
            self._gates_available = False
        
        self._initialized = True
        LOGGER.info("🏛️ Stock Engine initialized with config:")
        LOGGER.info(f"   Horizon: {self.config.horizon_hours}h")
        LOGGER.info(f"   Target: {self.config.target_pct}%")
        LOGGER.info(f"   RSI: {self.config.rsi_oversold}/{self.config.rsi_overbought}")
        LOGGER.info(f"   Min confirmations: {self.config.min_confirmations}")
    
    def _load_wolf_app(self):
        """Lazy load wolf_app to avoid circular imports"""
        global _wolf_app_loaded
        if not _wolf_app_loaded:
            try:
                import wolf_app
                self._wolf_app = wolf_app
                _wolf_app_loaded = True
            except ImportError as e:
                LOGGER.error(f"Failed to import wolf_app: {e}")
                self._wolf_app = None
    
    async def _get_vix(self) -> Optional[float]:
        """Get current VIX level (cached for 5 min)"""
        now = time.time()
        if now - self._vix_cache[1] < 300:  # 5 min cache
            return self._vix_cache[0]
        
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                value = hist['Close'].iloc[-1]
                self._vix_cache = (value, now)
                return value
        except Exception as e:
            LOGGER.warning(f"VIX fetch failed: {e}")
        
        return self._vix_cache[0] if self._vix_cache[0] > 0 else None
    
    async def _get_spy_regime(self) -> Tuple[bool, float]:
        """
        Check if SPY is above 20-day MA (bull market).
        
        Returns: (is_bullish, pct_vs_ma)
        """
        now = time.time()
        if now - self._spy_cache[1] < 300:  # 5 min cache
            data = self._spy_cache[0]
            return data.get("bullish", True), data.get("pct_vs_ma", 0)
        
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="30d")
            
            if len(hist) >= 20:
                current = hist['Close'].iloc[-1]
                ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                pct_vs_ma = ((current - ma20) / ma20) * 100
                bullish = current > ma20
                
                self._spy_cache = ({"bullish": bullish, "pct_vs_ma": pct_vs_ma}, now)
                return bullish, pct_vs_ma
        except Exception as e:
            LOGGER.warning(f"SPY regime check failed: {e}")
        
        return True, 0  # Default to bullish if can't check
    
    async def _get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Get technical indicators for stock.
        Uses Polygon API with yfinance fallback.
        """
        self._load_wolf_app()
        
        indicators = {
            "rsi_14": None,
            "macd_histogram": None,
            "bb_lower": None,
            "bb_upper": None,
            "ema_20": None,
            "current_price": None,
            "volume_ratio": None,
        }
        
        try:
            import pandas as pd
            import os
            import httpx
            from datetime import datetime, timedelta
            
            hist = None
            
            # Try Polygon first (more reliable than yfinance)
            polygon_key = os.getenv("POLYGON_API_KEY")
            if polygon_key:
                try:
                    end = datetime.now().strftime("%Y-%m-%d")
                    start = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&apiKey={polygon_key}"
                    
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(url)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("results"):
                                results = data["results"]
                                hist = pd.DataFrame(results)
                                hist['Close'] = hist['c']
                                hist['Volume'] = hist['v']
                                LOGGER.info(f"[STOCK-ENGINE] Got {len(hist)} bars from Polygon for {symbol}")
                except Exception as e:
                    LOGGER.warning(f"[STOCK-ENGINE] Polygon failed for {symbol}: {e}")
            
            # Fallback to yfinance
            if hist is None or hist.empty:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="30d")
                    if not hist.empty:
                        LOGGER.info(f"[STOCK-ENGINE] Got {len(hist)} bars from yfinance for {symbol}")
                except Exception as e:
                    LOGGER.warning(f"[STOCK-ENGINE] yfinance failed for {symbol}: {e}")
            
            if hist is None or hist.empty:
                return indicators
            
            close = hist['Close']
            volume = hist['Volume']
            
            # Current price
            indicators["current_price"] = float(close.iloc[-1])
            
            # RSI 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators["rsi_14"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            # EMA 12, 26 for MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            indicators["macd_histogram"] = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0
            
            # EMA 20
            ema20 = close.ewm(span=20).mean()
            indicators["ema_20"] = float(ema20.iloc[-1])
            
            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            indicators["bb_upper"] = float(bb_upper.iloc[-1])
            indicators["bb_lower"] = float(bb_lower.iloc[-1])
            
            # Volume ratio (current vs 20d avg)
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            indicators["volume_ratio"] = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
            
        except Exception as e:
            LOGGER.warning(f"Technical indicators failed for {symbol}: {e}")
        
        return indicators
    
    async def _count_confirmations(
        self,
        symbol: str,
        direction: str,
        indicators: Dict[str, Any],
        vix: float,
        spy_bullish: bool
    ) -> Tuple[int, List[str]]:
        """
        Count bullish/bearish confirmations using stock-tuned thresholds.
        """
        confirmations = 0
        reasons = []
        
        price = indicators.get("current_price", 0)
        rsi = indicators.get("rsi_14", 50)
        macd = indicators.get("macd_histogram", 0)
        bb_lower = indicators.get("bb_lower", 0)
        bb_upper = indicators.get("bb_upper", 0)
        ema20 = indicators.get("ema_20", 0)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        
        if direction == "UP":
            # 1. RSI oversold (< 35 for stocks)
            if rsi and rsi < self.config.rsi_oversold:
                confirmations += 1
                reasons.append(f"RSI oversold ({rsi:.0f} < {self.config.rsi_oversold})")
            
            # 2. MACD bullish
            if macd and macd > 0:
                confirmations += 1
                reasons.append("MACD bullish")
            
            # 3. Price near support (BB lower)
            if bb_lower and price and bb_lower > 0:
                distance = (price - bb_lower) / price
                if distance < 0.02:
                    confirmations += 1
                    reasons.append("Price near support")
            
            # 4. SPY bullish (required for stocks)
            if spy_bullish:
                confirmations += 1
                reasons.append("SPY above 20MA (bull market)")
            
            # 5. Low VIX
            if vix and vix < self.config.vix_max:
                confirmations += 1
                reasons.append(f"VIX low ({vix:.1f})")
            
            # 6. Volume spike
            if volume_ratio and volume_ratio > 1.3:
                confirmations += 1
                reasons.append(f"Volume spike ({volume_ratio:.1f}x)")
            
            # 7. Price above EMA20
            if price and ema20 and price > ema20:
                confirmations += 1
                reasons.append("Price above EMA20")
        
        elif direction == "DOWN":
            # 1. RSI overbought (> 65 for stocks)
            if rsi and rsi > self.config.rsi_overbought:
                confirmations += 1
                reasons.append(f"RSI overbought ({rsi:.0f} > {self.config.rsi_overbought})")
            
            # 2. MACD bearish
            if macd and macd < 0:
                confirmations += 1
                reasons.append("MACD bearish")
            
            # 3. Price near resistance (BB upper)
            if bb_upper and price and bb_upper > 0:
                distance = (bb_upper - price) / price
                if distance < 0.02:
                    confirmations += 1
                    reasons.append("Price near resistance")
            
            # 4. SPY bearish
            if not spy_bullish:
                confirmations += 1
                reasons.append("SPY below 20MA (bear market)")
            
            # 5. High VIX
            if vix and vix > 25:
                confirmations += 1
                reasons.append(f"VIX elevated ({vix:.1f})")
        
        return confirmations, reasons
    
    async def predict(self, symbol: str, bypass_calendar: bool = False) -> StockPrediction:
        """
        Generate stock prediction using stock-tuned model.
        
        This is the main entry point for stock predictions.
        
        Args:
            symbol: Stock ticker
            bypass_calendar: If True, skip FOMC/CPI/NFP blackout gates (for testing only)
        """
        if not self._initialized:
            await self.initialize()
        
        LOGGER.info(f"🏛️ Stock Engine predicting {symbol}{'[BYPASS CALENDAR]' if bypass_calendar else ''}...")
        
        gates_passed = []
        gates_failed = []
        all_reasons = []
        
        # Step 1: Economic Calendar Gate (FOMC, CPI, NFP, Earnings)
        if not bypass_calendar:
            try:
                from core.economic_calendar import economic_calendar_gate
                allow, reason = economic_calendar_gate(symbol)
                if not allow:
                    gates_failed.append(f"EconomicCalendar: {reason}")
                    return StockPrediction(
                        symbol=symbol,
                        direction="HOLD",
                        confidence=0.0,
                        entry_price=0,
                        target_price=0,
                        stop_loss=0,
                        horizon_hours=self.config.horizon_hours,
                        confirmations=0,
                        gates_passed=gates_passed,
                        gates_failed=gates_failed,
                        reasons=[f"BLOCKED: {reason}"]
                    )
                gates_passed.append("EconomicCalendar")
            except ImportError:
                LOGGER.warning("Economic calendar gate not available")
        else:
            gates_passed.append("EconomicCalendar (BYPASSED for testing)")
        
        # Step 2: Get VIX
        vix = await self._get_vix()
        if vix and vix > self.config.vix_max + 5:  # Hard block at VIX > 25
            gates_failed.append(f"VIX: {vix:.1f} > {self.config.vix_max + 5}")
            return StockPrediction(
                symbol=symbol,
                direction="HOLD",
                confidence=0.0,
                entry_price=0,
                target_price=0,
                stop_loss=0,
                horizon_hours=self.config.horizon_hours,
                confirmations=0,
                gates_passed=gates_passed,
                gates_failed=gates_failed,
                reasons=[f"VIX too high ({vix:.1f})"]
            )
        if vix and vix < self.config.vix_max:
            gates_passed.append(f"VIX ({vix:.1f})")
        
        # Step 3: SPY Regime Gate
        spy_bullish, spy_pct = await self._get_spy_regime()
        if self.config.require_spy_bull and not spy_bullish and spy_pct < -2:
            gates_failed.append(f"SPYRegime: {spy_pct:+.1f}% vs MA")
            return StockPrediction(
                symbol=symbol,
                direction="HOLD",
                confidence=0.0,
                entry_price=0,
                target_price=0,
                stop_loss=0,
                horizon_hours=self.config.horizon_hours,
                confirmations=0,
                gates_passed=gates_passed,
                gates_failed=gates_failed,
                reasons=[f"Bear market (SPY {spy_pct:+.1f}% vs 20MA)"]
            )
        if spy_bullish:
            gates_passed.append(f"SPYRegime ({spy_pct:+.1f}%)")
        
        # Step 4: Sector Momentum Gate
        try:
            from core.sector_momentum import sector_momentum_gate
            sector_allowed, sector_reason, sector_modifier = sector_momentum_gate(symbol, "UP")
            if not sector_allowed:
                gates_failed.append(f"SectorMomentum: {sector_reason}")
            else:
                gates_passed.append(f"SectorMomentum: {sector_reason}")
                all_reasons.append(sector_reason)
        except ImportError:
            sector_modifier = 1.0
            LOGGER.warning("Sector momentum gate not available")
        
        # Step 5: Get Technical Indicators
        indicators = await self._get_technical_indicators(symbol)
        price = indicators.get("current_price", 0)
        
        if not price or price <= 0:
            return StockPrediction(
                symbol=symbol,
                direction="HOLD",
                confidence=0.0,
                entry_price=0,
                target_price=0,
                stop_loss=0,
                horizon_hours=self.config.horizon_hours,
                confirmations=0,
                gates_passed=gates_passed,
                gates_failed=["Price unavailable"],
                reasons=["Could not get current price"]
            )
        
        # Step 6: Determine Direction using ENSEMBLE + Technical Indicators
        rsi = indicators.get("rsi_14", 50)
        macd = indicators.get("macd_histogram", 0)
        price = indicators.get("current_price", 0)
        
        # PRIMARY: Use Ensemble Predictor (LSTM + XGBoost + Transformer)
        # This was missing - stocks need the same ML power as crypto
        direction = "HOLD"
        ensemble_confidence = 0.5
        try:
            from core.ensemble_predictor import get_ensemble_predictor
            from core.data_pillars.feature_orchestrator import get_feature_orchestrator
            
            # Get full feature set for ensemble
            orchestrator = get_feature_orchestrator()
            feature_data = orchestrator.get_all_features(symbol, period=90)
            features = feature_data.get("features", {})
            
            ensemble = get_ensemble_predictor()
            ensemble_result = ensemble.predict(features, method="confidence_weighted", symbol=symbol)
            
            if ensemble_result.confidence > 0.45:
                direction = ensemble_result.direction
                ensemble_confidence = ensemble_result.confidence
                all_reasons.append(f"Ensemble: {direction} ({ensemble_confidence:.0%})")
                LOGGER.info(f"🏛️ [{symbol}] Ensemble: {direction} ({ensemble_confidence:.0%})")
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Ensemble predictor failed: {e}")
        
        # FALLBACK: Technical indicators if ensemble is uncertain
        if direction == "HOLD" or ensemble_confidence < 0.55:
            # Use more lenient RSI thresholds for stocks (40/60 instead of 35/65)
            if rsi < 40 and macd > 0:
                direction = "UP"
                all_reasons.append(f"RSI oversold ({rsi:.0f}) + MACD bullish")
            elif rsi > 60 and macd < 0:
                direction = "DOWN"
                all_reasons.append(f"RSI overbought ({rsi:.0f}) + MACD bearish")
            elif rsi < 50 and macd > 0 and spy_bullish:
                direction = "UP"
                all_reasons.append("MACD bullish + SPY bull regime")
            elif rsi > 50 and macd < 0 and not spy_bullish:
                direction = "DOWN"
                all_reasons.append("MACD bearish + SPY bear regime")
        
        # Step 7: Apply Ghost Intel Rules (CRITICAL FIX - was missing!)
        intel_boost = 0.0
        try:
            from ghost_intel.integration import apply_intel_to_prediction
            
            # Apply Intel rules to stock prediction
            intel_direction, intel_confidence, intel_meta = apply_intel_to_prediction(
                symbol=symbol,
                direction=direction,
                confidence=ensemble_confidence
            )
            
            if intel_meta.get("intel_applied"):
                intel_boost = intel_meta.get("confidence_adjustment", 0)
                signals = intel_meta.get("signal_sources", [])
                
                # Intel can override direction in strong cases
                if intel_meta.get("direction_override") and intel_direction != direction:
                    direction = intel_direction
                    all_reasons.append(f"Intel override: {direction}")
                
                if signals:
                    all_reasons.extend([f"Intel: {s}" for s in signals[:3]])
                
                LOGGER.info(f"🏛️ [{symbol}] Intel applied: boost={intel_boost:+.0%}, signals={signals[:3]}")
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Intel integration failed: {e}")
        
        # Step 8: Count Confirmations
        confirmations, confirmation_reasons = await self._count_confirmations(
            symbol, direction, indicators, vix or 20, spy_bullish
        )
        all_reasons.extend(confirmation_reasons)
        
        # Step 9: Multi-Timeframe Check
        try:
            from core.stock_gates import StockConfirmationCounter
            # MTF adds to confirmations
            mtf_confirms = 0  # Would come from multi_timeframe module
            confirmations += mtf_confirms
        except ImportError:
            pass
        
        # Step 10: Calculate Confidence (incorporating ensemble + Intel)
        base_confidence = max(0.5, ensemble_confidence)  # Start from ensemble
        
        # Boost for confirmations
        conf_boost = min(0.25, confirmations * 0.04)
        
        # Intel boost (already calculated)
        intel_adj = intel_boost
        
        # Boost/penalty for sector
        sector_adj = (sector_modifier - 1.0) * 0.1 if 'sector_modifier' in dir() else 0
        
        # Penalty for high VIX
        vix_penalty = max(0, (vix - 15) * 0.01) if vix else 0
        
        confidence = base_confidence + conf_boost + intel_adj + sector_adj - vix_penalty
        # HARD CAP at 85% - we only have 52% win rate, no justification for higher
        confidence = max(0.1, min(0.85, confidence))
        
        # Step 10: Calculate Entry/Exit
        if direction == "UP":
            entry_price = price
            target_price = price * (1 + self.config.target_pct / 100)
            stop_loss = price * (1 - self.config.stop_loss_pct / 100)
        elif direction == "DOWN":
            entry_price = price
            target_price = price * (1 - self.config.target_pct / 100)
            stop_loss = price * (1 + self.config.stop_loss_pct / 100)
        else:
            entry_price = price
            target_price = price
            stop_loss = price
        
        # Final prediction
        prediction = StockPrediction(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            horizon_hours=self.config.horizon_hours,
            confirmations=confirmations,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            reasons=all_reasons[:5]  # Top 5 reasons
        )
        
        LOGGER.info(f"🏛️ {symbol} → {direction} ({confidence:.0%}) | {confirmations} confirmations")
        
        # Record pattern for accuracy tracking (only actionable predictions)
        if direction in ("UP", "DOWN") and confidence >= 0.6:
            try:
                record_pattern_detection(
                    pattern_type=f"stock_{direction.lower()}",
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    confidence=confidence
                )
            except Exception as e:
                LOGGER.warning(f"[PATTERN_TRACKER] Failed to record: {e}")
        
        return prediction
    
    async def predict_batch(self, symbols: List[str], bypass_calendar: bool = False) -> Dict[str, StockPrediction]:
        """Predict multiple stocks in parallel
        
        Args:
            symbols: List of stock symbols to predict
            bypass_calendar: If True, skip FOMC/CPI/NFP blackout checks (for testing)
        """
        if not self._initialized:
            await self.initialize()
        
        tasks = [self.predict(s, bypass_calendar=bypass_calendar) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        predictions = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                LOGGER.error(f"Prediction failed for {symbol}: {result}")
                predictions[symbol] = StockPrediction(
                    symbol=symbol,
                    direction="HOLD",
                    confidence=0,
                    entry_price=0,
                    target_price=0,
                    stop_loss=0,
                    horizon_hours=self.config.horizon_hours,
                    confirmations=0,
                    gates_passed=[],
                    gates_failed=[str(result)],
                    reasons=["Prediction failed"]
                )
            else:
                predictions[symbol] = result
        
        return predictions


# Singleton instance
_stock_engine: Optional[StockEngine] = None


def get_stock_engine() -> StockEngine:
    """Get or create the singleton stock engine"""
    global _stock_engine
    if _stock_engine is None:
        _stock_engine = StockEngine()
    return _stock_engine


async def run_stock_prediction(symbol: str) -> Dict[str, Any]:
    """
    Public API: Run stock prediction and return dict result.
    
    This is the main entry point from wolf_app.py
    """
    engine = get_stock_engine()
    prediction = await engine.predict(symbol)
    return prediction.to_dict()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    async def test():
        print("🏛️ Stock Engine Test")
        print("=" * 60)
        
        engine = StockEngine()
        await engine.initialize()
        
        print(f"\nConfig: {engine.config.to_dict()}")
        
        test_symbols = ["AAPL", "MSFT", "JPM"]
        
        for symbol in test_symbols:
            print(f"\n{'='*60}")
            print(f"Testing {symbol}:")
            print("-" * 40)
            
            prediction = await engine.predict(symbol)
            
            print(f"Direction: {prediction.direction}")
            print(f"Confidence: {prediction.confidence:.1%}")
            print(f"Entry: ${prediction.entry_price:.2f}")
            print(f"Target: ${prediction.target_price:.2f}")
            print(f"Stop: ${prediction.stop_loss:.2f}")
            print(f"Confirmations: {prediction.confirmations}/{engine.config.min_confirmations}")
            print(f"Actionable: {prediction.is_actionable}")
            
            print(f"\nGates Passed: {prediction.gates_passed}")
            print(f"Gates Failed: {prediction.gates_failed}")
            print(f"Reasons: {prediction.reasons}")
    
    asyncio.run(test())

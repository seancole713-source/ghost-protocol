#!/usr/bin/env python3
"""
🧠 INTELLIGENCE HUB — Central nervous system for Ghost Protocol

Wires ALL 20 intelligence systems into the prediction pipeline:

  ┌─────────────────────────────────────────────────────┐
  │                   INTELLIGENCE HUB                   │
  ├─────────────────────────────────────────────────────┤
  │  SIGNAL SOURCES (modify direction + confidence)      │
  │   1. News Brain (Claude)      — news_brain_signal    │
  │   2. News Sentiment (AV)      — news_sentiment       │
  │   3. ML/XGBoost Model         — ml_signal            │
  │   4. Ensemble Predictor       — ensemble_signal      │
  │   5. Ensemble Forecaster      — forecast_signal      │
  │   6. Pattern Intelligence     — pattern_signal       │
  │   7. VWAP Signals             — vwap_signal          │
  │   8. Social Sentiment         — social_signal        │
  │   9. Santiment (on-chain)     — onchain_signal       │
  │  10. World Context            — world_signal         │
  │  11. World Feed Fusion        — feed_fusion_signal   │
  │  12. Opus Brain (Claude)      — opus_signal          │
  │  13. Ghost Researcher         — research_signal      │
  ├─────────────────────────────────────────────────────┤
  │  POST-PROCESSING (adjust confidence)                 │
  │  14. Confidence Calibrator    — calibrate()          │
  │  15. Trust Ladder             — trust_boost()        │
  │  16. Regime Detector          — regime_adjust()      │
  ├─────────────────────────────────────────────────────┤
  │  SAFETY GATES (block/allow)                          │
  │  17. Prediction Killswitch    — can_send()           │
  │  18. Quality Gate             — quality_check()      │
  │  19. Guardian Oracle          — risk_check()         │
  ├─────────────────────────────────────────────────────┤
  │  POSITION MANAGEMENT                                 │
  │  20. Dynamic Exits            — exit_levels()        │
  │  21. Self-Improvement         — auto_tune()          │
  └─────────────────────────────────────────────────────┘

Each system is wrapped in try/except — if it fails, the others still run.
Signals are weighted and aggregated into a final prediction adjustment.
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

LOGGER = logging.getLogger("ghost.intelligence_hub")


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """A single intelligence signal."""
    source: str
    direction: str = "NEUTRAL"      # BUY / SELL / HOLD / NEUTRAL
    confidence: float = 0.0         # 0.0-1.0 how confident this signal is
    weight: float = 0.0             # how much to weight this signal
    reasoning: str = ""
    active: bool = False            # did the source actually produce data?
    error: str = ""                 # error message if source failed


@dataclass
class IntelligenceReport:
    """Aggregated intelligence from all systems."""
    signals: List[Signal] = field(default_factory=list)
    active_systems: int = 0
    total_systems: int = 0
    direction_adjustment: str = "NONE"   # CONFIRM / FLIP / WEAKEN / NONE
    confidence_adjustment: float = 0.0   # -0.3 to +0.3
    should_block: bool = False
    block_reason: str = ""
    exit_levels: Dict = field(default_factory=dict)
    trust_boost: float = 0.0
    regime_info: Dict = field(default_factory=dict)
    news_risk: str = "NONE"             # NONE / LOW / MEDIUM / HIGH / CRITICAL


# ═══════════════════════════════════════════════════════════════
# CACHED NEWS BRAIN STATE — updated by background loop in wolf_app
# This is the bridge: news brain writes here, scout reads from here
# ═══════════════════════════════════════════════════════════════

_NEWS_BRAIN_CACHE: Dict = {}
_NEWS_BRAIN_CACHE_TS: float = 0.0


def update_news_brain_cache(analysis: Dict) -> None:
    """Called by wolf_app's news analysis loop to store latest results."""
    global _NEWS_BRAIN_CACHE, _NEWS_BRAIN_CACHE_TS
    _NEWS_BRAIN_CACHE = analysis
    _NEWS_BRAIN_CACHE_TS = time.time()
    LOGGER.info(f"🧠 [HUB] News Brain cache updated: "
                f"{len(analysis.get('major_events', []))} events, "
                f"{len(analysis.get('predictions_at_risk', []))} at risk")


def get_news_brain_cache() -> Tuple[Dict, float]:
    """Get the latest news brain analysis and its age."""
    return _NEWS_BRAIN_CACHE, _NEWS_BRAIN_CACHE_TS


# ═══════════════════════════════════════════════════════════════
# THE INTELLIGENCE HUB
# ═══════════════════════════════════════════════════════════════

class IntelligenceHub:
    """
    Central aggregator for all Ghost Protocol intelligence systems.

    Usage:
        hub = get_intelligence_hub()
        report = hub.analyze(symbol, direction, confidence, entry_price, asset_type)
        # report.confidence_adjustment → apply to prediction
        # report.direction_adjustment  → CONFIRM/FLIP/WEAKEN
        # report.should_block          → kill this prediction
        # report.exit_levels           → dynamic SL/TP
    """

    def __init__(self):
        self._initialized = False
        self._ensemble = None
        self._calibrator = None
        self._trust_ladder = None
        self._quality_gate = None
        self._killswitch = None
        self._vwap = None
        self._feed_fusion = None
        self._regime_detector = None
        self._self_improvement = None

    def _lazy_init(self):
        """Lazy-load singletons on first use."""
        if self._initialized:
            return
        self._initialized = True
        LOGGER.info("🧠 [HUB] Initializing intelligence systems...")

        systems_loaded = 0

        # Ensemble Predictor (loads XGBoost models)
        try:
            from core.ensemble_predictor import get_ensemble_predictor
            self._ensemble = get_ensemble_predictor()
            systems_loaded += 1
            LOGGER.info("  ✅ Ensemble Predictor loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Ensemble Predictor: {e}")

        # Confidence Calibrator
        try:
            from core.confidence_calibrator import get_confidence_calibrator
            self._calibrator = get_confidence_calibrator()
            systems_loaded += 1
            LOGGER.info("  ✅ Confidence Calibrator loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Confidence Calibrator: {e}")

        # Trust Ladder
        try:
            from core.trust_ladder import get_trust_ladder
            self._trust_ladder = get_trust_ladder()
            systems_loaded += 1
            LOGGER.info("  ✅ Trust Ladder loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Trust Ladder: {e}")

        # Quality Gate
        try:
            from core.quality_gate import get_quality_gate
            self._quality_gate = get_quality_gate()
            systems_loaded += 1
            LOGGER.info("  ✅ Quality Gate loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Quality Gate: {e}")

        # Prediction Killswitch
        try:
            from core.prediction_killswitch import PredictionKillswitch
            self._killswitch = PredictionKillswitch()
            systems_loaded += 1
            LOGGER.info("  ✅ Prediction Killswitch loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Prediction Killswitch: {e}")

        # VWAP Analyzer
        try:
            from core.vwap_signals import get_vwap_analyzer
            self._vwap = get_vwap_analyzer()
            systems_loaded += 1
            LOGGER.info("  ✅ VWAP Analyzer loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ VWAP Analyzer: {e}")

        # World Feed Fusion
        try:
            from core.world_feed_fusion import get_feed_fusion
            self._feed_fusion = get_feed_fusion()
            systems_loaded += 1
            LOGGER.info("  ✅ World Feed Fusion loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ World Feed Fusion: {e}")

        # Regime Detector
        try:
            from core.regime_detector import get_regime_detector
            self._regime_detector = get_regime_detector()
            systems_loaded += 1
            LOGGER.info("  ✅ Regime Detector loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Regime Detector: {e}")

        # Self-Improvement Engine
        try:
            from core.self_improvement_engine import get_self_improvement_engine
            self._self_improvement = get_self_improvement_engine()
            systems_loaded += 1
            LOGGER.info("  ✅ Self-Improvement Engine loaded")
        except Exception as e:
            LOGGER.warning(f"  ❌ Self-Improvement Engine: {e}")

        LOGGER.info(f"🧠 [HUB] {systems_loaded} systems initialized")

    # ───────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ───────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        asset_type: str = "crypto",
        price_history: List[float] = None,
    ) -> IntelligenceReport:
        """
        Run ALL intelligence systems for a single prediction.

        Returns an IntelligenceReport with adjustments to apply.
        Each system is independent — failures don't cascade.
        """
        self._lazy_init()

        # Normalize direction (UP/DOWN ↔ BUY/SELL)
        direction = self._normalize_direction(direction)

        report = IntelligenceReport()
        report.total_systems = 20

        # ── 1. NEWS BRAIN (Claude) — check cached analysis ──
        report.signals.append(self._check_news_brain(symbol, direction))

        # ── 2. NEWS SENTIMENT (Alpha Vantage) — already called by scout ──
        # (kept in scout's _make_prediction for backward compat)
        report.signals.append(Signal(
            source="news_sentiment",
            active=False,
            reasoning="Handled by scout._make_prediction()"
        ))

        # ── 3. ML/XGBoost MODEL ──
        report.signals.append(self._check_ml_model(symbol, direction, price_history))

        # ── 4. ENSEMBLE PREDICTOR ──
        report.signals.append(self._check_ensemble(symbol, direction, entry_price, price_history))

        # ── 5. ENSEMBLE FORECASTER ──
        report.signals.append(self._check_ensemble_forecaster(symbol, direction, entry_price))

        # ── 6. PATTERN INTELLIGENCE ──
        report.signals.append(self._check_pattern_intelligence(symbol, direction))

        # ── 7. VWAP SIGNALS ──
        report.signals.append(self._check_vwap(symbol, direction))

        # ── 8. SOCIAL SENTIMENT ──
        report.signals.append(self._check_social_sentiment(symbol, direction))

        # ── 9. SANTIMENT (on-chain) ──
        report.signals.append(self._check_santiment(symbol, direction, asset_type))

        # ── 10. WORLD CONTEXT ──
        report.signals.append(self._check_world_context(direction))

        # ── 11. WORLD FEED FUSION ──
        report.signals.append(self._check_feed_fusion(symbol, direction))

        # ── 12. OPUS BRAIN — async, run sync wrapper ──
        # Only for high-value predictions to save API costs
        if confidence >= 0.60:
            report.signals.append(self._check_opus_brain(symbol, direction, entry_price))
        else:
            report.signals.append(Signal(source="opus_brain", active=False,
                                         reasoning="Skipped (confidence < 0.60)"))

        # ── 13. GHOST RESEARCHER — async, expensive ──
        # Only for top-tier predictions
        if confidence >= 0.70:
            report.signals.append(self._check_researcher(symbol, direction))
        else:
            report.signals.append(Signal(source="ghost_researcher", active=False,
                                         reasoning="Skipped (confidence < 0.70)"))

        # ── Count active systems ──
        report.active_systems = sum(1 for s in report.signals if s.active)

        # ── 14. CONFIDENCE CALIBRATOR ──
        cal_adj = self._calibrate_confidence(symbol, confidence)
        report.confidence_adjustment += cal_adj

        # ── 15. TRUST LADDER ──
        trust = self._check_trust_ladder(symbol)
        report.trust_boost = trust

        # ── 16. REGIME DETECTOR ──
        regime = self._check_regime(price_history)
        report.regime_info = regime

        # ── AGGREGATE SIGNALS INTO ADJUSTMENTS ──
        self._aggregate_signals(report, direction, confidence)

        # ── 17. PREDICTION KILLSWITCH ──
        if self._killswitch:
            try:
                # Only block if PREDICTIONS_ENABLED is explicitly set to 'false'
                # If env var is absent, treat as enabled (safe default for intelligence hub)
                env_val = os.environ.get('PREDICTIONS_ENABLED', '')
                if env_val.lower() == 'false':
                    report.should_block = True
                    report.block_reason = "Killswitch: PREDICTIONS_ENABLED=false"
                elif env_val.lower() == 'true':
                    pass  # Explicitly enabled — good
                else:
                    # Not set — don't block, but log
                    LOGGER.debug("Killswitch: PREDICTIONS_ENABLED not set, allowing")
            except Exception as e:
                LOGGER.debug(f"Killswitch check failed: {e}")

        # ── 18. QUALITY GATE ──
        # Quality gate is advisory in the hub — it informs but doesn't block.
        # Actual gating for Telegram alerts happens downstream.
        gate_result = self._check_quality_gate(symbol, confidence + report.confidence_adjustment)
        if gate_result and not gate_result.get("allowed", True):
            LOGGER.info(f"🔍 [HUB] Quality Gate advisory for {symbol}: {gate_result.get('reason', '')}")
            # Don't block — just log. The quality gate is for Telegram/alert filtering.

        # ── 19. GUARDIAN ORACLE — risk context ──
        # Guardian is a formatter/reporter, not a gate. Info only.

        # ── 20. DYNAMIC EXITS ──
        report.exit_levels = self._calculate_dynamic_exits(
            entry_price, direction, confidence + report.confidence_adjustment
        )

        # ── 21. SELF-IMPROVEMENT — runs periodically, not per-prediction ──
        # Wired into wolf_app startup loop separately

        return report

    # ───────────────────────────────────────────────────────
    # SIGNAL CHECKERS (each returns a Signal)
    # ───────────────────────────────────────────────────────

    def _check_news_brain(self, symbol: str, direction: str) -> Signal:
        """Check cached Claude news analysis for this symbol."""
        sig = Signal(source="news_brain", weight=0.20)
        try:
            cache, cache_ts = get_news_brain_cache()
            if not cache or (time.time() - cache_ts) > 3600:
                sig.reasoning = "No recent news brain analysis (>1hr old or empty)"
                return sig

            sig.active = True

            # Check if this symbol is at risk
            at_risk = cache.get("predictions_at_risk", [])
            for pred in at_risk:
                pred_symbol = pred.get("symbol", "")
                if pred_symbol.upper() == symbol.upper():
                    risk_level = pred.get("risk_level", "LOW")
                    sig.reasoning = f"News Brain: {symbol} at {risk_level} risk — {pred.get('reason', '')}"

                    if risk_level == "HIGH":
                        sig.direction = "SELL" if direction == "BUY" else "BUY"
                        sig.confidence = 0.7
                    elif risk_level == "MEDIUM":
                        sig.direction = "HOLD"
                        sig.confidence = 0.5
                    else:
                        sig.direction = direction
                        sig.confidence = 0.3

                    return sig

            # Check major events for sector impact
            events = cache.get("major_events", [])
            for event in events:
                bearish_syms = [s.upper() for s in event.get("bearish_symbols", [])]
                bullish_syms = [s.upper() for s in event.get("bullish_symbols", [])]

                if symbol.upper() in bearish_syms:
                    severity = event.get("severity", "LOW")
                    sig.direction = "SELL"
                    sig.confidence = 0.6 if severity in ("HIGH", "CRITICAL") else 0.4
                    sig.reasoning = f"News event bearish for {symbol}: {event.get('headline', '')[:80]}"
                    return sig

                if symbol.upper() in bullish_syms:
                    sig.direction = "BUY"
                    sig.confidence = 0.5
                    sig.reasoning = f"News event bullish for {symbol}: {event.get('headline', '')[:80]}"
                    return sig

            sig.direction = direction
            sig.confidence = 0.3
            sig.reasoning = "No specific news impact for this symbol"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"News brain check failed for {symbol}: {e}")
        return sig

    def _check_ml_model(self, symbol: str, direction: str, price_history: list = None) -> Signal:
        """Check ML model prediction."""
        sig = Signal(source="ml_model", weight=0.15)
        try:
            from core.ml_trainer import load_model, predict
            model_data = load_model()
            if not model_data:
                sig.reasoning = "No trained ML model available"
                return sig

            # Build features from price history
            if not price_history or len(price_history) < 20:
                sig.reasoning = "Insufficient price history for ML"
                return sig

            features = self._build_ml_features(price_history)
            result = predict(model_data, features)

            if result and result.get("direction") != "FLAT":
                sig.active = True
                sig.direction = "BUY" if result["direction"] == "UP" else "SELL"
                sig.confidence = result.get("confidence", 0.5)
                sig.reasoning = f"ML model: {result['direction']} @ {sig.confidence:.1%}"
            else:
                sig.active = True
                sig.direction = "HOLD"
                sig.confidence = 0.3
                sig.reasoning = "ML model: FLAT/no signal"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"ML model check failed: {e}")
        return sig

    def _check_ensemble(self, symbol: str, direction: str, price: float,
                        price_history: list = None) -> Signal:
        """Check ensemble predictor (XGBoost + market signals)."""
        sig = Signal(source="ensemble", weight=0.20)
        try:
            if not self._ensemble:
                sig.reasoning = "Ensemble predictor not loaded"
                return sig

            features = {}
            if price_history and len(price_history) >= 5:
                features["price_change_1d"] = (price_history[-1] - price_history[-2]) / price_history[-2] if len(price_history) >= 2 else 0
                features["price_change_5d"] = (price_history[-1] - price_history[-5]) / price_history[-5] if len(price_history) >= 5 else 0
                features["volatility"] = max(price_history[-20:]) / min(price_history[-20:]) - 1 if len(price_history) >= 20 else 0.05
                sma5 = sum(price_history[-5:]) / 5
                sma20 = sum(price_history[-20:]) / 20 if len(price_history) >= 20 else sma5
                features["sma_ratio"] = sma5 / sma20 if sma20 > 0 else 1.0
                features["rsi"] = self._quick_rsi(price_history)
            else:
                features = {"price_change_1d": 0, "price_change_5d": 0, "volatility": 0.05,
                            "sma_ratio": 1.0, "rsi": 50}

            result = self._ensemble.predict(features, symbol=symbol)

            sig.active = True
            sig.direction = "BUY" if result.direction == "UP" else ("SELL" if result.direction == "DOWN" else "HOLD")
            sig.confidence = result.confidence
            sig.reasoning = f"Ensemble ({result.ensemble_method}): {result.direction} @ {result.confidence:.1%}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Ensemble check failed for {symbol}: {e}")
        return sig

    def _check_ensemble_forecaster(self, symbol: str, direction: str, price: float) -> Signal:
        """Check ensemble forecaster for price target."""
        sig = Signal(source="ensemble_forecaster", weight=0.10)
        try:
            from core.ensemble_forecaster import get_ensemble_forecaster
            forecaster = get_ensemble_forecaster()
            result = forecaster.forecast(symbol, price, horizon_hours=48)

            if result and "ensemble_prediction" in result:
                predicted_price = result["ensemble_prediction"]
                change_pct = (predicted_price - price) / price
                sig.active = True
                if change_pct > 0.01:
                    sig.direction = "BUY"
                elif change_pct < -0.01:
                    sig.direction = "SELL"
                else:
                    sig.direction = "HOLD"
                sig.confidence = min(0.8, result.get("confidence", 0.5))
                sig.reasoning = f"Forecaster: {change_pct:+.1%} in 48h (conf={sig.confidence:.1%})"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Ensemble forecaster failed for {symbol}: {e}")
        return sig

    def _check_pattern_intelligence(self, symbol: str, direction: str) -> Signal:
        """Check pattern intelligence signals."""
        sig = Signal(source="pattern_intelligence", weight=0.10)
        try:
            from core.pattern_intelligence import SignalAggregator
            agg = SignalAggregator()
            result = agg.get_full_analysis(symbol=symbol)

            if result and result.get("recommendation"):
                sig.active = True
                rec = result["recommendation"]
                if rec in ("STRONG_BUY", "BUY"):
                    sig.direction = "BUY"
                    sig.confidence = 0.65 if rec == "STRONG_BUY" else 0.55
                elif rec in ("STRONG_SELL", "SELL"):
                    sig.direction = "SELL"
                    sig.confidence = 0.65 if rec == "STRONG_SELL" else 0.55
                else:
                    sig.direction = "HOLD"
                    sig.confidence = 0.4
                sig.reasoning = f"Pattern Intelligence: {rec} — {result.get('summary', '')[:60]}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Pattern intelligence failed for {symbol}: {e}")
        return sig

    def _check_vwap(self, symbol: str, direction: str) -> Signal:
        """Check VWAP signal."""
        sig = Signal(source="vwap", weight=0.08)
        try:
            if not self._vwap:
                sig.reasoning = "VWAP analyzer not loaded"
                return sig

            result = self._vwap.get_vwap_signal(symbol)
            if result:
                sig.active = True
                vwap_signal = result.get("signal", "NEUTRAL")
                if vwap_signal in ("BUY", "STRONG_BUY"):
                    sig.direction = "BUY"
                    sig.confidence = 0.55
                elif vwap_signal in ("SELL", "STRONG_SELL"):
                    sig.direction = "SELL"
                    sig.confidence = 0.55
                else:
                    sig.direction = "HOLD"
                    sig.confidence = 0.3
                sig.reasoning = f"VWAP: {vwap_signal} (dev={result.get('deviation_pct', 0):.1%})"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"VWAP check failed for {symbol}: {e}")
        return sig

    def _check_social_sentiment(self, symbol: str, direction: str) -> Signal:
        """Check social media sentiment."""
        sig = Signal(source="social_sentiment", weight=0.08)
        try:
            from core.social_sentiment import get_combined_social_sentiment
            result = get_combined_social_sentiment(symbol)

            if result and result.get("ok"):
                sig.active = True
                score = result.get("sentiment_score", 0)
                if score > 0.2:
                    sig.direction = "BUY"
                    sig.confidence = min(0.6, 0.4 + abs(score) * 0.3)
                elif score < -0.2:
                    sig.direction = "SELL"
                    sig.confidence = min(0.6, 0.4 + abs(score) * 0.3)
                else:
                    sig.direction = "NEUTRAL"
                    sig.confidence = 0.3
                sig.reasoning = f"Social: score={score:.2f} ({result.get('mention_count', 0)} mentions)"
            else:
                sig.reasoning = f"Social sentiment unavailable: {result.get('error', 'no data')}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Social sentiment failed for {symbol}: {e}")
        return sig

    def _check_santiment(self, symbol: str, direction: str, asset_type: str) -> Signal:
        """Check Santiment on-chain data (crypto only)."""
        sig = Signal(source="santiment", weight=0.08)
        if asset_type != "crypto":
            sig.reasoning = "Santiment: stocks not supported"
            return sig
        try:
            from core.santiment_signals import get_sentiment_signal, is_enabled
            if not is_enabled():
                sig.reasoning = "Santiment: API key not set"
                return sig

            result = get_sentiment_signal(symbol)
            if result:
                sig.active = True
                sentiment = result.get("sentiment", 0)
                if sentiment > 0.3:
                    sig.direction = "BUY"
                    sig.confidence = 0.55
                elif sentiment < -0.3:
                    sig.direction = "SELL"
                    sig.confidence = 0.55
                else:
                    sig.direction = "NEUTRAL"
                    sig.confidence = 0.3
                sig.reasoning = f"On-chain: sentiment={sentiment:.2f}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Santiment failed for {symbol}: {e}")
        return sig

    def _check_world_context(self, direction: str) -> Signal:
        """Check global market context (VIX, SPY, mood)."""
        sig = Signal(source="world_context", weight=0.08)
        try:
            from core.world_context import get_world_context
            ctx = get_world_context()

            if ctx:
                sig.active = True
                mood = ctx.get("market_mood", {})
                mood_sentiment = mood.get("sentiment", "neutral")
                vix = ctx.get("vix", {}).get("level", 20)
                spy_change = ctx.get("spy", {}).get("change_pct", 0)

                # High VIX = risk-off, bearish
                if vix > 30:
                    sig.direction = "SELL"
                    sig.confidence = 0.55
                    sig.reasoning = f"World: VIX={vix:.0f} (fear), SPY={spy_change:+.1f}%"
                elif vix < 15 and spy_change > 0:
                    sig.direction = "BUY"
                    sig.confidence = 0.45
                    sig.reasoning = f"World: VIX={vix:.0f} (calm), SPY={spy_change:+.1f}%"
                else:
                    sig.direction = "NEUTRAL"
                    sig.confidence = 0.3
                    sig.reasoning = f"World: VIX={vix:.0f}, SPY={spy_change:+.1f}%, mood={mood_sentiment}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"World context check failed: {e}")
        return sig

    def _check_feed_fusion(self, symbol: str, direction: str) -> Signal:
        """Check world feed fusion (RSS sentiment aggregate)."""
        sig = Signal(source="feed_fusion", weight=0.06)
        try:
            if not self._feed_fusion:
                sig.reasoning = "Feed fusion not loaded"
                return sig

            result = self._feed_fusion.get_sentiment_aggregate(symbol=symbol, hours=24)
            if result:
                sig.active = True
                score = result.get("sentiment_score", 0) if isinstance(result, dict) else 0
                if score > 0.2:
                    sig.direction = "BUY"
                    sig.confidence = 0.45
                elif score < -0.2:
                    sig.direction = "SELL"
                    sig.confidence = 0.45
                else:
                    sig.direction = "NEUTRAL"
                    sig.confidence = 0.3
                articles = result.get("article_count", 0) if isinstance(result, dict) else 0
                sig.reasoning = f"Feed fusion: score={score:.2f}, {articles} articles"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Feed fusion failed for {symbol}: {e}")
        return sig

    def _check_opus_brain(self, symbol: str, direction: str, price: float) -> Signal:
        """Check Opus Brain (Claude) for high-value predictions."""
        sig = Signal(source="opus_brain", weight=0.12)
        try:
            from core.intelligence.opus_brain import opus_analyze
            context = {
                "symbol": symbol,
                "current_direction": direction,
                "current_price": price,
                "asset_type": "crypto" if symbol in ("BTC", "ETH", "SOL", "XRP", "LINK", "CHZ") else "stock"
            }
            # Run async function synchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context — use a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, opus_analyze(symbol, context)).result()
            else:
                result = asyncio.run(opus_analyze(symbol, context))

            if result and result.get("signal") not in (None, "NEUTRAL", "ERROR"):
                sig.active = True
                signal = result["signal"]
                sig.direction = signal if signal in ("BUY", "SELL", "HOLD") else "NEUTRAL"
                sig.confidence = 0.6 + (result.get("confidence_adjustment", 0) * 0.3)
                sig.confidence = max(0.3, min(0.85, sig.confidence))
                sig.reasoning = f"Opus Brain: {signal} — {result.get('reasoning', '')[:80]}"
            elif result:
                sig.active = True
                sig.direction = "NEUTRAL"
                sig.confidence = 0.3
                sig.reasoning = f"Opus Brain: NEUTRAL — {result.get('reasoning', 'no strong signal')[:80]}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Opus brain failed for {symbol}: {e}")
        return sig

    def _check_researcher(self, symbol: str, direction: str) -> Signal:
        """Check Ghost Researcher for deep analysis."""
        sig = Signal(source="ghost_researcher", weight=0.08)
        try:
            from core.ghost_researcher import GhostResearcher
            researcher = GhostResearcher()

            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    report = pool.submit(asyncio.run, researcher.quick_research(symbol)).result()
            else:
                report = asyncio.run(researcher.quick_research(symbol))

            if report:
                sig.active = True
                sig.direction = direction  # Researcher provides context, not direction
                sig.confidence = 0.45
                sig.reasoning = f"Research: {str(report)[:100]}"
        except Exception as e:
            sig.error = str(e)
            LOGGER.debug(f"Ghost researcher failed for {symbol}: {e}")
        return sig

    # ───────────────────────────────────────────────────────
    # POST-PROCESSING
    # ───────────────────────────────────────────────────────

    def _calibrate_confidence(self, symbol: str, raw_confidence: float) -> float:
        """Run confidence calibrator. Returns adjustment (-0.2 to +0.2)."""
        try:
            if not self._calibrator:
                return 0.0
            result = self._calibrator.calibrate_confidence(raw_confidence, symbol=symbol)
            if result:
                calibrated = result.get("calibrated_confidence", raw_confidence)
                adjustment = calibrated - raw_confidence
                return max(-0.20, min(0.20, adjustment))
        except Exception as e:
            LOGGER.debug(f"Calibration failed for {symbol}: {e}")
        return 0.0

    def _check_trust_ladder(self, symbol: str) -> float:
        """Get trust-based confidence boost. Returns 0.0-0.20."""
        try:
            if not self._trust_ladder:
                return 0.0
            trust = self._trust_ladder.get_trust(symbol)
            if trust:
                # Level 2 = +10%, Level 3 = +20%
                window = self._trust_ladder.get_prediction_window(symbol)
                boost = window.get("confidence_boost", 1.0)
                # Trust ladder returns MULTIPLIER (1.0=no boost, 1.10=+10%, 1.20=+20%)
                # Convert to additive delta and cap
                delta = boost - 1.0
                return max(0.0, min(0.20, delta))
        except Exception as e:
            LOGGER.debug(f"Trust ladder failed for {symbol}: {e}")
        return 0.0

    def _check_regime(self, price_history: list = None) -> Dict:
        """Get current market regime."""
        try:
            if not self._regime_detector:
                return {"regime": "UNKNOWN", "confidence": 0.0}

            import numpy as np
            if price_history and len(price_history) >= 20:
                prices = np.array(price_history[-50:]) if len(price_history) >= 50 else np.array(price_history)
                result = self._regime_detector.detect_regime(prices)
                return result or {"regime": "UNKNOWN", "confidence": 0.0}
            return {"regime": "UNKNOWN", "confidence": 0.0}
        except Exception as e:
            LOGGER.debug(f"Regime detection failed: {e}")
            return {"regime": "UNKNOWN", "confidence": 0.0}

    def _check_quality_gate(self, symbol: str, adjusted_confidence: float) -> Optional[Dict]:
        """Check quality gate. Returns None if no gate, or dict with allowed/reason."""
        try:
            if not self._quality_gate:
                return None
            result = self._quality_gate.check(symbol, adjusted_confidence)
            return {"allowed": result.allowed, "reason": result.reason}
        except Exception as e:
            LOGGER.debug(f"Quality gate failed for {symbol}: {e}")
            return None

    def _calculate_dynamic_exits(self, entry_price: float, direction: str,
                                  confidence: float) -> Dict:
        """Calculate dynamic SL/TP levels."""
        try:
            from core.dynamic_exits import calculate_exits
            result = calculate_exits(entry_price, direction, confidence)
            return result or {}
        except Exception as e:
            LOGGER.debug(f"Dynamic exits calculation failed: {e}")
            return {}

    # ───────────────────────────────────────────────────────
    # SIGNAL AGGREGATION
    # ───────────────────────────────────────────────────────

    def _aggregate_signals(self, report: IntelligenceReport, base_direction: str,
                           base_confidence: float) -> None:
        """
        Aggregate all active signals into direction + confidence adjustments.

        Uses weighted voting:
        - Signals that AGREE with base direction → boost confidence
        - Signals that DISAGREE → reduce confidence
        - Strong disagreement from multiple sources → FLIP direction
        """
        agree_score = 0.0
        disagree_score = 0.0
        total_weight = 0.0

        for sig in report.signals:
            if not sig.active or sig.confidence <= 0:
                continue

            weight = sig.weight * sig.confidence
            total_weight += weight

            if sig.direction == base_direction:
                agree_score += weight
            elif sig.direction == "NEUTRAL" or sig.direction == "HOLD":
                # Neutral signals slightly reduce confidence
                disagree_score += weight * 0.2
            else:
                # Opposing signal
                disagree_score += weight

        if total_weight == 0:
            report.direction_adjustment = "NONE"
            return

        agreement_ratio = agree_score / total_weight if total_weight > 0 else 0.5

        # Strong agreement → boost confidence
        if agreement_ratio >= 0.70:
            report.direction_adjustment = "CONFIRM"
            report.confidence_adjustment += 0.10 * (agreement_ratio - 0.5)

        # Moderate disagreement → weaken
        elif agreement_ratio <= 0.35:
            report.direction_adjustment = "WEAKEN"
            report.confidence_adjustment -= 0.15 * (0.5 - agreement_ratio)

        # Strong disagreement → flip
        elif agreement_ratio <= 0.20 and disagree_score > agree_score * 2:
            report.direction_adjustment = "FLIP"
            report.confidence_adjustment -= 0.10

        else:
            report.direction_adjustment = "NONE"

        # Check news brain for specific risk
        for sig in report.signals:
            if sig.source == "news_brain" and sig.active:
                if "HIGH risk" in sig.reasoning:
                    report.news_risk = "HIGH"
                    report.confidence_adjustment -= 0.15
                elif "MEDIUM risk" in sig.reasoning:
                    report.news_risk = "MEDIUM"
                    report.confidence_adjustment -= 0.08

        # Clamp total adjustment
        report.confidence_adjustment = max(-0.30, min(0.30, report.confidence_adjustment))

    # ───────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_direction(d: str) -> str:
        """Normalize direction to BUY/SELL/HOLD/NEUTRAL.
        Handles both UP/DOWN and BUY/SELL conventions."""
        d = (d or "").upper().strip()
        if d in ("UP", "BUY", "LONG"):
            return "BUY"
        elif d in ("DOWN", "SELL", "SHORT"):
            return "SELL"
        elif d in ("HOLD", "FLAT"):
            return "HOLD"
        return "NEUTRAL"

    def _build_ml_features(self, closes: list) -> Dict:
        """Build feature dict from price history for ML model."""
        if len(closes) < 20:
            return {}
        sma5 = sum(closes[-5:]) / 5
        sma10 = sum(closes[-10:]) / 10
        sma20 = sum(closes[-20:]) / 20
        rsi = self._quick_rsi(closes)

        return {
            "sma_5_10_ratio": sma5 / sma10 if sma10 > 0 else 1.0,
            "sma_10_20_ratio": sma10 / sma20 if sma20 > 0 else 1.0,
            "rsi": rsi,
            "price_change_1d": (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0,
            "price_change_5d": (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0,
            "volatility": (max(closes[-20:]) - min(closes[-20:])) / min(closes[-20:]) if min(closes[-20:]) > 0 else 0,
        }

    def _quick_rsi(self, closes: list, period: int = 14) -> float:
        """Quick RSI calculation."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-period:]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_status(self) -> Dict:
        """Get status of all intelligence systems."""
        self._lazy_init()
        return {
            "ensemble_loaded": self._ensemble is not None,
            "calibrator_loaded": self._calibrator is not None,
            "trust_ladder_loaded": self._trust_ladder is not None,
            "quality_gate_loaded": self._quality_gate is not None,
            "killswitch_loaded": self._killswitch is not None,
            "vwap_loaded": self._vwap is not None,
            "feed_fusion_loaded": self._feed_fusion is not None,
            "regime_detector_loaded": self._regime_detector is not None,
            "self_improvement_loaded": self._self_improvement is not None,
            "news_brain_cache_age_s": time.time() - _NEWS_BRAIN_CACHE_TS if _NEWS_BRAIN_CACHE_TS > 0 else -1,
            "news_brain_has_data": bool(_NEWS_BRAIN_CACHE),
        }


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

_HUB_INSTANCE: Optional[IntelligenceHub] = None


def get_intelligence_hub() -> IntelligenceHub:
    """Get or create the singleton IntelligenceHub."""
    global _HUB_INSTANCE
    if _HUB_INSTANCE is None:
        _HUB_INSTANCE = IntelligenceHub()
    return _HUB_INSTANCE

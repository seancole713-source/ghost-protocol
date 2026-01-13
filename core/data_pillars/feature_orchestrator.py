"""
Feature Orchestrator
====================

Unified interface to all 6 data pillars.
Fetches features from all pillars and returns structured feature vector.

Usage:
    orchestrator = FeatureOrchestrator()
    features = orchestrator.get_all_features("AAPL")
    
    # Returns dict with 50+ features from all pillars
    {
        "symbol": "AAPL",
        "timestamp": 1700000000.0,
        "features": {
            "PRICE": 150.25,
            "RSI_14": 67.5,
            "MACD_HISTOGRAM": 0.45,
            "VOLUME_SPIKE": 0.23,
            "NEWS_SENTIMENT_SCORE": 0.65,
            "SPY_PRICE": 450.12,
            ...
        },
        "feature_availability": {
            "price_engine": 6/6,
            "technical_engine": 12/15,
            "volume_engine": 5/5,
            ...
        },
        "execution_time_ms": 234.5
    }

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import time
from typing import Any

from core.data_pillars.flow_engine import FlowEngine
from core.data_pillars.price_engine import PriceEngine
from core.data_pillars.sentiment_engine import SentimentEngine  # FIXED: Now uses Ghost News Brain + RSS
from core.data_pillars.technical_engine import TechnicalEngine
from core.data_pillars.volume_engine import VolumeEngine
from core.data_pillars.world_context_engine import WorldContextEngine  # FIXED: Added yfinance fallback

logger = logging.getLogger(__name__)


class FeatureOrchestrator:
    """
    Orchestrates feature extraction from all 6 data pillars.
    
    Handles parallel fetching, error recovery, and feature aggregation.
    """

    def __init__(self):
        """Initialize all 6 data pillar engines"""
        self.price_engine = PriceEngine()
        self.technical_engine = TechnicalEngine()
        self.volume_engine = VolumeEngine()
        self.sentiment_engine = SentimentEngine()  # FIXED: Ghost News Brain integration
        self.world_context_engine = WorldContextEngine()  # FIXED: yfinance fallback
        self.flow_engine = FlowEngine()

    def get_all_features(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Fetch features from all 6 pillars.
        
        Args:
            symbol: Stock/crypto ticker (e.g., "AAPL", "BTC")
            **kwargs: Options for pillar engines
        
        Returns:
            {
                "ok": True,
                "symbol": str,
                "timestamp": float,
                "features": dict[str, Any],  # All features flattened
                "feature_count": int,
                "available_count": int,
                "unavailable_count": int,
                "feature_availability": dict[str, str],  # Per-pillar status
                "execution_time_ms": float,
                "errors": list[str]
            }
        """
        start_time = time.time()
        all_features = {}
        all_errors = []
        pillar_stats = {}

        # Fetch from each pillar
        # Note: Could parallelize with asyncio/threads in future
        
        # 1. Price Engine
        try:
            price_resp = self.price_engine.get_signals(symbol, **kwargs)
            for signal in price_resp.signals:
                all_features[signal.name] = signal.value
            all_errors.extend(price_resp.errors)
            pillar_stats["price_engine"] = f"{price_resp.available_signal_count()}/{price_resp.signal_count()}"
        except Exception as e:
            logger.error(f"Price engine failed: {e}")
            all_errors.append(f"Price engine: {str(e)}")
            pillar_stats["price_engine"] = "0/0 (failed)"

        # 2. Technical Engine
        try:
            tech_resp = self.technical_engine.get_signals(symbol, **kwargs)
            for signal in tech_resp.signals:
                all_features[signal.name] = signal.value
            all_errors.extend(tech_resp.errors)
            pillar_stats["technical_engine"] = f"{tech_resp.available_signal_count()}/{tech_resp.signal_count()}"
        except Exception as e:
            logger.error(f"Technical engine failed: {e}")
            all_errors.append(f"Technical engine: {str(e)}")
            pillar_stats["technical_engine"] = "0/0 (failed)"

        # 3. Volume Engine
        try:
            vol_resp = self.volume_engine.get_signals(symbol, **kwargs)
            for signal in vol_resp.signals:
                all_features[signal.name] = signal.value
            all_errors.extend(vol_resp.errors)
            pillar_stats["volume_engine"] = f"{vol_resp.available_signal_count()}/{vol_resp.signal_count()}"
        except Exception as e:
            logger.error(f"Volume engine failed: {e}")
            all_errors.append(f"Volume engine: {str(e)}")
            pillar_stats["volume_engine"] = "0/0 (failed)"

        # 4. Sentiment Engine - FIXED (Ghost News Brain + RSS feeds)
        try:
            sent_resp = self.sentiment_engine.get_signals(symbol, **kwargs)
            for signal in sent_resp.signals:
                all_features[signal.name] = signal.value
            all_errors.extend(sent_resp.errors)
            pillar_stats["sentiment_engine"] = f"{sent_resp.available_signal_count()}/{sent_resp.signal_count()}"
        except Exception as e:
            logger.error(f"Sentiment engine failed: {e}")
            all_errors.append(f"Sentiment engine: {str(e)}")
            pillar_stats["sentiment_engine"] = "0/0 (failed)"

        # 5. World Context Engine - FIXED (yfinance fallback for SPY/VIX)
        try:
            world_resp = self.world_context_engine.get_signals()
            for signal in world_resp.signals:
                all_features[signal.name] = signal.value
            all_errors.extend(world_resp.errors)
            pillar_stats["world_context_engine"] = f"{world_resp.available_signal_count()}/{world_resp.signal_count()}"
        except Exception as e:
            logger.error(f"World context engine failed: {e}")
            all_errors.append(f"World context: {str(e)}")
            pillar_stats["world_context_engine"] = "0/0 (failed)"

        # 6. Flow Engine (orderbook/on-chain)
        try:
            flow_resp = self.flow_engine.get_signals(symbol, **kwargs)
            for signal in flow_resp.signals:
                all_features[signal.name] = signal.value
            all_errors.extend(flow_resp.errors)
            pillar_stats["flow_engine"] = f"{flow_resp.available_signal_count()}/{flow_resp.signal_count()}"
        except Exception as e:
            logger.error(f"Flow engine failed: {e}")
            all_errors.append(f"Flow engine: {str(e)}")
            pillar_stats["flow_engine"] = "0/0 (failed)"

        # Calculate stats
        feature_count = len(all_features)
        available_count = sum(1 for v in all_features.values() if v is not None)
        unavailable_count = feature_count - available_count

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "ok": True,
            "symbol": symbol,
            "timestamp": time.time(),
            "features": all_features,
            "feature_count": feature_count,
            "available_count": available_count,
            "unavailable_count": unavailable_count,
            "feature_availability": pillar_stats,
            "execution_time_ms": round(execution_time_ms, 2),
            "errors": all_errors,
        }

    def health_check(self) -> dict[str, Any]:
        """
        Run health checks on all 6 pillars.
        
        Returns:
            {
                "ok": bool,
                "pillars": {
                    "price_engine": {"ok": True, ...},
                    ...
                },
                "summary": {
                    "healthy": int,
                    "degraded": int,
                    "failed": int
                }
            }
        """
        pillar_results = {}
        healthy = 0
        degraded = 0
        failed = 0

        # Check each pillar
        for pillar_name, pillar in [
            ("price_engine", self.price_engine),
            ("technical_engine", self.technical_engine),
            ("volume_engine", self.volume_engine),
            ("sentiment_engine", self.sentiment_engine),  # FIXED
            ("world_context_engine", self.world_context_engine),  # FIXED
            ("flow_engine", self.flow_engine),
        ]:
            try:
                result = pillar.health_check()
                pillar_results[pillar_name] = result

                if result.get("ok"):
                    healthy += 1
                elif result.get("errors"):
                    failed += 1
                else:
                    degraded += 1

            except Exception as e:
                logger.error(f"Health check failed for {pillar_name}: {e}")
                pillar_results[pillar_name] = {
                    "ok": False,
                    "errors": [str(e)],
                }
                failed += 1

        return {
            "ok": healthy >= 4,  # At least 4/6 pillars must be healthy
            "pillars": pillar_results,
            "summary": {
                "healthy": healthy,
                "degraded": degraded,
                "failed": failed,
                "total": 6,  # Back to 6 (sentiment + world_context FIXED)
            },
        }


# Singleton instance
_orchestrator = None


def get_feature_orchestrator() -> FeatureOrchestrator:
    """Get singleton feature orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = FeatureOrchestrator()
    return _orchestrator

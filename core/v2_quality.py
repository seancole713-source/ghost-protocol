#!/usr/bin/env python3
"""
🎯 GHOST PROTOCOL V2 - ASSET QUALITY SYSTEM

Phase 2: Find the Edge
- Track which assets Ghost predicts well
- Dynamic whitelist/blacklist based on performance
- Quality gates for predictions

Only predict assets where we have demonstrated edge.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict

LOGGER = logging.getLogger("ghost.v2_quality")


@dataclass
class AssetQualityMetrics:
    """Quality metrics for a single asset"""
    symbol: str
    win_rate: float
    total_predictions: int
    recent_trend: str  # "improving", "stable", "declining"
    avg_confidence: float
    last_updated: datetime
    status: str  # "whitelist", "watchlist", "blacklist"


class V2AssetQualitySystem:
    """
    Manages asset quality and filters predictions.
    
    Rules:
    1. Whitelist: Win rate >= 55%, predict freely
    2. Watchlist: Win rate 45-55%, predict cautiously (high confidence only)
    3. Blacklist: Win rate < 45%, DO NOT predict
    
    Updates daily based on rolling 30-day performance.
    """
    
    def __init__(self):
        self.config_file = "ghost_v2_quality.json"
        self._whitelist: Set[str] = set()
        self._blacklist: Set[str] = set()
        self._metrics: Dict[str, AssetQualityMetrics] = {}
        
        # V2 Configuration
        self.MIN_PREDICTIONS_FOR_EVAL = int(os.getenv("V2_MIN_PREDICTIONS", "20"))
        self.WHITELIST_WIN_RATE = float(os.getenv("V2_WHITELIST_WR", "0.55"))  # 55%
        self.BLACKLIST_WIN_RATE = float(os.getenv("V2_BLACKLIST_WR", "0.45"))  # 45%
        self.WATCHLIST_HIGH_CONFIDENCE = float(os.getenv("V2_WATCHLIST_CONF", "0.80"))  # 80%
        
        # Load existing config
        self._load_config()
        
        LOGGER.info(f"[V2-QUALITY] Initialized: {len(self._whitelist)} whitelist, {len(self._blacklist)} blacklist")
    
    def _load_config(self):
        """Load saved whitelist/blacklist from disk"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self._whitelist = set(data.get('whitelist', []))
                    self._blacklist = set(data.get('blacklist', []))
                    
                    # Load metrics
                    for symbol, metrics in data.get('metrics', {}).items():
                        metrics['last_updated'] = datetime.fromisoformat(metrics['last_updated'])
                        self._metrics[symbol] = AssetQualityMetrics(**metrics)
                    
                    LOGGER.info(f"[V2-QUALITY] Loaded config: {len(self._whitelist)} whitelist, {len(self._blacklist)} blacklist")
        except Exception as e:
            LOGGER.warning(f"[V2-QUALITY] Failed to load config: {e} - starting fresh")
    
    def _save_config(self):
        """Save whitelist/blacklist to disk"""
        try:
            data = {
                'whitelist': list(self._whitelist),
                'blacklist': list(self._blacklist),
                'metrics': {
                    symbol: {
                        **asdict(metrics),
                        'last_updated': metrics.last_updated.isoformat()
                    }
                    for symbol, metrics in self._metrics.items()
                },
                'last_updated': datetime.utcnow().isoformat(),
                'config': {
                    'min_predictions': self.MIN_PREDICTIONS_FOR_EVAL,
                    'whitelist_wr': self.WHITELIST_WIN_RATE,
                    'blacklist_wr': self.BLACKLIST_WIN_RATE
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            LOGGER.info(f"[V2-QUALITY] Config saved")
        except Exception as e:
            LOGGER.error(f"[V2-QUALITY] Failed to save config: {e}")
    
    def update_from_verification(self, days: int = 30):
        """
        Update whitelist/blacklist based on verified performance.
        Should be run daily (automated) or manually.
        """
        from core.v2_verification import get_verifier
        
        LOGGER.info(f"[V2-QUALITY] Updating quality metrics from last {days} days...")
        
        verifier = get_verifier()
        performances = verifier.get_symbol_performance(days, self.MIN_PREDICTIONS_FOR_EVAL)
        
        new_whitelist = set()
        new_blacklist = set()
        
        for perf in performances:
            # Update metrics
            self._metrics[perf.symbol] = AssetQualityMetrics(
                symbol=perf.symbol,
                win_rate=perf.win_rate / 100.0,  # Convert to decimal
                total_predictions=perf.total_predictions,
                recent_trend=perf.recent_performance,
                avg_confidence=perf.avg_confidence,
                last_updated=datetime.utcnow(),
                status=""  # Will set below
            )
            
            # Determine status
            if perf.win_rate >= self.WHITELIST_WIN_RATE * 100 and perf.recent_performance != "declining":
                new_whitelist.add(perf.symbol)
                self._metrics[perf.symbol].status = "whitelist"
            elif perf.win_rate < self.BLACKLIST_WIN_RATE * 100:
                new_blacklist.add(perf.symbol)
                self._metrics[perf.symbol].status = "blacklist"
            else:
                # Watchlist (45-55% win rate)
                self._metrics[perf.symbol].status = "watchlist"
        
        # Update sets
        old_whitelist = len(self._whitelist)
        old_blacklist = len(self._blacklist)
        
        self._whitelist = new_whitelist
        self._blacklist = new_blacklist
        
        # Save to disk
        self._save_config()
        
        LOGGER.info(f"[V2-QUALITY] Update complete:")
        LOGGER.info(f"  Whitelist: {old_whitelist} → {len(self._whitelist)} ({len(new_whitelist - set(list(self._whitelist)[:old_whitelist]))} added)")
        LOGGER.info(f"  Blacklist: {old_blacklist} → {len(self._blacklist)} ({len(new_blacklist - set(list(self._blacklist)[:old_blacklist]))} added)")
    
    def should_predict(self, symbol: str, confidence: float) -> tuple[bool, str]:
        """
        Determine if we should make a prediction for this symbol.
        
        Returns:
            (should_predict: bool, reason: str)
        
        Rules:
        1. Blacklist → NEVER predict
        2. Whitelist → Predict freely
        3. Watchlist → Only if confidence >= 80%
        4. Unknown → Predict cautiously (confidence >= 75%)
        """
        # Check blacklist
        if symbol in self._blacklist:
            return False, f"blacklisted (historical WR < {self.BLACKLIST_WIN_RATE*100:.0f}%)"
        
        # Check whitelist
        if symbol in self._whitelist:
            return True, "whitelisted (proven performer)"
        
        # Watchlist (not whitelist, not blacklist, but we have data)
        if symbol in self._metrics:
            metrics = self._metrics[symbol]
            if metrics.status == "watchlist":
                if confidence >= self.WATCHLIST_HIGH_CONFIDENCE:
                    return True, f"watchlist (WR {metrics.win_rate*100:.0f}%, high confidence)"
                else:
                    return False, f"watchlist (needs {self.WATCHLIST_HIGH_CONFIDENCE*100:.0f}%+ confidence)"
        
        # Unknown asset (no historical data)
        if confidence >= 0.75:
            return True, "unknown asset (high confidence only)"
        else:
            return False, "unknown asset (needs 75%+ confidence)"
    
    def get_quality_filter_stats(self) -> Dict[str, any]:
        """Get current filter statistics"""
        watchlist_count = sum(1 for m in self._metrics.values() if m.status == "watchlist")
        
        return {
            "whitelist_count": len(self._whitelist),
            "blacklist_count": len(self._blacklist),
            "watchlist_count": watchlist_count,
            "total_tracked": len(self._metrics),
            "whitelist": sorted(list(self._whitelist)),
            "blacklist": sorted(list(self._blacklist)),
            "config": {
                "min_predictions": self.MIN_PREDICTIONS_FOR_EVAL,
                "whitelist_wr_threshold": f"{self.WHITELIST_WIN_RATE*100:.0f}%",
                "blacklist_wr_threshold": f"{self.BLACKLIST_WIN_RATE*100:.0f}%",
                "watchlist_min_confidence": f"{self.WATCHLIST_HIGH_CONFIDENCE*100:.0f}%"
            }
        }
    
    def get_asset_metrics(self, symbol: str) -> Optional[AssetQualityMetrics]:
        """Get quality metrics for a specific asset"""
        return self._metrics.get(symbol)
    
    def get_all_metrics(self) -> Dict[str, AssetQualityMetrics]:
        """Get all asset metrics"""
        return self._metrics.copy()


# ============================================================================
# Singleton
# ============================================================================

_quality_system: Optional[V2AssetQualitySystem] = None

def get_quality_system() -> V2AssetQualitySystem:
    """Get singleton quality system"""
    global _quality_system
    if _quality_system is None:
        _quality_system = V2AssetQualitySystem()
    return _quality_system


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    quality = get_quality_system()
    
    print("\n" + "=" * 70)
    print("🎯 GHOST PROTOCOL V2 - ASSET QUALITY SYSTEM")
    print("=" * 70)
    
    # Update from verification data
    print("\n📊 Updating quality metrics from verification system...")
    quality.update_from_verification(30)
    
    # Show stats
    stats = quality.get_quality_filter_stats()
    print(f"\n✅ WHITELIST ({stats['whitelist_count']}): Proven performers (WR >= 55%)")
    print(f"   {', '.join(stats['whitelist'][:20])}")
    if stats['whitelist_count'] > 20:
        print(f"   ... and {stats['whitelist_count'] - 20} more")
    
    print(f"\n⚠️  WATCHLIST ({stats['watchlist_count']}): Predict cautiously (WR 45-55%, need 80%+ confidence)")
    
    print(f"\n❌ BLACKLIST ({stats['blacklist_count']}): Do NOT predict (WR < 45%)")
    print(f"   {', '.join(stats['blacklist'])}")
    
    # Test prediction filter
    print("\n🧪 TESTING PREDICTION FILTER:")
    test_cases = [
        ("BTC", 0.85),
        ("BTC", 0.60),
        ("UNKNOWN_SYMBOL", 0.90),
        ("UNKNOWN_SYMBOL", 0.70),
    ]
    
    for symbol, conf in test_cases:
        should, reason = quality.should_predict(symbol, conf)
        emoji = "✅" if should else "❌"
        print(f"   {emoji} {symbol} @ {conf*100:.0f}% confidence: {reason}")
    
    print("\n" + "=" * 70)

#!/usr/bin/env python3
"""
🎯 GHOST PROTOCOL V2 - ASSET QUALITY SYSTEM

Phase 2: Find the Edge
- Track which assets Ghost predicts well
- Dynamic whitelist/blacklist based on performance
- Quality gates for predictions

Only predict assets where we have demonstrated edge.

PostgreSQL Persistence:
- Whitelist/blacklist stored in PostgreSQL for Railway persistence
- JSON file used as fallback for local dev
- Auto-sync on save to both storage backends
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict

LOGGER = logging.getLogger("ghost.v2_quality")

# PostgreSQL support for production (matches paper_tracker.py pattern)
DATABASE_URL = os.getenv("DATABASE_URL")


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
        self._trial_stocks: Set[str] = set()  # Learning tier for stocks
        self._metrics: Dict[str, AssetQualityMetrics] = {}
        self._config: Dict = {}  # Store full config for trial_stock_min_confidence etc
        
        # V2 Configuration
        self.MIN_PREDICTIONS_FOR_EVAL = int(os.getenv("V2_MIN_PREDICTIONS", "20"))
        self.WHITELIST_WIN_RATE = float(os.getenv("V2_WHITELIST_WR", "0.55"))  # 55%
        self.BLACKLIST_WIN_RATE = float(os.getenv("V2_BLACKLIST_WR", "0.45"))  # 45%
        self.WATCHLIST_HIGH_CONFIDENCE = float(os.getenv("V2_WATCHLIST_CONF", "0.80"))  # 80%
        self.TRIAL_STOCKS_MIN_CONFIDENCE = float(os.getenv("V2_TRIAL_CONF", "0.70"))  # 70%
        
        # PostgreSQL support
        self.use_postgres = bool(DATABASE_URL)
        self._ensure_postgres_table()
        
        # Load existing config (PostgreSQL first, JSON fallback)
        self._load_config()
        
        # Auto-scheduler for daily updates (backup for cron)
        self._scheduler_running = False
        
        LOGGER.info(f"[V2-QUALITY] Initialized: {len(self._whitelist)} whitelist, {len(self._blacklist)} blacklist, {len(self._trial_stocks)} trial_stocks (postgres={self.use_postgres})")
    
    @property
    def whitelist(self) -> Set[str]:
        """Public read-only access to whitelist"""
        return self._whitelist
    
    @property
    def blacklist(self) -> Set[str]:
        """Public read-only access to blacklist"""
        return self._blacklist
    
    @property
    def metrics(self) -> Dict[str, AssetQualityMetrics]:
        """Public read-only access to metrics"""
        return self._metrics
    
    def _get_postgres_connection(self):
        """Get PostgreSQL connection via shared pool bridge."""
        from core.db_pool import get_sync_connection
        return get_sync_connection().__enter__()
    
    def _ensure_postgres_table(self):
        """Create v2_quality_config table if not exists"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS v2_quality_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            conn.commit()
            conn.close()
            LOGGER.info("[V2-QUALITY] PostgreSQL table ready")
        except Exception as e:
            LOGGER.warning(f"[V2-QUALITY] Failed to create PostgreSQL table: {e}")
    
    def _load_from_postgres(self) -> Optional[dict]:
        """Load config from PostgreSQL"""
        if not self.use_postgres:
            return None
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            cur.execute("SELECT value FROM v2_quality_config WHERE key = 'config'")
            row = cur.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            LOGGER.warning(f"[V2-QUALITY] Failed to load from PostgreSQL: {e}")
            return None
    
    def _save_to_postgres(self, data: dict):
        """Save config to PostgreSQL"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            value_json = json.dumps(data)
            cur.execute("""
                INSERT INTO v2_quality_config (key, value, updated_at)
                VALUES ('config', %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
            """, (value_json, value_json))
            conn.commit()
            conn.close()
            LOGGER.info("[V2-QUALITY] Saved to PostgreSQL")
        except Exception as e:
            LOGGER.error(f"[V2-QUALITY] Failed to save to PostgreSQL: {e}")
    
    def _load_config(self):
        """Load saved whitelist/blacklist - PostgreSQL first, JSON fallback"""
        # Try PostgreSQL first (production)
        pg_data = self._load_from_postgres()
        if pg_data:
            self._whitelist = set(pg_data.get('whitelist', []))
            self._blacklist = set(pg_data.get('blacklist', []))
            self._trial_stocks = set(pg_data.get('trial_stocks', []))
            self._config = pg_data.get('config', {})
            for symbol, metrics in pg_data.get('metrics', {}).items():
                metrics['last_updated'] = datetime.fromisoformat(metrics['last_updated'])
                self._metrics[symbol] = AssetQualityMetrics(**metrics)
            LOGGER.info(f"[V2-QUALITY] Loaded from PostgreSQL: {len(self._whitelist)} whitelist, {len(self._blacklist)} blacklist, {len(self._trial_stocks)} trial_stocks")
            return
        
        # Fallback to JSON file
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self._whitelist = set(data.get('whitelist', []))
                    self._blacklist = set(data.get('blacklist', []))
                    self._trial_stocks = set(data.get('trial_stocks', []))
                    self._config = data.get('config', {})
                    
                    # Load metrics
                    for symbol, metrics in data.get('metrics', {}).items():
                        metrics['last_updated'] = datetime.fromisoformat(metrics['last_updated'])
                        self._metrics[symbol] = AssetQualityMetrics(**metrics)
                    
                    LOGGER.info(f"[V2-QUALITY] Loaded config: {len(self._whitelist)} whitelist, {len(self._blacklist)} blacklist, {len(self._trial_stocks)} trial_stocks")
        except Exception as e:
            LOGGER.warning(f"[V2-QUALITY] Failed to load config: {e} - starting fresh")
    
    def _save_config(self, pinned_whitelist: set = None):
        """Save whitelist/blacklist/trial_stocks to both PostgreSQL (primary) and JSON (backup)"""
        try:
            data = {
                'whitelist': sorted(list(self._whitelist)),
                'blacklist': sorted(list(self._blacklist)),
                'trial_stocks': sorted(list(self._trial_stocks)),  # NEW: Save trial stocks
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
                    'blacklist_wr': self.BLACKLIST_WIN_RATE,
                    'trial_stock_min_confidence': self._config.get('trial_stock_min_confidence', 0.70)
                }
            }
            
            # Preserve pinned_whitelist if provided (manual curation)
            if pinned_whitelist:
                data['pinned_whitelist'] = sorted(list(pinned_whitelist))
            
            # Save to PostgreSQL (primary - survives deploys)
            self._save_to_postgres(data)
            
            # Also save to JSON (backup/local dev)
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            LOGGER.info(f"[V2-QUALITY] Config saved to PostgreSQL and JSON (trial_stocks={len(self._trial_stocks)})")
        except Exception as e:
            LOGGER.error(f"[V2-QUALITY] Failed to save config: {e}")
    
    def update_from_verification(self, days: int = 30):
        """
        Update whitelist/blacklist based on verified performance.
        Should be run daily (automated) or manually.
        
        IMPORTANT: Pinned symbols (manually curated) are NEVER removed from whitelist.
        IMPORTANT: Manual blacklist entries are ALWAYS preserved.
        """
        from core.v2_verification import get_verifier
        
        LOGGER.info(f"[V2-QUALITY] Updating quality metrics from last {days} days...")
        
        # Load pinned whitelist AND manual blacklist from JSON (curated lists that survive auto-updates)
        pinned_whitelist = set()
        pinned_blacklist = set()  # NEW: Manual blacklist entries to preserve
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    pinned_whitelist = set(data.get('pinned_whitelist', data.get('whitelist', [])))
                    pinned_blacklist = set(data.get('blacklist', []))  # Preserve ALL JSON blacklist entries
                    LOGGER.info(f"[V2-QUALITY] Preserving {len(pinned_whitelist)} pinned whitelist, {len(pinned_blacklist)} pinned blacklist")
        except Exception as e:
            LOGGER.warning(f"[V2-QUALITY] Could not load pinned lists: {e}")
        
        verifier = get_verifier()
        performances = verifier.get_symbol_performance(days, self.MIN_PREDICTIONS_FOR_EVAL)
        
        new_whitelist = set(pinned_whitelist)  # Start with pinned symbols
        new_blacklist = set(pinned_blacklist)  # NEW: Start with manual blacklist
        
        for perf in performances:
            # Update metrics (convert Decimal to float)
            wr_decimal = float(perf.win_rate) / 100.0  # perf.win_rate is percentage (0-100)
            
            self._metrics[perf.symbol] = AssetQualityMetrics(
                symbol=perf.symbol,
                win_rate=wr_decimal,  # Store as decimal (0.0-1.0)
                total_predictions=perf.total_predictions,
                recent_trend=perf.recent_performance,
                avg_confidence=perf.avg_confidence,
                last_updated=datetime.utcnow(),
                status=""  # Will set below
            )
            
            # Determine status (compare percentage values)
            wr_pct = float(perf.win_rate)  # win_rate is 0-100
            
            # Skip pinned whitelist symbols - they stay whitelisted regardless of performance
            if perf.symbol in pinned_whitelist:
                self._metrics[perf.symbol].status = "whitelist (pinned)"
                continue
            
            # Skip pinned blacklist symbols - they stay blacklisted regardless of performance
            if perf.symbol in pinned_blacklist:
                self._metrics[perf.symbol].status = "blacklist (pinned)"
                continue
            
            if wr_pct >= self.WHITELIST_WIN_RATE * 100 and perf.recent_performance != "declining":
                new_whitelist.add(perf.symbol)
                self._metrics[perf.symbol].status = "whitelist"
            elif wr_pct < self.BLACKLIST_WIN_RATE * 100:
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
        
        # Save to disk (preserve pinned_whitelist)
        self._save_config(pinned_whitelist)
        
        LOGGER.info(f"[V2-QUALITY] Update complete:")
        LOGGER.info(f"  Whitelist: {old_whitelist} → {len(self._whitelist)} (includes {len(pinned_whitelist)} pinned)")
        LOGGER.info(f"  Blacklist: {old_blacklist} → {len(self._blacklist)}")
    
    def should_predict(self, symbol: str, confidence: float) -> tuple[bool, str]:
        """
        Determine if we should make a prediction for this symbol.
        
        Returns:
            (should_predict: bool, reason: str)
        
        Rules:
        1. Blacklist → NEVER predict
        2. Whitelist → Predict freely
        3. Trial Stocks → Only if confidence >= 70% (learning mode)
        4. Watchlist → Only if confidence >= 80%
        5. Unknown → Predict cautiously (confidence >= 75%)
        """
        # Check blacklist
        if symbol in self._blacklist:
            return False, f"blacklisted (historical WR < {self.BLACKLIST_WIN_RATE*100:.0f}%)"
        
        # Check whitelist
        if symbol in self._whitelist:
            return True, "whitelisted (proven performer)"
        
        # Check trial stocks (new learning tier for stocks)
        if symbol in self._trial_stocks:
            trial_min_conf = float(self._config.get("trial_stock_min_confidence", 0.70))
            if confidence >= trial_min_conf:
                return True, f"trial stock (learning mode, {confidence:.0%} >= {trial_min_conf:.0%})"
            else:
                return False, f"trial stock (needs {trial_min_conf:.0%}+ confidence, got {confidence:.0%})"
        
        # Watchlist (not whitelist, not blacklist, but we have data)
        if symbol in self._metrics:
            metrics = self._metrics[symbol]
            if metrics.status == "watchlist":
                if confidence >= self.WATCHLIST_HIGH_CONFIDENCE:
                    return True, f"watchlist (WR {metrics.win_rate*100:.0f}%, high confidence)"
                else:
                    return False, f"watchlist (needs {self.WATCHLIST_HIGH_CONFIDENCE*100:.0f}%+ confidence)"
        
        # Unknown asset (no historical data)
        # V2 STRICT MODE: Only predict whitelisted symbols
        # Reject all unknown symbols until we have performance data
        return False, "not whitelisted (V2 strict mode: whitelist-only predictions)"
    
    def get_quality_filter_stats(self) -> Dict[str, any]:
        """Get current filter statistics"""
        watchlist_count = sum(1 for m in self._metrics.values() if m.status == "watchlist")
        
        return {
            "whitelist_count": len(self._whitelist),
            "blacklist_count": len(self._blacklist),
            "watchlist_count": watchlist_count,
            "trial_stocks_count": len(self._trial_stocks),
            "total_tracked": len(self._metrics),
            "whitelist": sorted(list(self._whitelist)),
            "blacklist": sorted(list(self._blacklist)),
            "trial_stocks": sorted(list(self._trial_stocks)),
            "config": {
                "min_predictions": self.MIN_PREDICTIONS_FOR_EVAL,
                "whitelist_wr_threshold": f"{self.WHITELIST_WIN_RATE*100:.0f}%",
                "blacklist_wr_threshold": f"{self.BLACKLIST_WIN_RATE*100:.0f}%",
                "watchlist_min_confidence": f"{self.WATCHLIST_HIGH_CONFIDENCE*100:.0f}%",
                "trial_stocks_confidence": f"{self.TRIAL_STOCKS_MIN_CONFIDENCE*100:.0f}%"
            }
        }
    
    def get_asset_metrics(self, symbol: str) -> Optional[AssetQualityMetrics]:
        """Get quality metrics for a specific asset"""
        return self._metrics.get(symbol)
    
    def get_all_metrics(self) -> Dict[str, AssetQualityMetrics]:
        """Get all asset metrics"""
        return self._metrics.copy()
    
    def start_auto_update_scheduler(self, interval_hours: int = 24):
        """
        Start background scheduler for automatic quality updates.
        
        Runs daily at startup + every interval_hours.
        Use cron-job.org for more reliable scheduling via API endpoint.
        
        Args:
            interval_hours: How often to update (default: 24 hours)
        """
        if self._scheduler_running:
            LOGGER.info("[V2-QUALITY] Auto-scheduler already running")
            return
        
        def _run_scheduler():
            import time
            
            # Wait a bit on startup to let everything initialize
            time.sleep(60)  # 1 minute after startup
            
            while self._scheduler_running:
                try:
                    LOGGER.info("[V2-QUALITY] 🔄 Running scheduled quality update...")
                    self.update_from_verification(days=30)
                    LOGGER.info("[V2-QUALITY] ✅ Scheduled quality update complete")
                except Exception as e:
                    LOGGER.error(f"[V2-QUALITY] ❌ Scheduled update failed: {e}")
                
                # Sleep for interval
                for _ in range(interval_hours * 60):  # Check every minute for graceful shutdown
                    if not self._scheduler_running:
                        break
                    time.sleep(60)
        
        self._scheduler_running = True
        scheduler_thread = threading.Thread(target=_run_scheduler, daemon=True, name="V2QualityScheduler")
        scheduler_thread.start()
        LOGGER.info(f"[V2-QUALITY] 🚀 Auto-scheduler started (every {interval_hours}h)")
    
    def stop_auto_update_scheduler(self):
        """Stop the background scheduler"""
        self._scheduler_running = False
        LOGGER.info("[V2-QUALITY] Auto-scheduler stopped")


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

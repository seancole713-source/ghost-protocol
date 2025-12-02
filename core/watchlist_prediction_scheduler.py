#!/usr/bin/env python3
"""
Ghost Protocol Watchlist Prediction Scheduler
==============================================

Ensures watchlist symbols get predicted on schedule:
- Daily: market open + close for stocks
- Intraday: big-move detection triggers
- Continuous: crypto (reuses AUTO-PREDICT cycles)

Integration:
- Uses services/predictor.py for prediction generation
- Marks watchlist symbols as high-priority (no skipping)
- Records tracking data in watchlist_prediction_tracking table
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Environment configuration
WATCHLIST_SCHEDULER_ENABLED = os.getenv("WATCHLIST_SCHEDULER_ENABLED", "1") == "1"
WATCHLIST_OPEN_HOUR = int(os.getenv("WATCHLIST_OPEN_HOUR", "9"))  # 9 AM EST market open
WATCHLIST_CLOSE_HOUR = int(os.getenv("WATCHLIST_CLOSE_HOUR", "16"))  # 4 PM EST market close
WATCHLIST_BIG_MOVE_CHECK_MINUTES = int(os.getenv("WATCHLIST_BIG_MOVE_CHECK_MINUTES", "15"))
WATCHLIST_BIG_MOVE_THRESHOLD_PCT = float(os.getenv("WATCHLIST_BIG_MOVE_THRESHOLD_PCT", "5.0"))


class WatchlistPredictionScheduler:
    """
    Schedules prediction generation for personal watchlist symbols.
    """

    def __init__(self):
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.last_open_check = 0.0
        self.last_close_check = 0.0
        self.last_big_move_check = 0.0

    def start(self):
        """Start the scheduler background thread."""
        if not WATCHLIST_SCHEDULER_ENABLED:
            LOGGER.info("🔕 Watchlist scheduler disabled (WATCHLIST_SCHEDULER_ENABLED=0)")
            return

        if self.running:
            LOGGER.warning("⚠️  Watchlist scheduler already running")
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="watchlist-scheduler")
        self.scheduler_thread.start()
        LOGGER.info("🚀 Watchlist prediction scheduler started")

    def stop(self):
        """Stop the scheduler background thread."""
        if not self.running:
            return

        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        LOGGER.info("🛑 Watchlist prediction scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop (runs in background thread)."""
        LOGGER.info("📅 Watchlist scheduler loop active")

        while self.running:
            try:
                now = time.time()

                # Check market open (once per day)
                if self._should_run_open_check(now):
                    self._run_market_open_predictions()
                    self.last_open_check = now

                # Check market close (once per day)
                if self._should_run_close_check(now):
                    self._run_market_close_predictions()
                    self.last_close_check = now

                # Check big moves (every N minutes)
                if now - self.last_big_move_check > (WATCHLIST_BIG_MOVE_CHECK_MINUTES * 60):
                    self._run_big_move_detection()
                    self.last_big_move_check = now

                # Sleep before next iteration
                time.sleep(60)  # Check every minute

            except Exception as e:
                LOGGER.error(f"❌ Watchlist scheduler error: {e}", exc_info=True)
                time.sleep(60)

    def _should_run_open_check(self, now: float) -> bool:
        """Check if we should run market open predictions."""
        # Prevent running multiple times per day
        if now - self.last_open_check < (6 * 3600):  # 6 hour cooldown
            return False

        # Check if current time is near market open hour
        current_hour = datetime.now().hour
        return current_hour == WATCHLIST_OPEN_HOUR

    def _should_run_close_check(self, now: float) -> bool:
        """Check if we should run market close predictions."""
        # Prevent running multiple times per day
        if now - self.last_close_check < (6 * 3600):  # 6 hour cooldown
            return False

        # Check if current time is near market close hour
        current_hour = datetime.now().hour
        return current_hour == WATCHLIST_CLOSE_HOUR

    def _run_market_open_predictions(self):
        """Generate predictions for all watchlist stocks at market open."""
        LOGGER.info("🔔 Running market open predictions for watchlist stocks...")

        try:
            from core.personal_watchlist import get_personal_watchlist_manager

            pwm = get_personal_watchlist_manager()
            stock_symbols = pwm.get_symbols_by_type("stock", active_only=True)

            LOGGER.info(f"📊 {len(stock_symbols)} stocks in watchlist")

            for symbol in stock_symbols:
                try:
                    self._generate_prediction(symbol, "stock", reason="market_open")
                except Exception as e:
                    LOGGER.error(f"❌ Market open prediction failed for {symbol}: {e}")

            LOGGER.info(f"✅ Market open predictions complete ({len(stock_symbols)} stocks)")

        except Exception as e:
            LOGGER.error(f"❌ Market open predictions failed: {e}", exc_info=True)

    def _run_market_close_predictions(self):
        """Generate predictions for all watchlist stocks at market close."""
        LOGGER.info("🔔 Running market close predictions for watchlist stocks...")

        try:
            from core.personal_watchlist import get_personal_watchlist_manager

            pwm = get_personal_watchlist_manager()
            stock_symbols = pwm.get_symbols_by_type("stock", active_only=True)

            LOGGER.info(f"📊 {len(stock_symbols)} stocks in watchlist")

            for symbol in stock_symbols:
                try:
                    self._generate_prediction(symbol, "stock", reason="market_close")
                except Exception as e:
                    LOGGER.error(f"❌ Market close prediction failed for {symbol}: {e}")

            LOGGER.info(f"✅ Market close predictions complete ({len(stock_symbols)} stocks)")

        except Exception as e:
            LOGGER.error(f"❌ Market close predictions failed: {e}", exc_info=True)

    def _run_big_move_detection(self):
        """Detect big price moves and generate predictions for affected symbols."""
        LOGGER.debug("🔍 Checking for big price moves in watchlist...")

        try:
            from core.personal_watchlist import get_personal_watchlist_manager

            pwm = get_personal_watchlist_manager()

            # First, update price snapshots for all active watchlist items
            watchlist = pwm.get_watchlist(active_only=True)

            for item in watchlist:
                try:
                    self._record_price_snapshot(item)
                except Exception as e:
                    LOGGER.debug(f"Could not record price snapshot for {item['symbol']}: {e}")

            # Detect big moves
            big_movers = pwm.detect_big_moves(lookback_minutes=WATCHLIST_BIG_MOVE_CHECK_MINUTES)

            if big_movers:
                LOGGER.info(f"🚨 {len(big_movers)} big movers detected in watchlist")

                for mover in big_movers:
                    symbol = mover["symbol"]
                    asset_type = mover["asset_type"]
                    move_pct = mover["move_pct"]

                    LOGGER.info(f"📈 {symbol} ({asset_type}) moved {move_pct:+.2f}% (threshold: ±{mover['threshold_pct']}%)")

                    # Generate fresh prediction
                    try:
                        self._generate_prediction(symbol, asset_type, reason="big_move")
                    except Exception as e:
                        LOGGER.error(f"❌ Big move prediction failed for {symbol}: {e}")

        except Exception as e:
            LOGGER.error(f"❌ Big move detection failed: {e}", exc_info=True)

    def _record_price_snapshot(self, watchlist_item: Dict[str, Any]):
        """
        Record a price snapshot for a watchlist item.

        Args:
            watchlist_item: Dict with id, symbol, asset_type
        """
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()
        symbol = watchlist_item["symbol"]
        asset_type = watchlist_item["asset_type"]

        # Fetch live price
        price = pwm._fetch_live_price(symbol, asset_type)

        if price:
            pwm.record_price_snapshot(
                watchlist_item_id=watchlist_item["id"],
                symbol=symbol,
                price=price,
                change_pct_24h=None,  # TODO: Calculate from previous snapshot
                volume_24h=None,
            )

    def _generate_prediction(self, symbol: str, asset_type: str, reason: str):
        """
        Generate a prediction for a watchlist symbol and track it.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            reason: 'market_open', 'market_close', 'big_move', 'manual'
        """
        try:
            from services.predictor import predict_symbol
            from core.personal_watchlist import get_personal_watchlist_manager

            # Generate prediction
            result = predict_symbol(symbol)

            if not result.get("ok"):
                LOGGER.warning(f"⚠️  Prediction failed for {symbol}: {result.get('error', 'unknown')}")
                return

            # Extract prediction data
            prediction_id = result.get("prediction_id")
            direction = result.get("direction", "FLAT")
            confidence = result.get("confidence", 0.0)
            expected_move_pct = result.get("expected_move_pct", 0.0)
            horizon_h = result.get("horizon_h", 48)
            current_price = result.get("current_price")

            # Track in watchlist system
            pwm = get_personal_watchlist_manager()
            watchlist_items = pwm.get_watchlist(active_only=True)

            # Find matching watchlist item
            watchlist_item_id = None
            for item in watchlist_items:
                if item["symbol"] == symbol and item["asset_type"] == asset_type:
                    watchlist_item_id = item["id"]
                    break

            if watchlist_item_id:
                pwm.track_prediction(
                    watchlist_item_id=watchlist_item_id,
                    symbol=symbol,
                    prediction_id=prediction_id,
                    direction=direction,
                    confidence=confidence,
                    expected_move_pct=expected_move_pct,
                    horizon_h=horizon_h,
                    price_at_prediction=current_price,
                    reason=reason,
                )

                LOGGER.info(f"✅ Prediction tracked for {symbol} (reason: {reason}, dir: {direction}, conf: {confidence:.0%})")
            else:
                LOGGER.warning(f"⚠️  Watchlist item not found for {symbol} ({asset_type})")

        except Exception as e:
            LOGGER.error(f"❌ Failed to generate prediction for {symbol}: {e}", exc_info=True)

    def trigger_manual_prediction(self, symbol: str, asset_type: str) -> Dict[str, Any]:
        """
        Manually trigger a prediction for a watchlist symbol (API endpoint use).

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'

        Returns:
            Dict with prediction result
        """
        try:
            self._generate_prediction(symbol, asset_type, reason="manual")
            return {"ok": True, "symbol": symbol, "reason": "manual"}
        except Exception as e:
            LOGGER.error(f"❌ Manual prediction trigger failed for {symbol}: {e}")
            return {"ok": False, "error": str(e)}


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_WATCHLIST_PREDICTION_SCHEDULER = None


def get_watchlist_prediction_scheduler() -> WatchlistPredictionScheduler:
    """Get singleton instance of WatchlistPredictionScheduler."""
    global _WATCHLIST_PREDICTION_SCHEDULER
    if _WATCHLIST_PREDICTION_SCHEDULER is None:
        _WATCHLIST_PREDICTION_SCHEDULER = WatchlistPredictionScheduler()
    return _WATCHLIST_PREDICTION_SCHEDULER


def start_watchlist_scheduler():
    """Start the watchlist prediction scheduler (call at app startup)."""
    scheduler = get_watchlist_prediction_scheduler()
    scheduler.start()


def stop_watchlist_scheduler():
    """Stop the watchlist prediction scheduler (call at app shutdown)."""
    scheduler = get_watchlist_prediction_scheduler()
    scheduler.stop()

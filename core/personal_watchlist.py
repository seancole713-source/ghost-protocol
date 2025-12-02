#!/usr/bin/env python3
"""
Ghost Protocol Personal Watchlist Manager
==========================================

Single-owner persistent watchlist for stocks and crypto with prediction tracking.

Features:
- Postgres-backed persistence (via db_engine unified interface)
- Manual add/remove from Cockpit UI
- Prediction tracking (daily + intraday)
- Telegram alert integration
- Big-move detection
- Position tracking (owns_position flag)

Architecture:
- Uses ghost_watchlist_items table (not the legacy watchlist.db)
- Integrates with prediction_store for enriched data
- Works with existing telegram_hunter alert pipeline
- Single-owner (no multi-tenant auth)
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.db_engine import execute_query, execute_many, get_db_connection

LOGGER = logging.getLogger(__name__)


class PersonalWatchlistManager:
    """
    Manages single-owner persistent watchlist with prediction tracking.
    """

    def __init__(self):
        """Initialize watchlist manager."""
        self._ensure_schema()

    def _ensure_schema(self):
        """Verify schema exists (migration should have run first)."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as cnt
                    FROM information_schema.tables
                    WHERE table_name = 'ghost_watchlist_items'
                """)
                result = cursor.fetchone()
                if result and result[0] == 0:
                    LOGGER.warning("⚠️  ghost_watchlist_items table not found - run migration first")
        except Exception as e:
            LOGGER.debug(f"Schema check skipped: {e}")

    # ========================================================================
    # CORE CRUD OPERATIONS
    # ========================================================================

    def add_symbol(
        self,
        symbol: str,
        asset_type: str,
        owns_position: bool = False,
        notes: str = "",
        alert_threshold_pct: float = 5.0,
        priority: int = 1,
    ) -> Dict[str, Any]:
        """
        Add symbol to watchlist (or re-activate if previously soft-deleted).

        Args:
            symbol: Ticker symbol (will be uppercased)
            asset_type: 'crypto' or 'stock'
            owns_position: TRUE if user currently holds this asset
            notes: Optional notes/comments
            alert_threshold_pct: Price move % to trigger big-move alert (default 5.0)
            priority: 1=normal, 2=high, 3=critical

        Returns:
            Dict with item details or error
        """
        symbol = symbol.upper().strip()
        asset_type = asset_type.lower().strip()

        if asset_type not in ("crypto", "stock"):
            return {"ok": False, "error": f"Invalid asset_type: {asset_type} (must be 'crypto' or 'stock')"}

        if not symbol or len(symbol) > 20:
            return {"ok": False, "error": "Symbol must be 1-20 characters"}

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Check if symbol already exists (active or inactive)
                cursor.execute(
                    """
                    SELECT id, active, owns_position, priority, alert_threshold_pct
                    FROM ghost_watchlist_items
                    WHERE symbol = %s AND asset_type = %s
                    """,
                    (symbol, asset_type),
                )
                existing = cursor.fetchone()

                if existing:
                    # Re-activate if inactive, otherwise update
                    item_id = existing[0]
                    was_active = existing[1]

                    cursor.execute(
                        """
                        UPDATE ghost_watchlist_items
                        SET active = TRUE,
                            owns_position = %s,
                            notes = %s,
                            alert_threshold_pct = %s,
                            priority = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (owns_position, notes, alert_threshold_pct, priority, item_id),
                    )

                    action = "re-activated" if not was_active else "updated"
                    LOGGER.info(f"✅ {symbol} ({asset_type}) {action} in watchlist")

                    return {
                        "ok": True,
                        "action": action,
                        "id": item_id,
                        "symbol": symbol,
                        "asset_type": asset_type,
                        "owns_position": owns_position,
                    }
                else:
                    # Insert new item
                    cursor.execute(
                        """
                        INSERT INTO ghost_watchlist_items
                        (symbol, asset_type, owns_position, notes, alert_threshold_pct, priority)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id, added_at
                        """,
                        (symbol, asset_type, owns_position, notes, alert_threshold_pct, priority),
                    )
                    result = cursor.fetchone()
                    item_id = result[0]
                    added_at = result[1]

                    LOGGER.info(f"✅ {symbol} ({asset_type}) added to watchlist (id={item_id})")

                    return {
                        "ok": True,
                        "action": "added",
                        "id": item_id,
                        "symbol": symbol,
                        "asset_type": asset_type,
                        "owns_position": owns_position,
                        "added_at": str(added_at),
                    }

        except Exception as e:
            LOGGER.error(f"❌ Failed to add {symbol} to watchlist: {e}")
            return {"ok": False, "error": str(e)}

    def remove_symbol(self, symbol: str, asset_type: str) -> Dict[str, Any]:
        """
        Soft-delete symbol from watchlist (sets active=FALSE).

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'

        Returns:
            Dict with success status
        """
        symbol = symbol.upper().strip()
        asset_type = asset_type.lower().strip()

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE ghost_watchlist_items
                    SET active = FALSE, updated_at = NOW()
                    WHERE symbol = %s AND asset_type = %s AND active = TRUE
                    """,
                    (symbol, asset_type),
                )

                if cursor.rowcount > 0:
                    LOGGER.info(f"✅ {symbol} ({asset_type}) removed from watchlist")
                    return {"ok": True, "symbol": symbol, "asset_type": asset_type}
                else:
                    return {"ok": False, "error": f"{symbol} not found in active watchlist"}

        except Exception as e:
            LOGGER.error(f"❌ Failed to remove {symbol} from watchlist: {e}")
            return {"ok": False, "error": str(e)}

    def get_watchlist(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all watchlist items.

        Args:
            active_only: If TRUE, only return active items (default)

        Returns:
            List of watchlist items with metadata
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                where_clause = "WHERE active = TRUE" if active_only else ""
                cursor.execute(
                    f"""
                    SELECT id, symbol, asset_type, owns_position, notes,
                           alert_threshold_pct, priority, added_at, updated_at
                    FROM ghost_watchlist_items
                    {where_clause}
                    ORDER BY priority DESC, added_at DESC
                    """
                )

                items = []
                for row in cursor.fetchall():
                    items.append(
                        {
                            "id": row[0],
                            "symbol": row[1],
                            "asset_type": row[2],
                            "owns_position": row[3],
                            "notes": row[4],
                            "alert_threshold_pct": row[5],
                            "priority": row[6],
                            "added_at": str(row[7]) if row[7] else None,
                            "updated_at": str(row[8]) if row[8] else None,
                        }
                    )

                return items

        except Exception as e:
            LOGGER.error(f"❌ Failed to get watchlist: {e}")
            return []

    def get_symbols_by_type(self, asset_type: str, active_only: bool = True) -> List[str]:
        """
        Get list of symbols filtered by asset type.

        Args:
            asset_type: 'crypto' or 'stock'
            active_only: If TRUE, only return active symbols

        Returns:
            List of symbol strings
        """
        items = self.get_watchlist(active_only=active_only)
        return [item["symbol"] for item in items if item["asset_type"] == asset_type]

    def update_position_flag(self, symbol: str, asset_type: str, owns_position: bool) -> Dict[str, Any]:
        """
        Update the owns_position flag for a symbol.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            owns_position: TRUE if user now holds this asset

        Returns:
            Dict with success status
        """
        symbol = symbol.upper().strip()
        asset_type = asset_type.lower().strip()

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE ghost_watchlist_items
                    SET owns_position = %s, updated_at = NOW()
                    WHERE symbol = %s AND asset_type = %s AND active = TRUE
                    """,
                    (owns_position, symbol, asset_type),
                )

                if cursor.rowcount > 0:
                    LOGGER.info(f"✅ {symbol} owns_position updated to {owns_position}")
                    return {"ok": True, "symbol": symbol, "owns_position": owns_position}
                else:
                    return {"ok": False, "error": f"{symbol} not found in active watchlist"}

        except Exception as e:
            LOGGER.error(f"❌ Failed to update owns_position for {symbol}: {e}")
            return {"ok": False, "error": str(e)}

    # ========================================================================
    # ENRICHED DATA (WITH PREDICTIONS)
    # ========================================================================

    def get_enriched_watchlist(self) -> List[Dict[str, Any]]:
        """
        Get watchlist with live predictions and price data.

        Returns:
            List of items with current_price, prediction (direction, confidence, expected_move)
        """
        items = self.get_watchlist(active_only=True)

        # Import prediction store
        try:
            from core.prediction_store import get_prediction_store

            pred_store = get_prediction_store()
        except ImportError:
            LOGGER.warning("Prediction store not available")
            pred_store = None

        enriched = []
        for item in items:
            symbol = item["symbol"]
            asset_type = item["asset_type"]

            # Get latest prediction
            prediction = None
            current_price = None

            if pred_store:
                try:
                    pred_dict = pred_store.get_latest_prediction(symbol)
                    if pred_dict:
                        prediction = {
                            "prediction_id": pred_dict.get("id"),
                            "direction": pred_dict.get("direction"),
                            "confidence": pred_dict.get("confidence"),
                            "expected_move": pred_dict.get("expected_move_pct", 0.0),
                            "horizon_h": pred_dict.get("horizon_h", 48),
                            "run_at": pred_dict.get("run_at"),
                        }
                        current_price = pred_dict.get("current_price")
                except Exception as e:
                    LOGGER.debug(f"Could not fetch prediction for {symbol}: {e}")

            # If no prediction or no price, try to fetch live price
            if current_price is None:
                current_price = self._fetch_live_price(symbol, asset_type)

            enriched.append(
                {
                    **item,
                    "current_price": current_price,
                    "prediction": prediction,
                }
            )

        return enriched

    def _fetch_live_price(self, symbol: str, asset_type: str) -> Optional[float]:
        """
        Fetch live price for symbol using turbo providers.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'

        Returns:
            Price as float or None
        """
        try:
            if asset_type == "crypto":
                from core.providers.turbo_provider import turbo_crypto_price

                result = turbo_crypto_price(symbol, max_budget_s=3.0)
                if result.get("ok") and result.get("price"):
                    return float(result["price"])
            else:
                from core.providers.turbo_provider import turbo_stock_price

                result = turbo_stock_price(symbol, max_budget_s=5.0)
                if result.get("ok") and result.get("price"):
                    return float(result["price"])
        except Exception as e:
            LOGGER.debug(f"Could not fetch live price for {symbol}: {e}")

        return None

    # ========================================================================
    # PREDICTION TRACKING
    # ========================================================================

    def track_prediction(
        self,
        watchlist_item_id: int,
        symbol: str,
        prediction_id: int,
        direction: str,
        confidence: float,
        expected_move_pct: float,
        horizon_h: int,
        price_at_prediction: Optional[float],
        reason: str,
    ) -> int:
        """
        Record a prediction generation event for watchlist tracking.

        Args:
            watchlist_item_id: FK to ghost_watchlist_items
            symbol: Ticker symbol
            prediction_id: FK to ghost_predictions
            direction: UP/DOWN/FLAT
            confidence: 0.0-1.0
            expected_move_pct: Expected price change %
            horizon_h: Prediction horizon in hours
            price_at_prediction: Price when prediction was made
            reason: 'market_open', 'market_close', 'big_move', 'manual'

        Returns:
            Tracking record ID
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO watchlist_prediction_tracking
                    (watchlist_item_id, symbol, prediction_id, direction, confidence,
                     expected_move_pct, horizon_h, price_at_prediction, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        watchlist_item_id,
                        symbol,
                        prediction_id,
                        direction,
                        confidence,
                        expected_move_pct,
                        horizon_h,
                        price_at_prediction,
                        reason,
                    ),
                )

                tracking_id = cursor.fetchone()[0]
                LOGGER.debug(f"✅ Prediction tracked for {symbol} (reason: {reason}, id: {tracking_id})")
                return tracking_id

        except Exception as e:
            LOGGER.error(f"❌ Failed to track prediction for {symbol}: {e}")
            return -1

    def get_prediction_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get prediction history for a watchlist symbol.

        Args:
            symbol: Ticker symbol
            limit: Max number of records

        Returns:
            List of prediction tracking records
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, prediction_id, direction, confidence, expected_move_pct,
                           horizon_h, price_at_prediction, generated_at, reason, alert_sent
                    FROM watchlist_prediction_tracking
                    WHERE symbol = %s
                    ORDER BY generated_at DESC
                    LIMIT %s
                    """,
                    (symbol.upper(), limit),
                )

                history = []
                for row in cursor.fetchall():
                    history.append(
                        {
                            "id": row[0],
                            "prediction_id": row[1],
                            "direction": row[2],
                            "confidence": row[3],
                            "expected_move_pct": row[4],
                            "horizon_h": row[5],
                            "price_at_prediction": row[6],
                            "generated_at": str(row[7]) if row[7] else None,
                            "reason": row[8],
                            "alert_sent": row[9],
                        }
                    )

                return history

        except Exception as e:
            LOGGER.error(f"❌ Failed to get prediction history for {symbol}: {e}")
            return []

    # ========================================================================
    # PRICE SNAPSHOTS & BIG MOVE DETECTION
    # ========================================================================

    def record_price_snapshot(
        self, watchlist_item_id: int, symbol: str, price: float, change_pct_24h: Optional[float], volume_24h: Optional[float]
    ) -> int:
        """
        Record a price snapshot for big-move detection.

        Args:
            watchlist_item_id: FK to ghost_watchlist_items
            symbol: Ticker symbol
            price: Current price
            change_pct_24h: 24h price change %
            volume_24h: 24h volume

        Returns:
            Snapshot record ID
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO watchlist_price_snapshots
                    (watchlist_item_id, symbol, price, change_pct_24h, volume_24h)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (watchlist_item_id, symbol, price, change_pct_24h, volume_24h),
                )

                snapshot_id = cursor.fetchone()[0]
                return snapshot_id

        except Exception as e:
            LOGGER.error(f"❌ Failed to record price snapshot for {symbol}: {e}")
            return -1

    def detect_big_moves(self, lookback_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Detect symbols with significant price moves in the last N minutes.

        Args:
            lookback_minutes: Time window for move detection (default 60 min)

        Returns:
            List of symbols with big moves and their details
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get watchlist items with their alert thresholds
                cursor.execute(
                    """
                    SELECT w.id, w.symbol, w.asset_type, w.alert_threshold_pct,
                           w.owns_position, w.priority
                    FROM ghost_watchlist_items w
                    WHERE w.active = TRUE
                    """
                )

                watchlist_items = cursor.fetchall()
                big_movers = []

                for item in watchlist_items:
                    item_id, symbol, asset_type, threshold_pct, owns_position, priority = item

                    # Get price snapshots from last N minutes
                    cursor.execute(
                        """
                        SELECT price, snapshot_at
                        FROM watchlist_price_snapshots
                        WHERE watchlist_item_id = %s
                          AND snapshot_at >= NOW() - INTERVAL '%s minutes'
                        ORDER BY snapshot_at ASC
                        """,
                        (item_id, lookback_minutes),
                    )

                    snapshots = cursor.fetchall()

                    if len(snapshots) >= 2:
                        first_price = snapshots[0][0]
                        last_price = snapshots[-1][0]

                        if first_price and last_price and first_price > 0:
                            move_pct = ((last_price - first_price) / first_price) * 100.0

                            if abs(move_pct) >= threshold_pct:
                                big_movers.append(
                                    {
                                        "watchlist_item_id": item_id,
                                        "symbol": symbol,
                                        "asset_type": asset_type,
                                        "price_start": first_price,
                                        "price_current": last_price,
                                        "move_pct": move_pct,
                                        "threshold_pct": threshold_pct,
                                        "owns_position": owns_position,
                                        "priority": priority,
                                        "lookback_minutes": lookback_minutes,
                                    }
                                )

                return big_movers

        except Exception as e:
            LOGGER.error(f"❌ Failed to detect big moves: {e}")
            return []

    # ========================================================================
    # ALERT LOGGING
    # ========================================================================

    def log_alert(
        self,
        watchlist_item_id: int,
        symbol: str,
        alert_type: str,
        direction: Optional[str],
        confidence: Optional[float],
        expected_move_pct: Optional[float],
        current_price: Optional[float],
        change_pct: Optional[float],
        message: str,
        telegram_sent: bool = False,
        telegram_chat_id: Optional[int] = None,
    ) -> int:
        """
        Log an alert sent for a watchlist symbol.

        Args:
            watchlist_item_id: FK to ghost_watchlist_items
            symbol: Ticker symbol
            alert_type: 'open', 'close', 'big_move', 'target_hit'
            direction: UP/DOWN/FLAT (prediction direction)
            confidence: Prediction confidence
            expected_move_pct: Expected move %
            current_price: Price at alert time
            change_pct: Price change %
            message: Alert message text
            telegram_sent: TRUE if delivered to Telegram
            telegram_chat_id: Telegram chat ID

        Returns:
            Alert log record ID
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO watchlist_alerts_log
                    (watchlist_item_id, symbol, alert_type, direction, confidence,
                     expected_move_pct, current_price, change_pct, message,
                     telegram_sent, telegram_chat_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        watchlist_item_id,
                        symbol,
                        alert_type,
                        direction,
                        confidence,
                        expected_move_pct,
                        current_price,
                        change_pct,
                        message,
                        telegram_sent,
                        telegram_chat_id,
                    ),
                )

                alert_id = cursor.fetchone()[0]
                LOGGER.info(f"✅ Alert logged for {symbol} (type: {alert_type}, id: {alert_id})")
                return alert_id

        except Exception as e:
            LOGGER.error(f"❌ Failed to log alert for {symbol}: {e}")
            return -1

    def check_alert_cooldown(self, symbol: str, alert_type: str, cooldown_hours: int = 4) -> bool:
        """
        Check if an alert can be sent (cooldown enforcement).

        Args:
            symbol: Ticker symbol
            alert_type: 'open', 'close', 'big_move', 'target_hit'
            cooldown_hours: Minimum hours between alerts of same type

        Returns:
            TRUE if alert can be sent (no recent alert of this type)
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT COUNT(*) as cnt
                    FROM watchlist_alerts_log
                    WHERE symbol = %s
                      AND alert_type = %s
                      AND created_at >= NOW() - INTERVAL '%s hours'
                    """,
                    (symbol.upper(), alert_type, cooldown_hours),
                )

                result = cursor.fetchone()
                recent_count = result[0] if result else 0

                return recent_count == 0

        except Exception as e:
            LOGGER.error(f"❌ Failed to check alert cooldown for {symbol}: {e}")
            return True  # Fail open (allow alert)

    def get_alert_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get alert statistics for the last N days.

        Args:
            days: Lookback period

        Returns:
            Dict with alert counts by type
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT alert_type, COUNT(*) as cnt
                    FROM watchlist_alerts_log
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY alert_type
                    """,
                    (days,),
                )

                stats = {"total": 0, "by_type": {}}
                for row in cursor.fetchall():
                    alert_type = row[0]
                    count = row[1]
                    stats["by_type"][alert_type] = count
                    stats["total"] += count

                return stats

        except Exception as e:
            LOGGER.error(f"❌ Failed to get alert stats: {e}")
            return {"total": 0, "by_type": {}}


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_PERSONAL_WATCHLIST_MANAGER = None


def get_personal_watchlist_manager() -> PersonalWatchlistManager:
    """Get singleton instance of PersonalWatchlistManager."""
    global _PERSONAL_WATCHLIST_MANAGER
    if _PERSONAL_WATCHLIST_MANAGER is None:
        _PERSONAL_WATCHLIST_MANAGER = PersonalWatchlistManager()
    return _PERSONAL_WATCHLIST_MANAGER

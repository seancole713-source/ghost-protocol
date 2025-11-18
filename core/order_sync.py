#!/usr/bin/env python3
"""
📊 GHOST ORDER STATUS SYNC - Background Task

Syncs order statuses from Alpaca and updates local database.
Sends notifications when orders fill.

Usage:
    python3 core/order_sync.py

Or run as background task in wolf_app.py:
    asyncio.create_task(start_order_sync())
"""

import asyncio
import logging
import os
import sqlite3
import time
from typing import Any

LOGGER = logging.getLogger("ghost.order_sync")

# Configuration
SYNC_INTERVAL_SECONDS = int(os.getenv("ORDER_SYNC_INTERVAL", "30"))  # Sync every 30 seconds
ENABLED = os.getenv("ORDER_SYNC_ENABLED", "1") == "1"
WOLF_SQLITE_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
ORDERS_TABLE = os.getenv("ORDERS_TABLE", "orders")


async def sync_order_status(order_id: str, broker_id: str) -> dict[str, Any] | None:
    """
    Fetch latest status for a single order from Alpaca.
    
    Args:
        order_id: Local order ID
        broker_id: Alpaca order ID
        
    Returns:
        Updated order data or None if error
    """
    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return None

        # Get order from Alpaca
        order = broker.get_order(broker_id)
        
        if order:
            LOGGER.debug(
                f"Synced order {order_id[:8]}... → status={order.get('status')}, "
                f"filled_qty={order.get('filled_qty', 0)}/{order.get('qty', 0)}"
            )
        
        return order

    except Exception as e:
        LOGGER.error(f"Failed to sync order {order_id}: {e}")
        return None


async def update_local_order(order_id: str, broker_order: dict[str, Any]) -> bool:
    """
    Update local database with latest order data from broker.
    
    Args:
        order_id: Local order ID
        broker_order: Order data from Alpaca
        
    Returns:
        True if updated successfully
    """
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()

        # Extract fields
        status = broker_order.get("status", "unknown")
        filled_qty = float(broker_order.get("filled_qty", 0))
        filled_avg_price = float(broker_order.get("filled_avg_price", 0)) if broker_order.get("filled_avg_price") else None
        updated_at = broker_order.get("updated_at", "")

        # Update order in database
        cur.execute(
            f"""
            UPDATE {ORDERS_TABLE}
            SET status = ?,
                filled_qty = ?,
                filled_avg_price = ?,
                updated_at = ?,
                last_synced = ?
            WHERE id = ?
            """,
            (
                status,
                filled_qty,
                filled_avg_price,
                updated_at,
                time.time(),
                order_id,
            ),
        )

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        LOGGER.error(f"Failed to update local order {order_id}: {e}")
        return False


async def check_for_fills(order_id: str, old_status: str, new_status: str, order_data: dict[str, Any]):
    """
    Check if order was filled and send notification.
    
    Args:
        order_id: Local order ID
        old_status: Previous status from database
        new_status: Current status from Alpaca
        order_data: Full order data from Alpaca
    """
    # Check if status changed to filled
    if old_status != "filled" and new_status == "filled":
        symbol = order_data.get("symbol", "???")
        side = order_data.get("side", "???").upper()
        qty = order_data.get("filled_qty", 0)
        avg_price = float(order_data.get("filled_avg_price", 0))
        
        LOGGER.info(
            f"✅ ORDER FILLED: {side} {qty} {symbol} @ ${avg_price:.2f} (order_id={order_id[:8]}...)"
        )

        # Add event to database
        try:
            from wolf_app import _add_event

            _add_event(
                "order.filled",
                f"{side} {qty} {symbol} filled @ ${avg_price:.2f}",
                {
                    "order_id": order_id,
                    "broker_id": order_data.get("id"),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "avg_price": avg_price,
                    "timestamp": int(time.time()),
                },
            )
        except Exception:
            pass

        # Send Telegram notification
        try:
            from core.telegram_alerts import send_alert

            await send_alert(
                f"🎯 Trade Executed\n\n"
                f"{side} {qty} {symbol}\n"
                f"Price: ${avg_price:.2f}\n"
                f"Total: ${qty * avg_price:,.2f}",
                priority="high",
            )
        except Exception as e:
            LOGGER.warning(f"Failed to send Telegram notification: {e}")

    # Check if partially filled
    elif old_status != "partially_filled" and new_status == "partially_filled":
        filled_qty = order_data.get("filled_qty", 0)
        total_qty = order_data.get("qty", 0)
        symbol = order_data.get("symbol", "???")
        
        LOGGER.info(f"⚡ PARTIAL FILL: {filled_qty}/{total_qty} {symbol} (order_id={order_id[:8]}...)")


async def sync_pending_orders():
    """
    Sync all pending/open orders from database.
    
    Returns:
        Number of orders synced
    """
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get all non-terminal orders
        cur.execute(
            f"""
            SELECT id, broker_id, status
            FROM {ORDERS_TABLE}
            WHERE status NOT IN ('filled', 'canceled', 'expired', 'rejected', 'stopped')
            AND broker = 'alpaca'
            AND broker_id IS NOT NULL
            ORDER BY ts DESC
            LIMIT 100
            """
        )

        rows = cur.fetchall()
        conn.close()

        if not rows:
            LOGGER.debug("No pending orders to sync")
            return 0

        synced_count = 0

        for row in rows:
            order_id = row["id"]
            broker_id = row["broker_id"]
            old_status = row["status"]

            # Fetch latest from Alpaca
            broker_order = await sync_order_status(order_id, broker_id)

            if broker_order:
                new_status = broker_order.get("status", old_status)

                # Update local database
                await update_local_order(order_id, broker_order)

                # Check for fills and notify
                await check_for_fills(order_id, old_status, new_status, broker_order)

                synced_count += 1

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

        if synced_count > 0:
            LOGGER.info(f"Synced {synced_count} pending order(s)")

        return synced_count

    except Exception as e:
        LOGGER.error(f"Error syncing pending orders: {e}", exc_info=True)
        return 0


async def order_sync_loop():
    """
    Main sync loop - runs continuously in background.
    Syncs pending orders every SYNC_INTERVAL_SECONDS.
    """
    LOGGER.info(f"📊 Order sync started (interval={SYNC_INTERVAL_SECONDS}s)")

    while True:
        try:
            # Check if sync is enabled
            if not ENABLED:
                LOGGER.debug("Order sync disabled via env var")
                await asyncio.sleep(SYNC_INTERVAL_SECONDS)
                continue

            # Sync pending orders
            await sync_pending_orders()

            # Wait before next sync
            await asyncio.sleep(SYNC_INTERVAL_SECONDS)

        except Exception as e:
            LOGGER.error(f"Error in order sync loop: {e}", exc_info=True)
            # Wait longer after error
            await asyncio.sleep(SYNC_INTERVAL_SECONDS * 2)


async def start_order_sync():
    """Start the order sync monitor as a background task."""
    if not ENABLED:
        LOGGER.info("Order sync is disabled (set ORDER_SYNC_ENABLED=1 to enable)")
        return

    # Run the sync loop
    await order_sync_loop()


if __name__ == "__main__":
    # Run standalone
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("📊 Starting Ghost Order Status Sync (standalone mode)")
    print(f"   Sync interval: {SYNC_INTERVAL_SECONDS}s")
    print(f"   Enabled: {ENABLED}")
    print()

    asyncio.run(order_sync_loop())

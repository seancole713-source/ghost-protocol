"""Live recalculator.

The orchestrator expects `live_recalculator_loop()`.

This module periodically snapshots live positions from the broker, computes
potential exit signals using the existing SL/TP logic, and stores the results
in a small sqlite DB under `data/live_recalculator.db`.

It does **not** place orders; actual exits remain the responsibility of
`core.sl_tp_monitor` and/or the execution engine.
"""

from __future__ import annotations

import asyncio
import sqlite3
import os
import time
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "live_recalculator.db"


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL,
            avg_entry_price REAL,
            current_price REAL,
            unrealized_pl REAL,
            unrealized_plpc REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS exit_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            type TEXT NOT NULL,
            reason TEXT,
            pnl_pct REAL,
            entry_price REAL,
            current_price REAL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_snap_ts ON position_snapshots(ts DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_exit_ts ON exit_signals(ts DESC)")
    conn.commit()
    conn.close()


def _store_snapshot(ts: int, positions: list[dict]) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    for p in positions:
        raw_plpc = float(p.get("unrealized_plpc") or 0)
        plpc_pct = raw_plpc * 100.0 if -1.0 <= raw_plpc <= 1.0 else raw_plpc
        cur.execute(
            """
            INSERT INTO position_snapshots(
                ts,symbol,qty,avg_entry_price,current_price,unrealized_pl,unrealized_plpc
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                ts,
                str(p.get("symbol") or ""),
                float(p.get("qty") or 0),
                float(p.get("avg_entry_price") or 0),
                float(p.get("current_price") or 0),
                float(p.get("unrealized_pl") or 0),
                plpc_pct,
            ),
        )
    conn.commit()
    conn.close()


def _store_exit_signals(ts: int, signals: list[dict]) -> None:
    if not signals:
        return
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    for s in signals:
        cur.execute(
            """
            INSERT INTO exit_signals(ts,symbol,type,reason,pnl_pct,entry_price,current_price)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                ts,
                str(s.get("symbol") or ""),
                str(s.get("type") or ""),
                str(s.get("reason") or ""),
                float(s.get("pnl_pct") or 0),
                float(s.get("entry_price") or 0),
                float(s.get("current_price") or s.get("exit_price") or 0),
            ),
        )
    conn.commit()
    conn.close()


def _send_message(text: str) -> None:
    try:
        from core.telegram_hunter import send_telegram_message

        send_telegram_message(text)
    except Exception:
        return


async def live_recalculator_loop() -> None:
    """Periodically recalculate live position guidance.

    Designed to be safe in constrained environments (Railway free tier).
    """
    interval_s = int(os.getenv("LIVE_RECALCULATOR_INTERVAL_S", "300"))
    alert_on_exit_signals = os.getenv("LIVE_RECALCULATOR_ALERTS", "1") == "1"

    _init_db()

    while True:
        try:
            from core.alpaca_broker import get_broker
            from core.risk_engine import get_risk_engine
            from core.sl_tp_monitor import check_positions_for_exits

            broker = get_broker()
            if not getattr(broker, "enabled", False):
                LOGGER.debug("live_recalculator_broker_disabled")
                await asyncio.sleep(max(5, interval_s))
                continue

            ts = int(time.time())
            positions = broker.get_positions() or []
            _store_snapshot(ts, positions)

            # Compute exit signals (does not execute)
            signals = await check_positions_for_exits()

            # Also run basic risk_engine SL/TP scan for coverage
            try:
                risk_positions = []
                for pos in positions:
                    risk_positions.append(
                        {
                            "symbol": pos.get("symbol"),
                            "qty": float(pos.get("qty", 0) or 0),
                            "avg_cost": float(pos.get("avg_entry_price", 0) or 0),
                            "current_price": float(pos.get("current_price", 0) or 0),
                        }
                    )
                signals.extend(get_risk_engine().scan_positions_for_exits(risk_positions))
            except Exception:
                pass

            # De-dupe signals by (symbol,type)
            dedup = {}
            for s in signals or []:
                k = f"{s.get('symbol')}::{s.get('type')}"
                dedup[k] = s
            signals = list(dedup.values())

            _store_exit_signals(ts, signals)

            if alert_on_exit_signals and signals:
                summary = ", ".join([f"{s.get('symbol')}({s.get('type')})" for s in signals[:8]])
                _send_message(f"⚠️ Exit signals detected: {summary}")

            LOGGER.info(
                "live_recalculator_updated",
                extra={"positions": len(positions), "exit_signals": len(signals)},
            )
        except Exception:
            LOGGER.exception("live_recalculator_error")

        await asyncio.sleep(max(5, interval_s))

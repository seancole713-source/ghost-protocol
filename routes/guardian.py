"""Routes: guardian — extracted from wolf_app.py (Step 12)"""

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
import httpx
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

try:
    from wolf_helpers import *
except ImportError:
    pass

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- Routes: guardian (2 endpoints) ---

try:
    import psycopg2
    import psycopg2.extras

    @router.get("/api/v3/guardian/alerts")
    async def get_guardian_alerts(
        symbol: str = None,
        severity: str = None,
        acknowledged: bool = None,
        limit: int = 50
    ):
        """
        Get guardian alerts with optional filters.
        
        Usage:
            GET /api/v3/guardian/alerts
            GET /api/v3/guardian/alerts?symbol=BTC&severity=CRITICAL
            GET /api/v3/guardian/alerts?acknowledged=false
        """
        try:
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                query = "SELECT * FROM guardian_alerts WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol.upper())
                if severity:
                    query += " AND severity = %s"
                    params.append(severity.upper())
                if acknowledged is not None:
                    query += " AND acknowledged = %s"
                    params.append(acknowledged)
                
                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                alerts = cur.fetchall()
            
            # Convert to serializable format
            result = []
            for a in alerts:
                alert_dict = dict(a)
                # Convert Decimal to float for JSON
                if alert_dict.get('price_at_alert'):
                    alert_dict['price_at_alert'] = float(alert_dict['price_at_alert'])
                if alert_dict.get('confidence'):
                    alert_dict['confidence'] = float(alert_dict['confidence'])
                # Convert datetime to ISO string
                if alert_dict.get('created_at'):
                    alert_dict['created_at'] = alert_dict['created_at'].isoformat()
                if alert_dict.get('acknowledged_at'):
                    alert_dict['acknowledged_at'] = alert_dict['acknowledged_at'].isoformat()
                result.append(alert_dict)
            
            return {"ok": True, "alerts": result, "count": len(result)}
        except Exception as e:
            LOGGER.error(f"Failed to get guardian alerts: {e}")
            return {"ok": False, "error": str(e), "alerts": []}

    @router.post("/api/v3/guardian/alert")
    async def create_guardian_alert(
        symbol: str,
        alert_type: str,
        message: str,
        severity: str = "INFO",
        price_at_alert: float = None,
        confidence: float = None,
        prediction_id: int = None,
        news_event_id: int = None
    ):
        """
        Create a guardian alert.
        
        Args:
            symbol: Trading symbol (e.g., BTC, AAPL)
            alert_type: Type of alert (e.g., NEWS_OVERRIDE, STOP_APPROACHING)
            message: Alert message
            severity: INFO, WARNING, HIGH, CRITICAL
            price_at_alert: Current price when alert was created
            confidence: Model confidence (0-1)
            prediction_id: Related prediction ID
            news_event_id: Related news event ID
        """
        try:
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO guardian_alerts 
                    (symbol, alert_type, severity, message, price_at_alert, confidence, prediction_id, news_event_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING alert_id
                """, (symbol.upper(), alert_type, severity.upper(), message, price_at_alert, confidence, prediction_id, news_event_id))
                
                alert_id = cur.fetchone()[0]
                
                LOGGER.info(f"[GUARDIAN] Created alert {alert_id} for {symbol}: {alert_type}")
                return {"ok": True, "alert_id": alert_id}
        except Exception as e:
            LOGGER.error(f"Failed to create guardian alert: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/v3/guardian/acknowledge/{alert_id}")
    async def acknowledge_guardian_alert(alert_id: int):
        """Acknowledge a guardian alert"""
        try:
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    UPDATE guardian_alerts 
                    SET acknowledged = TRUE, acknowledged_at = CURRENT_TIMESTAMP
                    WHERE alert_id = %s
                """, (alert_id,))
                
                return {"ok": True, "alert_id": alert_id, "acknowledged": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    LOGGER.info("✅ Guardian Alerts API endpoints registered (/api/v3/guardian/*)")


except Exception as _sec_e:
    LOGGER.warning(f'Route section error: {_sec_e}')

try:
    from core.guardian_oracle import get_guardian_oracle
    from core.guardian_heartbeat_scheduler import force_heartbeat

    @router.get("/api/v3/guardian/status")
    async def api_v3_guardian_status():
        """
        Get Guardian Oracle monitoring status.
        
        Returns:
            - Active positions being monitored
            - Alert history
            - Overall P&L
        """
        try:
            guardian = get_guardian_oracle()
            status = await guardian._get_current_status()
            
            return {
                "ok": True,
                "guardian_active": guardian.monitoring,
                "positions_active": status.get('active_count', 0),
                "positions_on_track": status.get('on_track', 0),
                "positions_weakened": status.get('weakened', 0),
                "total_pnl_pct": status.get('total_pnl', 0.0),
                "alert_count": len(guardian.alert_history)
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to get guardian status: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/v3/guardian/positions")
    async def api_v3_guardian_positions():
        """
        Get all positions Guardian is monitoring.
        
        Shows current status, P&L, confidence changes.
        """
        try:
            import sqlite3
            conn = sqlite3.connect("data/ghost_predictions.db")
            conn.row_factory = sqlite3.Row
            
            rows = conn.execute("""
                SELECT * FROM guardian_positions
                WHERE status = 'active'
                ORDER BY entry_time DESC
            """).fetchall()
            
            positions = [dict(row) for row in rows]
            conn.close()
            
            return {
                "ok": True,
                "count": len(positions),
                "positions": positions
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to get guardian positions: {e}")
            return {"ok": False, "error": str(e)}


    @router.post("/api/v3/guardian/heartbeat/{heartbeat_type}")
    async def api_v3_force_heartbeat(heartbeat_type: str):
        """
        Manually trigger a heartbeat for testing.
        
        Args:
            heartbeat_type: 'morning', 'midday', 'evening', or 'night'
        """
        try:
            if heartbeat_type not in ['morning', 'midday', 'evening', 'night']:
                return {
                    "ok": False,
                    "error": f"Invalid heartbeat type: {heartbeat_type}. Use: morning, midday, evening, night"
                }
            
            force_heartbeat(heartbeat_type)
            
            return {
                "ok": True,
                "message": f"Triggered {heartbeat_type} heartbeat",
                "heartbeat_type": heartbeat_type
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to force heartbeat: {e}")
            return {"ok": False, "error": str(e)}

    LOGGER.info("✅ Guardian Oracle endpoints registered (/api/v3/guardian/*)")
except Exception as _route_e:
    LOGGER.warning(f'Route section load error: {_route_e}')

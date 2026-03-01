"""
VIP Microcap Scanner
DISABLED (Mar 1, 2026): Presale coins (WEPE, LILPEPE, DORKL, SLOTH, APC)
are not available on any exchange. Scanner was importing dead providers
and sending Telegram alerts with wrong signature. Now a safe no-op.
"""

import logging
import os
import time
from typing import Any

LOGGER = logging.getLogger(__name__)

# VIP watchlist (disabled — presale coins not on exchanges)
VIP_WATCHLIST: list[str] = []

# Scanner configuration (kept for interface compatibility)
VIP_SCAN_INTERVAL_S = int(os.getenv("VIP_SCAN_INTERVAL_S", "60"))
VIP_VOLUME_SURGE_THRESHOLD = float(os.getenv("VIP_VOLUME_SURGE_THRESHOLD", "3.0"))
VIP_PRICE_SPIKE_THRESHOLD = float(os.getenv("VIP_PRICE_SPIKE_THRESHOLD", "10.0"))
VIP_ALERT_COOLDOWN_S = int(os.getenv("VIP_ALERT_COOLDOWN_S", "300"))

# State tracking
_VIP_LAST_PRICES: dict[str, float] = {}
_VIP_LAST_VOLUMES: dict[str, float] = {}
_VIP_LAST_ALERT: dict[str, float] = {}
_VIP_SCAN_COUNT = 0


def scan_vip_coins() -> dict[str, Any]:
    """
    Scan VIP coins for opportunities.
    DISABLED: VIP_WATCHLIST is empty — presale coins not on exchanges.
    Returns an empty result immediately.
    """
    global _VIP_SCAN_COUNT
    _VIP_SCAN_COUNT += 1

    if not VIP_WATCHLIST:
        return {
            "scanned": 0,
            "available": 0,
            "alerts_sent": 0,
            "opportunities": [],
            "scan_count": _VIP_SCAN_COUNT,
            "status": "disabled",
            "reason": "VIP watchlist empty — presale coins not on exchanges",
        }
    
    result = {
        "scanned": scanned,
        "available": available,
        "alerts_sent": alerts_sent,
        "opportunities": opportunities,
        "scan_count": _VIP_SCAN_COUNT,
        "timestamp": time.time()
    }
    
    LOGGER.info(
        f"VIP scan #{_VIP_SCAN_COUNT}: {available}/{scanned} available, "
        f"{len(opportunities)} opportunities, {alerts_sent} alerts sent"
    )
    
    return result


def get_vip_scanner_status() -> dict[str, Any]:
    """
    Get VIP scanner health and statistics
    
    Returns:
        {
            'enabled': True,
            'scan_count': 123,
            'watchlist': ['WEPE', 'LILPEPE', ...],
            'last_prices': {'WEPE': 0.00123, ...},
            'last_alerts': {'WEPE': 1731654000, ...}
        }
    """
    return {
        "enabled": True,
        "scan_count": _VIP_SCAN_COUNT,
        "watchlist": VIP_WATCHLIST,
        "last_prices": _VIP_LAST_PRICES.copy(),
        "last_volumes": _VIP_LAST_VOLUMES.copy(),
        "last_alerts": _VIP_LAST_ALERT.copy(),
        "config": {
            "scan_interval_s": VIP_SCAN_INTERVAL_S,
            "volume_surge_threshold": VIP_VOLUME_SURGE_THRESHOLD,
            "price_spike_threshold_pct": VIP_PRICE_SPIKE_THRESHOLD,
            "alert_cooldown_s": VIP_ALERT_COOLDOWN_S
        }
    }


def reset_vip_scanner() -> dict[str, str]:
    """Reset VIP scanner state (for testing)"""
    global _VIP_SCAN_COUNT
    _VIP_LAST_PRICES.clear()
    _VIP_LAST_VOLUMES.clear()
    _VIP_LAST_ALERT.clear()
    _VIP_SCAN_COUNT = 0
    
    return {"status": "reset", "message": "VIP scanner state cleared"}

"""
VIP Microcap Scanner
Priority 2: Hunt WEPE, LILPEPE, DORKL, SLOTH, APC for life-changing opportunities

Strategy:
1. Scan VIP coins every 60 seconds
2. Detect volume surges (3x+ average)
3. Detect price spikes (>10% in 1h)
4. Send Cash-App style alerts for significant moves
5. Track last alert time to avoid spam
"""

import logging
import os
import time
from typing import Any

from core.crypto.vip_providers import get_vip_price
from core.telegram_alerts import send_mover_alert

LOGGER = logging.getLogger(__name__)

# VIP watchlist (Priority microcaps)
VIP_WATCHLIST = ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC"]

# Scanner configuration
VIP_SCAN_INTERVAL_S = int(os.getenv("VIP_SCAN_INTERVAL_S", "60"))  # 60 seconds
VIP_VOLUME_SURGE_THRESHOLD = float(os.getenv("VIP_VOLUME_SURGE_THRESHOLD", "3.0"))  # 3x
VIP_PRICE_SPIKE_THRESHOLD = float(os.getenv("VIP_PRICE_SPIKE_THRESHOLD", "10.0"))  # 10%
VIP_ALERT_COOLDOWN_S = int(os.getenv("VIP_ALERT_COOLDOWN_S", "300"))  # 5 minutes

# State tracking
_VIP_LAST_PRICES: dict[str, float] = {}
_VIP_LAST_VOLUMES: dict[str, float] = {}
_VIP_LAST_ALERT: dict[str, float] = {}
_VIP_SCAN_COUNT = 0


def scan_vip_coins() -> dict[str, Any]:
    """
    Scan VIP coins for opportunities
    
    Returns:
        {
            'scanned': 5,
            'available': 3,
            'alerts_sent': 1,
            'opportunities': [
                {
                    'symbol': 'WEPE',
                    'price': 0.00123,
                    'change_1h_pct': 15.2,
                    'volume_surge': 4.5,
                    'alert_sent': True
                }
            ]
        }
    """
    global _VIP_SCAN_COUNT
    _VIP_SCAN_COUNT += 1
    
    scanned = 0
    available = 0
    alerts_sent = 0
    opportunities = []
    
    for symbol in VIP_WATCHLIST:
        scanned += 1
        
        # Get current price
        price_data = get_vip_price(symbol, use_cache=False)
        
        if not price_data.get("available"):
            LOGGER.debug(f"VIP scanner: {symbol} not available - {price_data.get('reason')}")
            continue
        
        available += 1
        current_price = price_data["price"]
        current_volume = price_data.get("volume_24h_usd", 0)
        change_24h_pct = price_data.get("change_24h_pct", 0)
        
        # Calculate 1-hour change (using last scan as proxy)
        last_price = _VIP_LAST_PRICES.get(symbol)
        change_1h_pct = 0.0
        
        if last_price and last_price > 0:
            change_1h_pct = ((current_price - last_price) / last_price) * 100
        
        # Calculate volume surge
        last_volume = _VIP_LAST_VOLUMES.get(symbol, current_volume)
        volume_surge = 0.0
        
        if last_volume and last_volume > 0:
            volume_surge = current_volume / last_volume
        
        # Update state
        _VIP_LAST_PRICES[symbol] = current_price
        _VIP_LAST_VOLUMES[symbol] = current_volume
        
        # Detect opportunities
        is_opportunity = False
        reason = None
        
        if abs(change_1h_pct) >= VIP_PRICE_SPIKE_THRESHOLD:
            is_opportunity = True
            reason = f"Price spike: {change_1h_pct:+.1f}% in 1h"
        elif volume_surge >= VIP_VOLUME_SURGE_THRESHOLD:
            is_opportunity = True
            reason = f"Volume surge: {volume_surge:.1f}x"
        
        if is_opportunity:
            # Check alert cooldown
            last_alert_time = _VIP_LAST_ALERT.get(symbol, 0)
            time_since_alert = time.time() - last_alert_time
            
            alert_sent = False
            if time_since_alert >= VIP_ALERT_COOLDOWN_S:
                # Send Cash-App style alert
                try:
                    send_mover_alert(
                        symbol=symbol,
                        market="crypto",
                        current_price=current_price,
                        change_pct=change_1h_pct if abs(change_1h_pct) > abs(change_24h_pct) else change_24h_pct,
                        volume=current_volume,
                        volume_avg=last_volume,
                        tier="VIP",
                        provider=price_data.get("provider", "unknown")
                    )
                    _VIP_LAST_ALERT[symbol] = time.time()
                    alerts_sent += 1
                    alert_sent = True
                    LOGGER.info(f"VIP alert sent for {symbol}: {reason}")
                except Exception as e:
                    LOGGER.error(f"Failed to send VIP alert for {symbol}: {e}")
            
            opportunities.append({
                "symbol": symbol,
                "price": current_price,
                "change_1h_pct": change_1h_pct,
                "change_24h_pct": change_24h_pct,
                "volume_surge": volume_surge,
                "reason": reason,
                "alert_sent": alert_sent
            })
    
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

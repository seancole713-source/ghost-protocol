"""
Ghost Runtime Configuration Manager
Handles dynamic runtime settings with defaults
"""

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)

# Default configuration values
_DEFAULT_CONFIG = {
    # Price provider settings
    'price_ttl_s': 30,
    'price_ttl_open_s': 300,
    'news_ttl_s': 300,
    'yahoo_first': True,
    'price_max_deviation_open': 0.15,
    
    # Data feeds
    'reuters_feeds_on': True,
    'overlay_enabled': True,
    'learning_enabled': True,
    
    # Diagnostics
    'diag_collapse_dupes': True,
    'diag_ring_size': 100,
    
    # Overlay
    'overlay_dt_minutes': 30,
    
    # Bands
    'band_widen_factor': 1.5,
    
    # Focus symbol
    'focus_symbol': 'SPY',  # Changed from WOLF to SPY (liquid, valid)
}

# Runtime configuration (mutable)
_runtime_config = _DEFAULT_CONFIG.copy()


def get_config(key: str = None) -> Any:
    """
    Get runtime configuration value
    
    Args:
        key: Config key (if None, returns all config)
    
    Returns:
        Config value or full config dict
    """
    if key is None:
        return _runtime_config.copy()
    return _runtime_config.get(key, _DEFAULT_CONFIG.get(key))


def set_config(key: str, value: Any) -> bool:
    """
    Set runtime configuration value
    
    Args:
        key: Config key
        value: New value
    
    Returns:
        True if successful
    """
    try:
        old_value = _runtime_config.get(key)
        _runtime_config[key] = value
        LOGGER.info(f"Config updated: {key} = {value} (was: {old_value})")
        return True
    except Exception as e:
        LOGGER.error(f"Failed to set config {key}: {e}")
        return False


def update_config(updates: dict[str, Any]) -> dict[str, bool]:
    """
    Update multiple configuration values
    
    Args:
        updates: Dict of key-value pairs to update
    
    Returns:
        Dict of {key: success_status}
    """
    results = {}
    for key, value in updates.items():
        results[key] = set_config(key, value)
    return results


def reset_config() -> None:
    """Reset configuration to defaults"""
    global _runtime_config
    _runtime_config = _DEFAULT_CONFIG.copy()
    LOGGER.info("Configuration reset to defaults")


def get_focus_symbol() -> str:
    """Get current focus symbol (defaults to SPY)"""
    return _runtime_config.get('focus_symbol', 'SPY')


def set_focus_symbol(symbol: str) -> bool:
    """
    Set focus symbol
    
    Args:
        symbol: New focus symbol (should be valid and liquid)
    
    Returns:
        True if successful
    """
    symbol = symbol.upper().strip()
    if not symbol:
        LOGGER.error("Cannot set empty focus symbol")
        return False
    
    return set_config('focus_symbol', symbol)

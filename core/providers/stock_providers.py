"""
Stock Price Providers Compatibility Layer
==========================================

Wrapper for existing price quorum system.
Provides interface expected by data pillar engines.
"""


def get_stock_price(symbol: str) -> dict:
    """
    Get stock price using wolf_app price quorum system.
    
    Args:
        symbol: Stock ticker (e.g., "AAPL")
    
    Returns:
        Dict with price data: {"price": float, "prev_close": float, "provider": str}
    """
    # Import here to avoid circular imports
    import os
    import sys
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Use the working price fetcher from wolf_app
    from wolf_app import _get_price_quorum
    
    result = _get_price_quorum(symbol, asset_type="stock")
    return result or {"price": None, "prev_close": None, "provider": "none"}

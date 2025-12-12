"""
🛡️ RISK MANAGER
Portfolio heat tracking, position sizing, correlation analysis, auto-hedge
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)

# Risk limits
MAX_PORTFOLIO_HEAT = 20.0  # Max 20% capital at risk
MAX_POSITION_SIZE = 5.0    # Max 5% per position
MAX_CORRELATED_EXPOSURE = 15.0  # Max 15% in correlated positions

# State
_ACTIVE_POSITIONS: dict[str, dict] = {}
_PORTFOLIO_HEAT = 0.0


# ============================================================================
# POSITION SIZING
# ============================================================================

def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    confidence: float
) -> dict:
    """
    Calculate optimal position size based on risk
    Uses Kelly Criterion with confidence adjustment
    """
    try:
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            risk_per_share = entry_price * 0.06  # Default 6% stop
        
        # Max risk per trade (confidence-weighted)
        max_risk_pct = min(2.0, confidence / 50)  # 1-2% risk based on confidence
        max_risk_dollars = account_balance * (max_risk_pct / 100)
        
        # Position size
        position_size = max_risk_dollars / risk_per_share
        
        # Cap at max position size (5% of account)
        max_position_dollars = account_balance * (MAX_POSITION_SIZE / 100)
        max_shares = max_position_dollars / entry_price
        
        position_size = min(position_size, max_shares)
        
        # Position value
        position_value = position_size * entry_price
        position_pct = (position_value / account_balance) * 100
        
        return {
            "shares": int(position_size),
            "position_value": position_value,
            "position_pct": position_pct,
            "risk_dollars": position_size * risk_per_share,
            "risk_pct": max_risk_pct
        }
        
    except Exception as e:
        LOGGER.error(f"Position sizing failed: {e}")
        return {
            "shares": 0,
            "position_value": 0.0,
            "position_pct": 0.0,
            "risk_dollars": 0.0,
            "risk_pct": 0.0
        }


# ============================================================================
# PORTFOLIO HEAT
# ============================================================================

def calculate_portfolio_heat(positions: list[dict]) -> float:
    """
    Calculate total portfolio heat (% capital at risk)
    """
    try:
        total_risk = sum(pos.get("risk_dollars", 0) for pos in positions)
        
        if not positions:
            return 0.0
        
        # Assume account balance from first position
        account_balance = positions[0].get("account_balance", 100000)
        
        heat = (total_risk / account_balance) * 100
        return min(100.0, heat)
        
    except Exception as e:
        LOGGER.error(f"Portfolio heat calculation failed: {e}")
        return 0.0


def check_heat_limit(current_heat: float, new_position_risk: float, account_balance: float) -> bool:
    """
    Check if adding new position would exceed heat limit
    """
    try:
        new_risk_pct = (new_position_risk / account_balance) * 100
        projected_heat = current_heat + new_risk_pct
        
        if projected_heat > MAX_PORTFOLIO_HEAT:
            LOGGER.warning(f"⚠️ Heat limit exceeded: {projected_heat:.1f}% (max {MAX_PORTFOLIO_HEAT}%)")
            return False
        
        return True
        
    except Exception as e:
        LOGGER.error(f"Heat limit check failed: {e}")
        return False


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

async def calculate_correlation(symbol1: str, symbol2: str, days: int = 30) -> float:
    """
    Calculate price correlation between two symbols (0-1)
    TODO: Fetch historical data and compute correlation coefficient
    """
    try:
        # Placeholder: Would fetch historical data and compute correlation
        # For now, return default correlation estimates
        
        # Same sector = high correlation
        if _get_sector(symbol1) == _get_sector(symbol2):
            return 0.7
        
        # Different asset classes = low correlation
        if (_is_crypto(symbol1) and not _is_crypto(symbol2)) or \
           (not _is_crypto(symbol1) and _is_crypto(symbol2)):
            return 0.3
        
        # Default moderate correlation
        return 0.5
        
    except Exception as e:
        LOGGER.error(f"Correlation calculation failed: {e}")
        return 0.5


def _get_sector(symbol: str) -> str:
    """Get sector for symbol (placeholder)"""
    # Tech stocks
    if symbol in ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]:
        return "TECH"
    # Crypto
    elif symbol in ["BTC", "ETH", "SOL"]:
        return "CRYPTO"
    # Default
    else:
        return "OTHER"


def _is_crypto(symbol: str) -> bool:
    """Check if symbol is crypto"""
    return symbol in ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"]


async def check_correlation_limit(new_symbol: str, existing_positions: list[dict]) -> dict:
    """
    Check if new position is too correlated with existing positions
    """
    try:
        if not existing_positions:
            return {"allowed": True, "reason": "No existing positions"}
        
        # Calculate correlations
        correlations = []
        
        for pos in existing_positions:
            existing_symbol = pos.get("symbol")
            corr = await calculate_correlation(new_symbol, existing_symbol)
            correlations.append({
                "symbol": existing_symbol,
                "correlation": corr
            })
        
        # Check highly correlated positions
        high_corr_positions = [c for c in correlations if c["correlation"] > 0.7]
        
        if len(high_corr_positions) >= 2:
            return {
                "allowed": False,
                "reason": f"Too many correlated positions: {[p['symbol'] for p in high_corr_positions]}"
            }
        
        return {"allowed": True, "reason": "Correlation within limits"}
        
    except Exception as e:
        LOGGER.error(f"Correlation limit check failed: {e}")
        return {"allowed": True, "reason": "Check failed, allowing"}


# ============================================================================
# AUTO-HEDGE
# ============================================================================

async def check_hedge_needed(portfolio_heat: float, market_regime: str) -> dict:
    """
    Determine if portfolio needs hedging
    """
    try:
        # Hedge if heat >15% in bear/crash regime
        if portfolio_heat > 15.0 and market_regime in ["BEAR", "CRASH"]:
            return {
                "hedge_needed": True,
                "hedge_type": "PUT_OPTIONS",
                "hedge_size": portfolio_heat * 0.5,  # Hedge 50% of risk
                "reason": f"High heat ({portfolio_heat:.1f}%) in {market_regime} market"
            }
        
        # Hedge if crash detected
        if market_regime == "CRASH":
            return {
                "hedge_needed": True,
                "hedge_type": "VIX_CALLS",
                "hedge_size": 5.0,  # 5% of portfolio in VIX calls
                "reason": "Market crash detected"
            }
        
        return {"hedge_needed": False}
        
    except Exception as e:
        LOGGER.error(f"Hedge check failed: {e}")
        return {"hedge_needed": False}


# ============================================================================
# RISK MONITORING
# ============================================================================

async def monitor_risk_loop():
    """
    Background loop to monitor portfolio risk
    """
    LOGGER.info("🚀 Risk Manager: STARTED")
    
    while True:
        try:
            # Get active positions
            from core.live_recalculator import get_active_picks
            positions = get_active_picks()
            
            if positions:
                # Calculate portfolio heat
                heat = calculate_portfolio_heat(positions)
                
                # Check if hedge needed
                from core.market_regime import detect_market_regime
                regime = await detect_market_regime()
                
                hedge = await check_hedge_needed(heat, regime["regime"])
                
                if hedge["hedge_needed"]:
                    LOGGER.warning(f"⚠️ HEDGE NEEDED: {hedge['reason']}")
                    # TODO: Execute hedge trade
                
                LOGGER.info(f"🛡️ Portfolio Heat: {heat:.1f}% (max {MAX_PORTFOLIO_HEAT}%)")
            
            # Check every 10 minutes
            await asyncio.sleep(600)
            
        except Exception as e:
            LOGGER.error(f"Risk monitor loop error: {e}")
            await asyncio.sleep(60)

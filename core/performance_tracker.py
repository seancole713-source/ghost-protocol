"""
📊 PERFORMANCE TRACKER
Win/loss tracking, confidence calibration, strategy performance analysis
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)

# Storage path
PERFORMANCE_FILE = Path("/workspaces/ghost-protocol/data/performance_log.json")

# State
_PERFORMANCE_DATA: dict[str, Any] = {
    "trades": [],
    "daily_stats": {},
    "strategy_stats": {},
    "confidence_calibration": {}
}


# ============================================================================
# TRADE LOGGING
# ============================================================================

def log_trade(trade: dict):
    """
    Log completed trade for performance tracking
    """
    try:
        trade_record = {
            "symbol": trade["symbol"],
            "asset_type": trade["asset_type"],
            "entry_time": trade["entry_time"],
            "exit_time": datetime.utcnow().isoformat(),
            "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"],
            "pnl_pct": trade["pnl_pct"],
            "pnl_dollars": trade.get("pnl_dollars", 0),
            "confidence": trade["confidence"],
            "expected_gain": trade["expected_gain"],
            "exit_reason": trade.get("exit_reason", "UNKNOWN"),
            "hold_time_minutes": trade.get("hold_time_minutes", 0)
        }
        
        _PERFORMANCE_DATA["trades"].append(trade_record)
        
        # Save to disk
        _save_performance_data()
        
        LOGGER.info(f"📝 Trade logged: ${trade['symbol']} {trade['pnl_pct']:+.1f}%")
        
    except Exception as e:
        LOGGER.error(f"Trade logging failed: {e}")


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_statistics() -> dict:
    """
    Calculate comprehensive performance statistics
    """
    try:
        trades = _PERFORMANCE_DATA["trades"]
        
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_gain": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0
            }
        
        # Win/Loss stats
        winners = [t for t in trades if t["pnl_pct"] > 0]
        losers = [t for t in trades if t["pnl_pct"] <= 0]
        
        win_rate = (len(winners) / len(trades)) * 100 if trades else 0
        
        avg_gain = sum(t["pnl_pct"] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t["pnl_pct"] for t in losers) / len(losers) if losers else 0
        
        total_gains = sum(t["pnl_pct"] for t in winners)
        total_losses = abs(sum(t["pnl_pct"] for t in losers))
        
        profit_factor = total_gains / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t["pnl_pct"] for t in trades]
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = (avg_return / std_dev) if std_dev > 0 else 0
        
        return {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "best_trade": max(t["pnl_pct"] for t in trades) if trades else 0,
            "worst_trade": min(t["pnl_pct"] for t in trades) if trades else 0
        }
        
    except Exception as e:
        LOGGER.error(f"Statistics calculation failed: {e}")
        return {}


# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

def calibrate_confidence() -> dict:
    """
    Analyze if confidence scores match actual win rates
    E.g., 70% confidence picks should win ~70% of time
    """
    try:
        trades = _PERFORMANCE_DATA["trades"]
        
        if not trades:
            return {}
        
        # Bucket trades by confidence (10% buckets)
        buckets = {i: [] for i in range(50, 101, 10)}
        
        for trade in trades:
            conf = trade["confidence"]
            bucket = (int(conf) // 10) * 10
            
            if bucket in buckets:
                buckets[bucket].append(trade)
        
        # Calculate actual win rate per bucket
        calibration = {}
        
        for bucket, bucket_trades in buckets.items():
            if bucket_trades:
                winners = [t for t in bucket_trades if t["pnl_pct"] > 0]
                actual_win_rate = (len(winners) / len(bucket_trades)) * 100
                
                calibration[f"{bucket}%"] = {
                    "trades": len(bucket_trades),
                    "expected_win_rate": bucket,
                    "actual_win_rate": actual_win_rate,
                    "calibration_error": abs(bucket - actual_win_rate)
                }
        
        return calibration
        
    except Exception as e:
        LOGGER.error(f"Confidence calibration failed: {e}")
        return {}


# ============================================================================
# STRATEGY ANALYSIS
# ============================================================================

def analyze_by_strategy() -> dict:
    """
    Break down performance by strategy type (stocks vs crypto, bullish vs bearish)
    """
    try:
        trades = _PERFORMANCE_DATA["trades"]
        
        if not trades:
            return {}
        
        # Group by asset type
        stocks = [t for t in trades if t["asset_type"] == "stock"]
        crypto = [t for t in trades if t["asset_type"] == "crypto"]
        
        # Group by exit reason
        exits = {}
        for trade in trades:
            reason = trade["exit_reason"]
            if reason not in exits:
                exits[reason] = []
            exits[reason].append(trade)
        
        return {
            "stocks": {
                "total": len(stocks),
                "win_rate": (len([t for t in stocks if t["pnl_pct"] > 0]) / len(stocks) * 100) if stocks else 0,
                "avg_pnl": sum(t["pnl_pct"] for t in stocks) / len(stocks) if stocks else 0
            },
            "crypto": {
                "total": len(crypto),
                "win_rate": (len([t for t in crypto if t["pnl_pct"] > 0]) / len(crypto) * 100) if crypto else 0,
                "avg_pnl": sum(t["pnl_pct"] for t in crypto) / len(crypto) if crypto else 0
            },
            "exit_reasons": {
                reason: {
                    "total": len(trades),
                    "avg_pnl": sum(t["pnl_pct"] for t in trades) / len(trades) if trades else 0
                }
                for reason, trades in exits.items()
            }
        }
        
    except Exception as e:
        LOGGER.error(f"Strategy analysis failed: {e}")
        return {}


# ============================================================================
# DAILY SUMMARY
# ============================================================================

def generate_daily_summary() -> dict:
    """
    Generate daily performance summary
    """
    try:
        today = datetime.utcnow().date().isoformat()
        
        trades = _PERFORMANCE_DATA["trades"]
        today_trades = [t for t in trades if t["entry_time"].startswith(today)]
        
        if not today_trades:
            return {
                "date": today,
                "trades": 0,
                "pnl_pct": 0.0,
                "win_rate": 0.0
            }
        
        winners = [t for t in today_trades if t["pnl_pct"] > 0]
        
        return {
            "date": today,
            "trades": len(today_trades),
            "winners": len(winners),
            "losers": len(today_trades) - len(winners),
            "win_rate": (len(winners) / len(today_trades) * 100) if today_trades else 0,
            "total_pnl_pct": sum(t["pnl_pct"] for t in today_trades),
            "avg_pnl_pct": sum(t["pnl_pct"] for t in today_trades) / len(today_trades),
            "best_trade": max(t["pnl_pct"] for t in today_trades),
            "worst_trade": min(t["pnl_pct"] for t in today_trades)
        }
        
    except Exception as e:
        LOGGER.error(f"Daily summary failed: {e}")
        return {}


# ============================================================================
# DATA PERSISTENCE
# ============================================================================

def _save_performance_data():
    """
    Save performance data to disk
    """
    try:
        PERFORMANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(PERFORMANCE_FILE, "w") as f:
            json.dump(_PERFORMANCE_DATA, f, indent=2)
        
    except Exception as e:
        LOGGER.error(f"Performance data save failed: {e}")


def _load_performance_data():
    """
    Load performance data from disk
    """
    global _PERFORMANCE_DATA
    
    try:
        if PERFORMANCE_FILE.exists():
            with open(PERFORMANCE_FILE, "r") as f:
                _PERFORMANCE_DATA = json.load(f)
            
            LOGGER.info(f"📊 Loaded {len(_PERFORMANCE_DATA['trades'])} historical trades")
        
    except Exception as e:
        LOGGER.error(f"Performance data load failed: {e}")


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

async def performance_monitor_loop():
    """
    Background loop to monitor and report performance
    """
    LOGGER.info("🚀 Performance Tracker: STARTED")
    
    # Load historical data
    _load_performance_data()
    
    while True:
        try:
            # Calculate stats
            stats = calculate_statistics()
            calibration = calibrate_confidence()
            strategy_stats = analyze_by_strategy()
            daily_summary = generate_daily_summary()
            
            # Log summary
            if stats.get("total_trades", 0) > 0:
                LOGGER.info(
                    f"📊 Performance: {stats['total_trades']} trades, "
                    f"{stats['win_rate']:.1f}% win rate, "
                    f"+{stats['avg_gain']:.1f}% avg gain"
                )
            
            # Update cached stats
            _PERFORMANCE_DATA["daily_stats"] = daily_summary
            _PERFORMANCE_DATA["strategy_stats"] = strategy_stats
            _PERFORMANCE_DATA["confidence_calibration"] = calibration
            
            # Save to disk
            _save_performance_data()
            
            # Check every 30 minutes
            await asyncio.sleep(1800)
            
        except Exception as e:
            LOGGER.error(f"Performance monitor error: {e}")
            await asyncio.sleep(60)


# ============================================================================
# PUBLIC API
# ============================================================================

def get_performance_summary() -> dict:
    """
    Get comprehensive performance summary
    """
    return {
        "overall_stats": calculate_statistics(),
        "confidence_calibration": calibrate_confidence(),
        "strategy_breakdown": analyze_by_strategy(),
        "daily_summary": generate_daily_summary()
    }

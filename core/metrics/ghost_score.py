"""
Ghost Score V2 - Safety & Quality Metric

Improved scoring system that reflects:
- Data quality (provider redundancy, live prices)
- Prediction coverage (symbols with predictions)
- Risk behavior (within safe limits)

Score range: 0-100 (higher is better)
"""

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)

# Weight distribution for score components
WEIGHT_DATA_QUALITY = 0.40  # 40% - Most critical
WEIGHT_PREDICTION_COVERAGE = 0.35  # 35% - Second priority
WEIGHT_RISK_BEHAVIOR = 0.25  # 25% - Safety check


def compute_ghost_score_v2(
    data_quality: dict[str, Any] | None = None,
    prediction_coverage: dict[str, Any] | None = None,
    risk_status: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Compute Ghost Score V2 based on system health metrics.
    
    Args:
        data_quality: {
            'symbols_with_data': int,
            'total_symbols': int,
            'provider_redundancy': float (0-1),
            'avg_confidence': float (0-1)
        }
        
        prediction_coverage: {
            'predictions_generated': int,
            'total_expected': int,
            'success_rate_estimate': float (0-1)
        }
        
        risk_status: {
            'within_max_position': bool,
            'within_daily_drawdown': bool,
            'within_max_drawdown': bool,
            'stop_loss_configured': bool,
            'take_profit_configured': bool
        }
    
    Returns:
        {
            'score': float (0-100),
            'components': {
                'data_quality': float (0-100),
                'prediction_coverage': float (0-100),
                'risk_behavior': float (0-100)
            },
            'grade': str ('A+', 'A', 'B', 'C', 'D', 'F'),
            'status': str ('excellent', 'good', 'fair', 'poor', 'critical')
        }
    """
    # Component scores (0-100 scale)
    dq_score = _compute_data_quality_score(data_quality or {})
    pc_score = _compute_prediction_coverage_score(prediction_coverage or {})
    risk_score = _compute_risk_behavior_score(risk_status or {})
    
    # Weighted total
    total_score = (
        dq_score * WEIGHT_DATA_QUALITY +
        pc_score * WEIGHT_PREDICTION_COVERAGE +
        risk_score * WEIGHT_RISK_BEHAVIOR
    )
    
    # Ensure bounds
    total_score = max(0.0, min(100.0, total_score))
    
    # Determine grade and status
    grade = _score_to_grade(total_score)
    status = _score_to_status(total_score)
    
    return {
        "score": round(total_score, 2),
        "components": {
            "data_quality": round(dq_score, 2),
            "prediction_coverage": round(pc_score, 2),
            "risk_behavior": round(risk_score, 2)
        },
        "grade": grade,
        "status": status,
        "weights": {
            "data_quality": WEIGHT_DATA_QUALITY,
            "prediction_coverage": WEIGHT_PREDICTION_COVERAGE,
            "risk_behavior": WEIGHT_RISK_BEHAVIOR
        }
    }


def _compute_data_quality_score(data: dict[str, Any]) -> float:
    """
    Compute data quality component (0-100).
    
    Based on:
    - % of symbols with valid live prices
    - Provider redundancy (multiple sources)
    - Average confidence of price data
    """
    symbols_with_data = data.get("symbols_with_data", 0)
    total_symbols = data.get("total_symbols", 1)
    provider_redundancy = data.get("provider_redundancy", 0.5)
    avg_confidence = data.get("avg_confidence", 0.5)
    
    # Avoid division by zero
    if total_symbols == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = symbols_with_data / total_symbols
    
    # Weighted sub-components
    coverage_score = coverage_ratio * 100  # 0-100
    redundancy_score = provider_redundancy * 100  # 0-100
    confidence_score = avg_confidence * 100  # 0-100
    
    # Average the sub-components
    dq_score = (coverage_score * 0.5 + redundancy_score * 0.3 + confidence_score * 0.2)
    
    return max(0.0, min(100.0, dq_score))


def _compute_prediction_coverage_score(data: dict[str, Any]) -> float:
    """
    Compute prediction coverage component (0-100).
    
    Based on:
    - % of symbols that got predictions
    - Success rate estimate (if available)
    """
    predictions_generated = data.get("predictions_generated", 0)
    total_expected = data.get("total_expected", 1)
    success_rate_estimate = data.get("success_rate_estimate", 0.5)
    
    # Avoid division by zero
    if total_expected == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = predictions_generated / total_expected
    
    # Weighted: 70% coverage, 30% success rate
    coverage_score = coverage_ratio * 100
    success_score = success_rate_estimate * 100
    
    pc_score = coverage_score * 0.7 + success_score * 0.3
    
    return max(0.0, min(100.0, pc_score))


def _compute_risk_behavior_score(data: dict[str, Any]) -> float:
    """
    Compute risk behavior component (0-100).
    
    Based on:
    - Compliance with position limits
    - Compliance with drawdown limits
    - Stop-loss/take-profit configuration
    """
    # Read env risk limits
    within_max_position = data.get("within_max_position", True)
    within_daily_dd = data.get("within_daily_drawdown", True)
    within_max_dd = data.get("within_max_drawdown", True)
    sl_configured = data.get("stop_loss_configured", True)
    tp_configured = data.get("take_profit_configured", True)
    
    # Each check is worth 20 points
    risk_score = 0.0
    
    if within_max_position:
        risk_score += 20.0
    if within_daily_dd:
        risk_score += 20.0
    if within_max_dd:
        risk_score += 20.0
    if sl_configured:
        risk_score += 20.0
    if tp_configured:
        risk_score += 20.0
    
    return risk_score


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade"""
    if score >= 97:
        return "A+"
    elif score >= 93:
        return "A"
    elif score >= 90:
        return "A-"
    elif score >= 87:
        return "B+"
    elif score >= 83:
        return "B"
    elif score >= 80:
        return "B-"
    elif score >= 77:
        return "C+"
    elif score >= 73:
        return "C"
    elif score >= 70:
        return "C-"
    elif score >= 60:
        return "D"
    else:
        return "F"


def _score_to_status(score: float) -> str:
    """Convert numeric score to status string"""
    if score >= 90:
        return "excellent"
    elif score >= 75:
        return "good"
    elif score >= 60:
        return "fair"
    elif score >= 40:
        return "poor"
    else:
        return "critical"


def get_current_risk_status() -> dict[str, Any]:
    """
    Get current risk status from environment configuration.
    
    Returns dict suitable for compute_ghost_score_v2 risk_status parameter.
    """
    # Read risk environment variables
    max_pos_pct = float(os.getenv("RISK_MAX_POS_PCT", "5"))
    max_daily_dd_pct = float(os.getenv("RISK_MAX_DAILY_DD_PCT", "5"))
    max_risk_dd = float(os.getenv("MAX_RISK_DRAWDOWN", "0.05"))
    sl_pct = float(os.getenv("RISK_SL_PCT", "3"))
    tp_pct = float(os.getenv("RISK_TP_PCT", "6"))
    
    # For now, assume all limits are within bounds (no live P&L tracking yet)
    # In future, compare actual positions/drawdown against these limits
    return {
        "within_max_position": True,  # Placeholder - would check actual position sizes
        "within_daily_drawdown": True,  # Placeholder - would check daily P&L
        "within_max_drawdown": True,  # Placeholder - would check total drawdown
        "stop_loss_configured": sl_pct > 0,
        "take_profit_configured": tp_pct > 0,
        "config": {
            "max_position_pct": max_pos_pct,
            "max_daily_dd_pct": max_daily_dd_pct,
            "max_risk_drawdown": max_risk_dd,
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct
        }
    }

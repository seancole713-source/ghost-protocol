"""
Position Sizing and Risk Management for Ghost Protocol
Real Kelly Criterion position sizing and portfolio risk limits
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculate optimal position sizes using Kelly Criterion"""
    
    def __init__(self):
        self.enabled = os.getenv("RISK_MANAGEMENT_ENABLED", "1") == "1"
        
        # Portfolio settings
        self.max_portfolio_risk_pct = float(os.getenv("RISK_MAX_PORTFOLIO_PCT", "10.0"))
        self.max_single_position_pct = float(os.getenv("RISK_MAX_POSITION_PCT", "5.0"))
        self.max_daily_drawdown_pct = float(os.getenv("RISK_MAX_DAILY_DD_PCT", "5.0"))
        self.max_correlated_exposure_pct = float(os.getenv("RISK_MAX_CORRELATED_PCT", "15.0"))
        
        # Position sizing
        self.kelly_fraction = float(os.getenv("RISK_KELLY_FRACTION", "0.25"))  # Quarter Kelly
        self.min_position_pct = float(os.getenv("RISK_MIN_POSITION_PCT", "1.0"))
        
        # State
        self.daily_pnl = 0.0
        self.daily_reset_time: Optional[datetime] = None
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        confidence: float
    ) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            portfolio_value: Total portfolio value
            win_rate: Historical win rate (0-1)
            avg_win_pct: Average winning trade percentage
            avg_loss_pct: Average losing trade percentage (positive number)
            confidence: Prediction confidence (0-1)
            
        Returns:
            Dict with position_size, position_pct, kelly_pct
        """
        if not self.enabled:
            return {
                "enabled": False,
                "position_pct": self.max_single_position_pct,
                "position_size": portfolio_value * (self.max_single_position_pct / 100)
            }
        
        # Kelly Criterion: f* = (bp - q) / b
        # where b = odds (avg_win / avg_loss), p = win probability, q = lose probability
        
        if avg_loss_pct <= 0:
            avg_loss_pct = 2.0  # Default 2% loss
        
        b = avg_win_pct / avg_loss_pct  # Win/loss ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_pct = ((b * p) - q) / b if b > 0 else 0
        
        # Apply Kelly fraction (quarter Kelly for safety)
        adjusted_kelly = kelly_pct * self.kelly_fraction
        
        # Apply confidence multiplier
        confidence_adjusted = adjusted_kelly * confidence
        
        # Clamp to min/max
        final_pct = max(self.min_position_pct, min(self.max_single_position_pct, confidence_adjusted * 100))
        
        # Ensure non-negative
        final_pct = max(0, final_pct)
        
        position_size = portfolio_value * (final_pct / 100)
        
        return {
            "enabled": True,
            "portfolio_value": portfolio_value,
            "kelly_raw_pct": round(kelly_pct * 100, 2),
            "kelly_adjusted_pct": round(adjusted_kelly * 100, 2),
            "confidence_adjusted_pct": round(confidence_adjusted * 100, 2),
            "final_position_pct": round(final_pct, 2),
            "position_size": round(position_size, 2),
            "inputs": {
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "confidence": confidence
            }
        }
    
    def check_risk_limits(
        self,
        portfolio_value: float,
        proposed_position: float,
        symbol: str,
        current_exposure: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Check if a proposed position violates risk limits
        
        Args:
            portfolio_value: Total portfolio value
            proposed_position: Proposed position size in dollars
            symbol: Symbol being traded
            current_exposure: Dict of symbol -> current exposure
            
        Returns:
            Dict with allowed, violations, adjusted_size
        """
        violations = []
        warnings = []
        
        current_exposure = current_exposure or {}
        
        # Check single position limit
        position_pct = (proposed_position / portfolio_value) * 100 if portfolio_value > 0 else 0
        if position_pct > self.max_single_position_pct:
            violations.append(f"Position {position_pct:.1f}% exceeds max {self.max_single_position_pct}%")
            proposed_position = portfolio_value * (self.max_single_position_pct / 100)
        
        # Check total portfolio risk
        total_exposure = sum(current_exposure.values()) + proposed_position
        total_pct = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
        if total_pct > self.max_portfolio_risk_pct:
            violations.append(f"Total exposure {total_pct:.1f}% exceeds max {self.max_portfolio_risk_pct}%")
        
        # Check daily drawdown
        if self.daily_pnl < -(self.max_daily_drawdown_pct / 100) * portfolio_value:
            violations.append(f"Daily drawdown limit reached: {self.daily_pnl:.2f}")
        
        # Check correlated exposure (crypto vs crypto, tech vs tech)
        asset_class = self._get_asset_class(symbol)
        class_exposure = sum(v for s, v in current_exposure.items() if self._get_asset_class(s) == asset_class)
        class_pct = ((class_exposure + proposed_position) / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        if class_pct > self.max_correlated_exposure_pct:
            warnings.append(f"{asset_class} exposure {class_pct:.1f}% high (max {self.max_correlated_exposure_pct}%)")
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "original_size": proposed_position,
            "adjusted_size": proposed_position if len(violations) == 0 else portfolio_value * (self.max_single_position_pct / 100),
            "total_exposure_pct": round(total_pct, 1),
            "daily_pnl": round(self.daily_pnl, 2)
        }
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class for correlation grouping"""
        crypto = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LINK", "AVAX", "DOT", "MATIC", "PEPE", "SHIB"]
        tech = ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "NFLX", "TSLA", "AMZN"]
        
        if symbol.upper() in crypto:
            return "CRYPTO"
        elif symbol.upper() in tech:
            return "TECH"
        else:
            return "OTHER"
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracker"""
        now = datetime.utcnow()
        if self.daily_reset_time is None or now.date() > self.daily_reset_time.date():
            self.daily_pnl = 0.0
            self.daily_reset_time = now
        
        self.daily_pnl += pnl
    
    def get_risk_report(self, portfolio_value: float, positions: Dict[str, float]) -> Dict:
        """
        Generate comprehensive risk report
        
        Args:
            portfolio_value: Total portfolio value
            positions: Dict of symbol -> position size
            
        Returns:
            Risk report with metrics and recommendations
        """
        total_exposure = sum(positions.values())
        exposure_pct = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # Group by asset class
        by_class: Dict[str, float] = {}
        for symbol, size in positions.items():
            asset_class = self._get_asset_class(symbol)
            by_class[asset_class] = by_class.get(asset_class, 0) + size
        
        # Calculate concentration
        if positions:
            largest_position = max(positions.values())
            concentration = (largest_position / total_exposure) * 100 if total_exposure > 0 else 0
        else:
            largest_position = 0
            concentration = 0
        
        return {
            "portfolio_value": portfolio_value,
            "total_exposure": round(total_exposure, 2),
            "exposure_pct": round(exposure_pct, 1),
            "position_count": len(positions),
            "by_asset_class": {k: round(v, 2) for k, v in by_class.items()},
            "largest_position": round(largest_position, 2),
            "concentration_pct": round(concentration, 1),
            "daily_pnl": round(self.daily_pnl, 2),
            "risk_status": "OK" if exposure_pct < self.max_portfolio_risk_pct else "HIGH",
            "recommendations": self._get_recommendations(exposure_pct, concentration, by_class, portfolio_value)
        }
    
    def _get_recommendations(
        self,
        exposure_pct: float,
        concentration: float,
        by_class: Dict[str, float],
        portfolio_value: float
    ) -> List[str]:
        """Generate risk recommendations"""
        recs = []
        
        if exposure_pct > self.max_portfolio_risk_pct * 0.8:
            recs.append("Consider reducing overall exposure")
        
        if concentration > 40:
            recs.append("High concentration - diversify positions")
        
        for asset_class, exposure in by_class.items():
            class_pct = (exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
            if class_pct > self.max_correlated_exposure_pct:
                recs.append(f"Reduce {asset_class} exposure ({class_pct:.0f}% > {self.max_correlated_exposure_pct}%)")
        
        if not recs:
            recs.append("Risk levels within acceptable limits")
        
        return recs


# Singleton
_position_sizer: Optional[PositionSizer] = None


def get_position_sizer() -> PositionSizer:
    """Get or create PositionSizer singleton"""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer()
    return _position_sizer


def calculate_position_size(
    portfolio_value: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    confidence: float
) -> Dict:
    """Calculate position size using Kelly Criterion"""
    return get_position_sizer().calculate_position_size(
        portfolio_value, win_rate, avg_win, avg_loss, confidence
    )


def check_risk_limits(
    portfolio_value: float,
    proposed_position: float,
    symbol: str,
    current_exposure: Optional[Dict[str, float]] = None
) -> Dict:
    """Check if a proposed position violates risk limits"""
    return get_position_sizer().check_risk_limits(
        portfolio_value, proposed_position, symbol, current_exposure
    )


def get_risk_report(portfolio_value: float, positions: Dict[str, float]) -> Dict:
    """Generate comprehensive risk report"""
    return get_position_sizer().get_risk_report(portfolio_value, positions)

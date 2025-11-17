"""
Risk management dashboard calculations.
Provides portfolio risk metrics and hedging suggestions.
"""

from typing import Any


def calculate_portfolio_risk(
    positions: dict[str, dict[str, Any]],
    current_prices: dict[str, float],
    forecasts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Calculate comprehensive portfolio risk metrics.

    Args:
        positions: Current positions {symbol: {quantity, entry_price, ...}}
        current_prices: Current prices {symbol: price}
        forecasts: Latest predictions {symbol: forecast_data}

    Returns:
        Risk metrics including exposure, volatility, diversification
    """
    if not positions:
        return {
            "total_exposure_pct": 0,
            "sector_allocation": {},
            "risk_level": "NONE",
            "max_drawdown_estimate": 0,
            "diversification_score": 0,
            "hedging_suggestions": []
        }

    # Calculate total exposure
    total_value = 0
    sector_exposure = {}
    position_risks = []

    for symbol, pos in positions.items():
        current_price = current_prices.get(symbol, pos.get("entry_price", 0))
        position_value = pos["quantity"] * current_price
        total_value += position_value

        # Extract sector from forecast if available
        forecast = forecasts.get(symbol, {})
        confidence = forecast.get("confidence", 0.5)
        gain_potential = forecast.get("gain_potential_pct", 0)

        # Simplified sector mapping (would use real sector data in production)
        sector = _get_sector(symbol)
        sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value

        # Calculate individual position risk
        position_risk = abs(gain_potential) * (1 - confidence)
        position_risks.append(position_risk)

    # Calculate metrics
    exposure_pct = 100.0  # Assume full portfolio invested
    diversification_score = len(positions) / 10.0  # 10+ positions = perfect diversification
    diversification_score = min(diversification_score, 1.0)

    # Estimate max drawdown based on position risks
    avg_risk = sum(position_risks) / len(position_risks) if position_risks else 0
    max_drawdown_estimate = avg_risk * 100

    # Determine risk level
    if avg_risk > 0.15:
        risk_level = "HIGH"
    elif avg_risk > 0.08:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Generate hedging suggestions
    hedging_suggestions = []

    if avg_risk > 0.10:
        hedging_suggestions.append({
            "action": "Reduce position sizes",
            "reason": "High portfolio risk detected",
            "priority": "HIGH"
        })

    if len(positions) < 5:
        hedging_suggestions.append({
            "action": "Increase diversification",
            "reason": f"Only {len(positions)} positions - consider 8-10 for better risk spread",
            "priority": "MEDIUM"
        })

    # Check sector concentration
    for sector, value in sector_exposure.items():
        sector_pct = (value / total_value) * 100 if total_value > 0 else 0
        if sector_pct > 40:
            hedging_suggestions.append({
                "action": f"Reduce {sector} exposure",
                "reason": f"{sector} represents {sector_pct:.1f}% of portfolio",
                "priority": "MEDIUM"
            })

    return {
        "total_exposure_pct": exposure_pct,
        "sector_allocation": {
            sector: (value / total_value * 100 if total_value > 0 else 0)
            for sector, value in sector_exposure.items()
        },
        "risk_level": risk_level,
        "max_drawdown_estimate": max_drawdown_estimate,
        "diversification_score": diversification_score,
        "hedging_suggestions": hedging_suggestions,
        "total_value": total_value
    }


def _get_sector(symbol: str) -> str:
    """
    Map symbol to sector.
    Simplified mapping - production would use API lookup.
    """
    tech_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ORCL", "CRM", "ADBE", "NFLX", "INTC", "AMD"]
    finance_symbols = ["JPM", "BAC", "WFC", "GS", "V", "MA"]
    health_symbols = ["UNH", "JNJ", "PFE", "ABBV"]
    consumer_symbols = ["WMT", "HD", "NKE", "SBUX"]
    crypto_symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX"]

    symbol_upper = symbol.upper().replace("-USD", "")

    if symbol_upper in tech_symbols:
        return "Technology"
    elif symbol_upper in finance_symbols:
        return "Finance"
    elif symbol_upper in health_symbols:
        return "Healthcare"
    elif symbol_upper in consumer_symbols:
        return "Consumer"
    elif symbol_upper in crypto_symbols:
        return "Crypto"
    else:
        return "Other"


def calculate_correlation_risk(positions: list[str]) -> dict[str, Any]:
    """
    Estimate correlation risk between positions.
    Simplified version - production would use historical correlation matrix.
    """
    if len(positions) < 2:
        return {"correlation_risk": "LOW", "correlated_pairs": []}

    # Simplified: Assume same-sector positions are highly correlated
    sectors = [_get_sector(symbol) for symbol in positions]
    sector_counts = {}
    for sector in sectors:
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    high_correlation_sectors = [s for s, count in sector_counts.items() if count >= 3]

    if high_correlation_sectors:
        return {
            "correlation_risk": "HIGH",
            "correlated_pairs": high_correlation_sectors,
            "warning": f"Multiple positions in {', '.join(high_correlation_sectors)}"
        }

    return {"correlation_risk": "LOW", "correlated_pairs": []}

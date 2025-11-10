"""
GHOST Value at Risk (VaR) Calculator
Calculate VaR, CVaR, and other risk metrics using scipy.
Completely free - no paid services required!
"""

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]


class VaRCalculator:
    """Calculate Value at Risk and related risk metrics."""

    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series."""
        log_returns = np.log(prices / prices.shift(1))
        return (
            log_returns.dropna()
            if isinstance(log_returns, pd.Series)
            else pd.Series(log_returns).dropna()
        )

    def historical_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Historical VaR using empirical distribution.

        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 = 95%)

        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        percentile = (1 - confidence) * 100
        return float(np.percentile(returns, percentile))

    def parametric_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Parametric VaR assuming normal distribution.

        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 = 95%)

        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)

        return float(mean + z_score * std)

    def monte_carlo_var(
        self, returns: pd.Series, confidence: float = 0.95, simulations: int = 10000
    ) -> float:
        """
        Calculate Monte Carlo VaR using simulated returns.

        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 = 95%)
            simulations: Number of Monte Carlo simulations

        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        mean = returns.mean()
        std = returns.std()

        # Generate random returns
        simulated_returns = np.random.normal(mean, std, simulations)

        percentile = (1 - confidence) * 100
        return float(np.percentile(simulated_returns, percentile))

    def conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall).
        Average loss beyond the VaR threshold.

        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 = 95%)

        Returns:
            CVaR value (average loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0

        var = self.historical_var(returns, confidence)
        # Get returns worse than VaR
        tail_losses = returns[returns <= var]

        if len(tail_losses) == 0:
            return var

        return tail_losses.mean()

    def calculate_all_var(self, prices: pd.Series, portfolio_value: float = 10000.0) -> dict:
        """
        Calculate all VaR metrics at multiple confidence levels.

        Args:
            prices: Price series
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with VaR metrics
        """
        returns = self.calculate_returns(prices)

        results = {
            "portfolio_value": portfolio_value,
            "returns_mean": returns.mean(),
            "returns_std": returns.std(),
            "returns_skew": returns.skew(),
            "returns_kurtosis": returns.kurtosis(),
            "var_metrics": {},
        }

        for confidence in self.confidence_levels:
            conf_pct = int(confidence * 100)

            # Historical VaR
            hist_var = self.historical_var(returns, confidence)
            hist_var_dollar = abs(hist_var * portfolio_value)

            # Parametric VaR
            param_var = self.parametric_var(returns, confidence)
            param_var_dollar = abs(param_var * portfolio_value)

            # Monte Carlo VaR
            mc_var = self.monte_carlo_var(returns, confidence)
            mc_var_dollar = abs(mc_var * portfolio_value)

            # Conditional VaR
            cvar = self.conditional_var(returns, confidence)
            cvar_dollar = abs(cvar * portfolio_value)

            results["var_metrics"][f"{conf_pct}%"] = {
                "historical_var": {"percent": hist_var, "dollar": hist_var_dollar},
                "parametric_var": {"percent": param_var, "dollar": param_var_dollar},
                "monte_carlo_var": {"percent": mc_var, "dollar": mc_var_dollar},
                "conditional_var": {"percent": cvar, "dollar": cvar_dollar},
            }

        return results

    def portfolio_var(
        self,
        positions: list[dict],
        correlation_matrix: pd.DataFrame | None = None,
        confidence: float = 0.95,
    ) -> dict:
        """
        Calculate portfolio-level VaR considering correlations.

        Args:
            positions: List of dicts with 'symbol', 'value', 'returns'
            correlation_matrix: Correlation matrix of returns (optional)
            confidence: Confidence level

        Returns:
            Dictionary with portfolio VaR metrics
        """
        if not positions:
            return {"error": "No positions provided"}

        # Extract position values and returns
        values = np.array([pos["value"] for pos in positions])
        returns_data = [pos["returns"] for pos in positions]

        # Calculate portfolio value
        portfolio_value = values.sum()
        weights = values / portfolio_value

        # Calculate portfolio returns
        returns_df = pd.DataFrame(returns_data).T
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Calculate VaR
        var = self.historical_var(portfolio_returns, confidence)
        cvar = self.conditional_var(portfolio_returns, confidence)

        # Calculate diversification benefit
        individual_var = sum(
            [
                abs(self.historical_var(pd.Series(ret), confidence)) * val
                for ret, val in zip(returns_data, values, strict=False)
            ]
        )

        portfolio_var_dollar = abs(var * portfolio_value)
        diversification_benefit = individual_var - portfolio_var_dollar

        return {
            "portfolio_value": portfolio_value,
            "confidence": confidence,
            "var": {"percent": var, "dollar": portfolio_var_dollar},
            "cvar": {"percent": cvar, "dollar": abs(cvar * portfolio_value)},
            "individual_var_sum": individual_var,
            "diversification_benefit": diversification_benefit,
            "positions": len(positions),
        }

    def marginal_var(
        self,
        position_returns: pd.Series,
        portfolio_returns: pd.Series,
        position_weight: float,
        confidence: float = 0.95,
    ) -> dict:
        """
        Calculate Marginal VaR - contribution of a position to portfolio VaR.

        Args:
            position_returns: Returns of the position
            portfolio_returns: Returns of the entire portfolio
            position_weight: Weight of position in portfolio (0-1)
            confidence: Confidence level

        Returns:
            Dictionary with marginal VaR metrics
        """
        # Calculate beta of position relative to portfolio
        covariance = np.cov(position_returns, portfolio_returns)[0, 1]
        portfolio_variance = portfolio_returns.var()
        beta = covariance / portfolio_variance if portfolio_variance != 0 else 0

        # Calculate portfolio VaR
        portfolio_var = self.historical_var(portfolio_returns, confidence)

        # Marginal VaR = beta * portfolio VaR
        marginal_var = beta * portfolio_var

        # Component VaR = marginal VaR * position weight
        component_var = marginal_var * position_weight

        return {
            "beta": beta,
            "marginal_var": marginal_var,
            "component_var": component_var,
            "position_weight": position_weight,
        }

    def stress_test(self, returns: pd.Series, scenarios: list[dict]) -> dict:
        """
        Perform stress testing with custom scenarios.

        Args:
            returns: Historical returns
            scenarios: List of dicts with 'name' and 'shock' (e.g., -0.10 for -10%)

        Returns:
            Dictionary with stress test results
        """
        mean = returns.mean()
        std = returns.std()

        results = {"base_case": {"mean_return": mean, "std_return": std}, "scenarios": {}}

        for scenario in scenarios:
            name = scenario["name"]
            shock = scenario["shock"]

            # Apply shock to returns
            shocked_returns = returns + shock
            shocked_mean = shocked_returns.mean()
            shocked_std = shocked_returns.std()

            # Calculate VaR under stress
            stressed_var_95 = self.historical_var(shocked_returns, 0.95)
            stressed_cvar_95 = self.conditional_var(shocked_returns, 0.95)

            results["scenarios"][name] = {
                "shock": shock,
                "mean_return": shocked_mean,
                "std_return": shocked_std,
                "var_95": stressed_var_95,
                "cvar_95": stressed_cvar_95,
            }

        return results

    def backtesting_var(self, returns: pd.Series, var_values: pd.Series) -> dict:
        """
        Backtest VaR model by comparing actual losses to VaR predictions.

        Args:
            returns: Actual returns
            var_values: Predicted VaR values

        Returns:
            Dictionary with backtesting metrics
        """
        # Count violations (actual loss > VaR prediction)
        violations = (returns < var_values).sum()
        total_days = len(returns)
        violation_rate = violations / total_days

        # Expected violation rate at 95% confidence is 5%
        expected_violations = total_days * 0.05

        # Kupiec test (likelihood ratio test)
        if violations > 0:
            lr_stat = 2 * (
                violations * np.log(violation_rate / 0.05)
                + (total_days - violations) * np.log((1 - violation_rate) / 0.95)
            )
            p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            lr_stat = 0
            p_value = 1.0

        return {
            "total_days": total_days,
            "violations": violations,
            "violation_rate": violation_rate,
            "expected_violations": expected_violations,
            "kupiec_lr_stat": lr_stat,
            "kupiec_p_value": p_value,
            "model_adequate": p_value > 0.05,  # Model is adequate if p > 0.05
        }

    def tail_risk_metrics(self, returns: pd.Series) -> dict:
        """
        Calculate tail risk metrics: skewness, kurtosis, max drawdown.

        Args:
            returns: Returns series

        Returns:
            Dictionary with tail risk metrics
        """
        # Skewness (negative = left tail risk)
        skewness = returns.skew()

        # Excess kurtosis (positive = fat tails)
        kurtosis = returns.kurtosis()

        # Max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Downside deviation (semi-deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0

        # Sortino ratio (using 0% as minimum acceptable return)
        sortino = returns.mean() / downside_deviation if downside_deviation != 0 else 0

        # Value at Risk at multiple levels
        var_90 = self.historical_var(returns, 0.90)
        var_95 = self.historical_var(returns, 0.95)
        var_99 = self.historical_var(returns, 0.99)

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "max_drawdown": max_drawdown,
            "downside_deviation": downside_deviation,
            "sortino_ratio": sortino,
            "var_90": var_90,
            "var_95": var_95,
            "var_99": var_99,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_var(
    prices: pd.Series, confidence: float = 0.95, portfolio_value: float = 10000.0
) -> dict:
    """
    Quick VaR calculation for a single asset.

    Args:
        prices: Price series
        confidence: Confidence level (default 95%)
        portfolio_value: Portfolio value in dollars

    Returns:
        Dictionary with VaR summary
    """
    calc = VaRCalculator()
    returns = calc.calculate_returns(prices)

    hist_var = calc.historical_var(returns, confidence)
    cvar = calc.conditional_var(returns, confidence)

    return {
        "confidence": confidence,
        "historical_var": {"percent": hist_var, "dollar": abs(hist_var * portfolio_value)},
        "conditional_var": {"percent": cvar, "dollar": abs(cvar * portfolio_value)},
        "interpretation": f"At {int(confidence * 100)}% confidence, you could lose up to ${abs(hist_var * portfolio_value):.2f} ({abs(hist_var) * 100:.2f}%) in a single day.",
    }


def daily_var_report(prices: pd.Series, portfolio_value: float = 10000.0) -> dict:
    """
    Generate daily VaR report with all metrics.

    Args:
        prices: Price series
        portfolio_value: Current portfolio value

    Returns:
        Comprehensive VaR report
    """
    calc = VaRCalculator()

    # Calculate all VaR metrics
    var_metrics = calc.calculate_all_var(prices, portfolio_value)

    # Calculate tail risk
    returns = calc.calculate_returns(prices)
    tail_risk = calc.tail_risk_metrics(returns)

    # Combine results
    return {
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": portfolio_value,
        "var_metrics": var_metrics["var_metrics"],
        "tail_risk": tail_risk,
        "summary": {
            "var_95_dollar": var_metrics["var_metrics"]["95%"]["historical_var"]["dollar"],
            "cvar_95_dollar": var_metrics["var_metrics"]["95%"]["conditional_var"]["dollar"],
            "max_drawdown": tail_risk["max_drawdown"],
            "sortino_ratio": tail_risk["sortino_ratio"],
        },
    }


def stress_test_scenarios(returns: pd.Series) -> dict:
    """
    Run standard stress test scenarios.

    Args:
        returns: Historical returns

    Returns:
        Stress test results
    """
    calc = VaRCalculator()

    scenarios = [
        {"name": "Mild Correction", "shock": -0.05},
        {"name": "Market Correction", "shock": -0.10},
        {"name": "Bear Market", "shock": -0.20},
        {"name": "Market Crash", "shock": -0.30},
        {"name": "Black Swan", "shock": -0.50},
    ]

    return calc.stress_test(returns, scenarios)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = ["VaRCalculator", "quick_var", "daily_var_report", "stress_test_scenarios"]

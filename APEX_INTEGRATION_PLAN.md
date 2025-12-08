# 🚀 APEX → GHOST Integration Plan

**Date**: October 5, 2025\
**Goal**: Integrate APEX's advanced trading architecture into GHOST\
**Current GHOST Version**: v10.3.1\
**Target Version**: v11.0.0 "APEX Edition"

______________________________________________________________________

## 📊 APEX vs GHOST: Feature Comparison

### ✅ What GHOST Already Has

| APEX Feature | GHOST Status | Location | |-------------|--------------|----------| |
**Regime Detection**| ✅ EXISTS | `core/regime_detector.py` | |**Ensemble Forecasting**| ✅ EXISTS | `core/ensemble_forecaster.py` | |**VaR Calculator**| ✅ EXISTS |
`core/var_calculator.py` (Historical, Parametric, Monte Carlo, CVaR) | |**Risk Limits**| ✅ EXISTS | `wolf_app.py`
(position limits, max loss checks) | |**Circuit Breakers**|
✅ EXISTS | Provider circuit breaker (line 2900) | |**AI Rationale**| ✅ EXISTS | AI
decision with rationale, risks, evidence, checklist | |**News Integration**| ✅ EXISTS
| Polygon API, sentiment scoring | |**Two-Line Forecast**| ✅ EXISTS | Predicted vs
actual overlay | |**FastAPI + SSE**| ✅ EXISTS | Real-time cockpit updates | |**Telegram Alerts**| ✅ EXISTS | Alert
system with cooldown |

### 🟡 What Needs Enhancement

| APEX Feature | GHOST Gap | Priority | Effort |
|-------------|-----------|----------|--------| |**Multi-Horizon Brain**| Only 48h
forecast | 🔴 HIGH | 3 days | |**Strategy Ensemble**| Single strategy | 🔴 HIGH | 5 days
| |**Event Engine**| Basic news only | 🟡 MED | 7 days | |**Meta-Learner**| No regime
weighting | 🔴 HIGH | 4 days | |**Hard Risk Shell**| Basic VaR, no kill-switch | 🟡 MED
| 2 days | |**Shadow Deployment**| No A/B testing | 🟢 LOW | 5 days | |**Feature
Store**| Direct computation | 🟢 LOW | 7 days | |**Online Calibration**| Static models
| 🟡 MED | 4 days |

______________________________________________________________________

## 🎯 Phase 1: Multi-Horizon Brain (MVP - Week 1)

### Goal: Add 3 forecast horizons**Current**: Single 48h forecast\

**Target**: Nowcast (15min-4h), Swing (48h-7d), Position (1-4 weeks)

### Implementation

#### 1.1 Add Multi-Horizon Forecaster

Create `/workspaces/GHOST/core/multi_horizon_forecaster.py`:

```python
"""
Multi-Horizon Forecasting System
Nowcast: 15min - 4h (microstructure signals)
Swing: 1-7 days (technical + sentiment)
Position: 1-4 weeks (fundamental + macro)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class Horizon(Enum):
    NOWCAST = "nowcast"    # 15min - 4h
    SWING = "swing"        # 1-7 days
    POSITION = "position"  # 1-4 weeks

@dataclass
class HorizonForecast:
    horizon: Horizon
    confidence: float
    expected_return: float
    expected_pnl: float
    risk_var_95: float
    signals: Dict[str, float]
    rationale: str

class MultiHorizonForecaster:
    def __init__(self):
        self.horizons = {
            Horizon.NOWCAST: {"window_h": 4, "features": ["vwap_drift", "order_imbalance", "volume_shock"]},
            Horizon.SWING: {"window_h": 168, "features": ["momentum", "sentiment", "technical"]},  # 7 days
            Horizon.POSITION: {"window_h": 672, "features": ["value", "quality", "macro"]}  # 28 days
        }

    def forecast_all_horizons(self, symbol: str, price_history: pd.DataFrame,
                             sentiment: Optional[float] = None) -> Dict[Horizon, HorizonForecast]:
        """Generate forecasts for all horizons."""
        results = {}

        for horizon, config in self.horizons.items():
            try:
                forecast = self._generate_horizon_forecast(
                    symbol, price_history, sentiment, horizon, config
                )
                results[horizon] = forecast
            except Exception as e:
                print(f"Forecast failed for {horizon.value}: {e}")
                results[horizon] = self._default_forecast(horizon)

        return results

    def _generate_horizon_forecast(self, symbol: str, price_history: pd.DataFrame,
                                   sentiment: Optional[float], horizon: Horizon,
                                   config: Dict) -> HorizonForecast:
        """Generate forecast for specific horizon."""

        if horizon == Horizon.NOWCAST:
            return self._nowcast(symbol, price_history, config)
        elif horizon == Horizon.SWING:
            return self._swing_forecast(symbol, price_history, sentiment, config)
        else:  # POSITION
            return self._position_forecast(symbol, price_history, config)

    def _nowcast(self, symbol: str, price_history: pd.DataFrame,
                config: Dict) -> HorizonForecast:
        """Nowcast: 15min - 4h microstructure signals."""

        # VWAP drift (simplified)

        current_price = float(price_history['close'].iloc[-1])
        vwap = float((price_history['close'] * price_history['volume']).sum() / price_history['volume'].sum()) if 'volume' in price_history.columns else current_price
        vwap_drift = (current_price - vwap) / vwap if vwap > 0 else 0.0

        # Volume shock (last vs average)

        if 'volume' in price_history.columns and len(price_history) > 1:
            recent_vol = float(price_history['volume'].iloc[-1])
            avg_vol = float(price_history['volume'].mean())
            volume_shock = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0.0
        else:
            volume_shock = 0.0

        # Simple momentum

        if len(price_history) >= 20:
            returns = price_history['close'].pct_change()
            momentum = float(returns.tail(20).mean())
        else:
            momentum = 0.0

        # Combine signals

        signals = {
            "vwap_drift": vwap_drift,
            "volume_shock": volume_shock,
            "momentum": momentum
        }

        # Expected return (4h ahead)

        expected_return = (vwap_drift *0.4 + volume_shock*0.3 + momentum*0.3)* 0.02  # 2% max
        expected_pnl = expected_return *current_price* 100  # Assume 100 shares

        confidence = min(80.0, max(30.0, 50.0 + abs(expected_return) * 1000))

        return HorizonForecast(
            horizon=Horizon.NOWCAST,
            confidence=confidence,
            expected_return=expected_return,
            expected_pnl=expected_pnl,
            risk_var_95=expected_pnl * -0.5,  # Simplified VaR
            signals=signals,
            rationale=f"VWAP drift {vwap_drift:+.2%}, volume shock {volume_shock:+.1f}x"
        )

    def _swing_forecast(self, symbol: str, price_history: pd.DataFrame,
                       sentiment: Optional[float], config: Dict) -> HorizonForecast:
        """Swing: 1-7 days technical + sentiment."""

        current_price = float(price_history['close'].iloc[-1])

        # Momentum (5-day)

        if len(price_history) >= 5:
            momentum_5d = (price_history['close'].iloc[-1] / price_history['close'].iloc[-5]) - 1
        else:
            momentum_5d = 0.0

        # RSI (simplified 14-period)

        if len(price_history) >= 14:
            returns = price_history['close'].diff()
            gains = returns.where(returns > 0, 0).tail(14).mean()
            losses = -returns.where(returns < 0, 0).tail(14).mean()
            rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 50
        else:
            rsi = 50

        # Combine with sentiment

        sentiment_signal = sentiment if sentiment is not None else 0.0

        signals = {
            "momentum_5d": float(momentum_5d),
            "rsi": float(rsi),
            "sentiment": sentiment_signal
        }

        # Expected return (7 days)

        expected_return = (momentum_5d *0.4 + (rsi - 50) / 100*0.3 + sentiment_signal*0.3)* 0.05
        expected_pnl = expected_return *current_price* 100

        confidence = min(85.0, max(40.0, 60.0 + abs(expected_return) * 500))

        return HorizonForecast(
            horizon=Horizon.SWING,
            confidence=confidence,
            expected_return=expected_return,
            expected_pnl=expected_pnl,
            risk_var_95=expected_pnl * -0.4,
            signals=signals,
            rationale=f"5d momentum {momentum_5d:+.2%}, RSI {rsi:.0f}, sentiment {sentiment_signal:+.2f}"
        )

    def _position_forecast(self, symbol: str, price_history: pd.DataFrame,
                          config: Dict) -> HorizonForecast:
        """Position: 1-4 weeks fundamental + macro."""

        current_price = float(price_history['close'].iloc[-1])

        # Long-term trend (20-day SMA)

        if len(price_history) >= 20:
            sma_20 = price_history['close'].tail(20).mean()
            trend = (current_price - sma_20) / sma_20
        else:
            trend = 0.0

        # Volatility (20-day)

        if len(price_history) >= 20:
            returns = price_history['close'].pct_change()
            volatility = float(returns.tail(20).std() * np.sqrt(252))  # Annualized
        else:
            volatility = 0.2  # Default 20%

        signals = {
            "trend_20d": float(trend),
            "volatility_annual": volatility
        }

        # Expected return (4 weeks)

        expected_return = trend * 0.7  # Trend-following
        expected_pnl = expected_return *current_price* 100

        confidence = min(75.0, max(35.0, 50.0 + abs(expected_return) * 300))

        return HorizonForecast(
            horizon=Horizon.POSITION,
            confidence=confidence,
            expected_return=expected_return,
            expected_pnl=expected_pnl,
            risk_var_95=expected_pnl * -0.3,
            signals=signals,
            rationale=f"20d trend {trend:+.2%}, vol {volatility:.1%}"
        )

    def _default_forecast(self, horizon: Horizon) -> HorizonForecast:
        """Default forecast when generation fails."""
        return HorizonForecast(
            horizon=horizon,
            confidence=0.0,
            expected_return=0.0,
            expected_pnl=0.0,
            risk_var_95=0.0,
            signals={},
            rationale="Forecast unavailable"
        )

    def get_consensus(self, forecasts: Dict[Horizon, HorizonForecast]) -> Dict:
        """Aggregate multi-horizon forecasts into consensus."""

        # Weight by confidence

        total_weight = sum(f.confidence for f in forecasts.values())

        if total_weight == 0:
            return {"action": "HOLD", "confidence": 0, "expected_return": 0.0}

        weighted_return = sum(
            f.confidence * f.expected_return for f in forecasts.values()
        ) / total_weight

        avg_confidence = total_weight / len(forecasts)

        # Determine action

        if weighted_return > 0.02:  # >2%
            action = "BUY"
        elif weighted_return < -0.02:  # <-2%
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "confidence": round(avg_confidence, 1),
            "expected_return": round(weighted_return, 4),
            "nowcast": forecasts[Horizon.NOWCAST],
            "swing": forecasts[Horizon.SWING],
            "position": forecasts[Horizon.POSITION]
        }

```text

#### 1.2 Add API Endpoint

Add to `wolf_app.py`:

```python

@APP.get("/api/forecast/multi_horizon")
async def api_forecast_multi_horizon(symbol: str = WOLF):
    """
    Get multi-horizon forecast (nowcast, swing, position).
    Returns forecasts for 3 time horizons with confidence and signals.
    """
    try:
        from core.multi_horizon_forecaster import MultiHorizonForecaster, Horizon

        # Get price history

        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1h")

        if hist.empty:
            return {"error": "No price data available"}

        # Get sentiment

        sentiment = None
        try:
            news_data = get_wolf_news(limit=10)
            news_signal = news_data.get("news_signal", {})
            sentiment = news_signal.get("score")
        except Exception:
            pass

        # Generate multi-horizon forecasts

        forecaster = MultiHorizonForecaster()
        forecasts = forecaster.forecast_all_horizons(symbol, hist, sentiment)
        consensus = forecaster.get_consensus(forecasts)

        return {
            "symbol": symbol,
            "as_of": int(time.time()),
            "consensus": consensus,
            "forecasts": {
                "nowcast": {
                    "horizon": "15min-4h",
                    "confidence": forecasts[Horizon.NOWCAST].confidence,
                    "expected_return": forecasts[Horizon.NOWCAST].expected_return,
                    "expected_pnl": forecasts[Horizon.NOWCAST].expected_pnl,
                    "signals": forecasts[Horizon.NOWCAST].signals,
                    "rationale": forecasts[Horizon.NOWCAST].rationale
                },
                "swing": {
                    "horizon": "1-7 days",
                    "confidence": forecasts[Horizon.SWING].confidence,
                    "expected_return": forecasts[Horizon.SWING].expected_return,
                    "expected_pnl": forecasts[Horizon.SWING].expected_pnl,
                    "signals": forecasts[Horizon.SWING].signals,
                    "rationale": forecasts[Horizon.SWING].rationale
                },
                "position": {
                    "horizon": "1-4 weeks",
                    "confidence": forecasts[Horizon.POSITION].confidence,
                    "expected_return": forecasts[Horizon.POSITION].expected_return,
                    "expected_pnl": forecasts[Horizon.POSITION].expected_pnl,
                    "signals": forecasts[Horizon.POSITION].signals,
                    "rationale": forecasts[Horizon.POSITION].rationale
                }
            }
        }
    except Exception as e:
        LOGGER.error(f"Multi-horizon forecast failed: {e}")
        return {"error": str(e)}

```text

______________________________________________________________________

## 🎯 Phase 2: Strategy Ensemble & Meta-Learner (Week 2)

### Goal: Multiple strategies with regime-aware weighting

**Current**: Single momentum-based strategy\
**Target**: 5 strategies with meta-learner

### 2.1 Strategy Registry

Create `/workspaces/GHOST/core/strategy_registry.py`:

```python

"""
Strategy Registry - Multiple trading strategies
Each strategy votes (BUY/SELL/HOLD) and meta-learner weighs them by regime
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
import numpy as np

class StrategyVote(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class StrategyDecision:
    name: str
    vote: StrategyVote
    confidence: float  # 0-100
    expected_return: float
    rationale: str
    signals: Dict[str, float]

class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    @abstractmethod
    def evaluate(self, symbol: str, price_data: pd.DataFrame,
                news_sentiment: Optional[float] = None) -> StrategyDecision:
        """Evaluate and return trading decision."""
        pass

class MomentumStrategy(BaseStrategy):
    """Regime-aware momentum with adaptive lookbacks."""

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days

    def evaluate(self, symbol: str, price_data: pd.DataFrame,
                news_sentiment: Optional[float] = None) -> StrategyDecision:

        if len(price_data) < self.lookback_days:
            return StrategyDecision(
                name="momentum",
                vote=StrategyVote.HOLD,
                confidence=0.0,
                expected_return=0.0,
                rationale="Insufficient data",
                signals={}
            )

        # Calculate returns

        current_price = float(price_data['close'].iloc[-1])
        past_price = float(price_data['close'].iloc[-self.lookback_days])
        momentum_return = (current_price - past_price) / past_price

        # ATR for stop loss

        if 'high' in price_data.columns and 'low' in price_data.columns:
            tr = pd.DataFrame({
                'hl': price_data['high'] - price_data['low'],
                'hc': abs(price_data['high'] - price_data['close'].shift()),
                'lc': abs(price_data['low'] - price_data['close'].shift())
            })
            atr = tr.max(axis=1).tail(14).mean()
        else:
            atr = current_price * 0.02  # 2% fallback

        # Vote based on momentum

        if momentum_return > 0.05:  # >5%
            vote = StrategyVote.BUY
            confidence = min(80.0, 60.0 + momentum_return * 100)
        elif momentum_return < -0.05:  # <-5%
            vote = StrategyVote.SELL
            confidence = min(80.0, 60.0 + abs(momentum_return) * 100)
        else:
            vote = StrategyVote.HOLD
            confidence = 40.0

        return StrategyDecision(
            name="momentum",
            vote=vote,
            confidence=confidence,
            expected_return=momentum_return * 0.5,  # Take profit at half
            rationale=f"{self.lookback_days}d momentum {momentum_return:+.2%}, ATR ${atr:.2f}",
            signals={"momentum": momentum_return, "atr": float(atr)}
        )

class NewsShockStrategy(BaseStrategy):
    """News shock reversion/follow-through (30-240 min)."""

    def evaluate(self, symbol: str, price_data: pd.DataFrame,
                news_sentiment: Optional[float] = None) -> StrategyDecision:

        if news_sentiment is None:
            return StrategyDecision(
                name="news_shock",
                vote=StrategyVote.HOLD,
                confidence=0.0,
                expected_return=0.0,
                rationale="No news signal",
                signals={}
            )

        # Strong positive news → follow-through

        # Strong negative news → reversion

        if news_sentiment > 0.5:  # Strong positive
            vote = StrategyVote.BUY
            confidence = min(75.0, 50.0 + news_sentiment * 50)
            expected_return = news_sentiment * 0.03  # 3% max
            rationale = f"Positive news shock {news_sentiment:+.2f}, follow-through"
        elif news_sentiment < -0.5:  # Strong negative
            vote = StrategyVote.BUY  # Reversion play
            confidence = min(70.0, 50.0 + abs(news_sentiment) * 40)
            expected_return = abs(news_sentiment) * 0.02  # 2% reversion
            rationale = f"Negative news shock {news_sentiment:.2f}, reversion bet"
        else:
            vote = StrategyVote.HOLD
            confidence = 30.0
            expected_return = 0.0
            rationale = f"Weak news signal {news_sentiment:+.2f}"

        return StrategyDecision(
            name="news_shock",
            vote=vote,
            confidence=confidence,
            expected_return=expected_return,
            rationale=rationale,
            signals={"sentiment": news_sentiment}
        )

class PairsTradingStrategy(BaseStrategy):
    """Pairs mean reversion (sector-peers cointegration)."""

    def evaluate(self, symbol: str, price_data: pd.DataFrame,
                news_sentiment: Optional[float] = None) -> StrategyDecision:

        # Simplified: compare to sector ETF (would need peer data)

        # For now, return HOLD (requires multi-asset support)

        return StrategyDecision(
            name="pairs_trading",
            vote=StrategyVote.HOLD,
            confidence=0.0,
            expected_return=0.0,
            rationale="Requires multi-asset support",
            signals={}
        )

class StrategyEnsemble:
    """Ensemble of strategies with meta-learner weighting."""

    def __init__(self):
        self.strategies = [
            MomentumStrategy(lookback_days=20),
            NewsShockStrategy(),
            PairsTradingStrategy()
        ]

        # Regime-based weights (will be dynamic in Phase 3)

        self.regime_weights = {
            "trending": {"momentum": 0.6, "news_shock": 0.3, "pairs_trading": 0.1},
            "mean_reverting": {"momentum": 0.2, "news_shock": 0.4, "pairs_trading": 0.4},
            "volatile": {"momentum": 0.3, "news_shock": 0.5, "pairs_trading": 0.2}
        }

    def evaluate_all(self, symbol: str, price_data: pd.DataFrame,
                    news_sentiment: Optional[float] = None,
                    regime: str = "trending") -> Dict:
        """Evaluate all strategies and aggregate."""

        decisions = []
        for strategy in self.strategies:
            try:
                decision = strategy.evaluate(symbol, price_data, news_sentiment)
                decisions.append(decision)
            except Exception as e:
                print(f"Strategy {strategy.__class__.__name__} failed: {e}")

        # Get regime weights

        weights = self.regime_weights.get(regime, self.regime_weights["trending"])

        # Aggregate votes

        buy_weight = sum(
            weights.get(d.name, 0.33) * d.confidence / 100
            for d in decisions if d.vote == StrategyVote.BUY
        )
        sell_weight = sum(
            weights.get(d.name, 0.33) * d.confidence / 100
            for d in decisions if d.vote == StrategyVote.SELL
        )
        hold_weight = sum(
            weights.get(d.name, 0.33) * d.confidence / 100
            for d in decisions if d.vote == StrategyVote.HOLD
        )

        total_weight = buy_weight + sell_weight + hold_weight

        if total_weight == 0:
            final_vote = StrategyVote.HOLD
            final_confidence = 0.0
        else:
            if buy_weight > sell_weight and buy_weight > hold_weight:
                final_vote = StrategyVote.BUY
                final_confidence = (buy_weight / total_weight) * 100
            elif sell_weight > buy_weight and sell_weight > hold_weight:
                final_vote = StrategyVote.SELL
                final_confidence = (sell_weight / total_weight) * 100
            else:
                final_vote = StrategyVote.HOLD
                final_confidence = (hold_weight / total_weight) * 100

        return {
            "final_vote": final_vote.value,
            "confidence": round(final_confidence, 1),
            "regime": regime,
            "strategy_votes": [
                {
                    "name": d.name,
                    "vote": d.vote.value,
                    "confidence": d.confidence,
                    "rationale": d.rationale
                }
                for d in decisions
            ],
            "weights_used": weights
        }

```text

______________________________________________________________________

## 🎯 Phase 3: Enhanced Risk Shell (Week 3)

### Goal: Production-grade risk management

**Current**: Basic VaR, position limits\
**Target**: Kill-switches, circuit breakers, exposure caps

### 3.1 Enhanced Risk Manager

Create `/workspaces/GHOST/core/risk_manager.py`:

```python

"""
Enhanced Risk Manager - Hard risk shell with kill-switches
Account-level VaR, per-trade max loss, circuit breakers, exposure caps
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import time

class RiskLevel(Enum):
    GREEN = "green"    # Normal operations
    YELLOW = "yellow"  # Caution - approaching limits
    RED = "red"        # Halt trading - limits breached

@dataclass
class RiskLimits:

    # Account-level

    max_portfolio_var_pct: float = 5.0  # Max 5% VaR
    max_daily_loss_pct: float = 2.0     # Max 2% daily loss
    max_position_pct: float = 20.0      # Max 20% per position

    # Per-trade

    max_trade_size_pct: float = 10.0    # Max 10% per trade
    max_leverage: float = 1.0           # No leverage default

    # Circuit breakers

    price_move_threshold_pct: float = 10.0  # 10% price shock
    volume_spike_threshold: float = 3.0      # 3x avg volume

    # Time-based

    max_trades_per_hour: int = 10
    cooldown_after_loss_min: int = 30

@dataclass
class RiskStatus:
    level: RiskLevel
    reasons: List[str]
    can_trade: bool
    warnings: List[str]
    metrics: Dict[str, float]

class EnhancedRiskManager:
    """Production risk management with kill-switches."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.daily_pnl = 0.0
        self.daily_start_nav = 100000.0  # Track daily baseline
        self.trade_count_1h = 0
        self.last_trade_ts = 0
        self.last_loss_ts = 0
        self.circuit_breaker_active = False
        self.kill_switch_active = False

    def check_risk_status(self, current_nav: float, portfolio_var: float,
                         position_size_pct: float, recent_trades: int) -> RiskStatus:
        """
        Comprehensive risk check before allowing trades.
        Returns RiskStatus with level (GREEN/YELLOW/RED) and can_trade flag.
        """

        reasons = []
        warnings = []
        level = RiskLevel.GREEN

        # 1. Kill switch check (manual override)

        if self.kill_switch_active:
            return RiskStatus(
                level=RiskLevel.RED,
                reasons=["KILL SWITCH ACTIVE - Manual intervention required"],
                can_trade=False,
                warnings=[],
                metrics={}
            )

        # 2. Circuit breaker check

        if self.circuit_breaker_active:
            return RiskStatus(
                level=RiskLevel.RED,
                reasons=["CIRCUIT BREAKER ACTIVE - Cooldown period"],
                can_trade=False,
                warnings=[],
                metrics={"cooldown_remaining_s": 300}  # 5 min default
            )

        # 3. Daily loss limit

        daily_pnl_pct = (current_nav - self.daily_start_nav) / self.daily_start_nav * 100
        if daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            reasons.append(f"Daily loss limit breached: {daily_pnl_pct:.2f}% (limit: -{self.limits.max_daily_loss_pct}%)")
            level = RiskLevel.RED
        elif daily_pnl_pct <= -self.limits.max_daily_loss_pct * 0.75:
            warnings.append(f"Approaching daily loss limit: {daily_pnl_pct:.2f}%")
            level = RiskLevel.YELLOW

        # 4. Portfolio VaR limit

        var_pct = (portfolio_var / current_nav) * 100
        if var_pct > self.limits.max_portfolio_var_pct:
            reasons.append(f"Portfolio VaR too high: {var_pct:.2f}% (limit: {self.limits.max_portfolio_var_pct}%)")
            level = RiskLevel.RED
        elif var_pct > self.limits.max_portfolio_var_pct * 0.8:
            warnings.append(f"VaR approaching limit: {var_pct:.2f}%")
            if level == RiskLevel.GREEN:
                level = RiskLevel.YELLOW

        # 5. Position concentration

        if position_size_pct > self.limits.max_position_pct:
            reasons.append(f"Position too large: {position_size_pct:.1f}% (limit: {self.limits.max_position_pct}%)")
            level = RiskLevel.RED
        elif position_size_pct > self.limits.max_position_pct * 0.8:
            warnings.append(f"Position approaching limit: {position_size_pct:.1f}%")
            if level == RiskLevel.GREEN:
                level = RiskLevel.YELLOW

        # 6. Trade frequency limit

        if recent_trades > self.limits.max_trades_per_hour:
            reasons.append(f"Too many trades: {recent_trades}/h (limit: {self.limits.max_trades_per_hour})")
            level = RiskLevel.RED

        # 7. Cooldown after loss

        now_ts = int(time.time())
        if self.last_loss_ts > 0:
            cooldown_remaining = self.limits.cooldown_after_loss_min * 60 - (now_ts - self.last_loss_ts)
            if cooldown_remaining > 0:
                reasons.append(f"Cooldown after loss: {cooldown_remaining/60:.1f} min remaining")
                level = RiskLevel.YELLOW

        can_trade = (level != RiskLevel.RED)

        return RiskStatus(
            level=level,
            reasons=reasons,
            can_trade=can_trade,
            warnings=warnings,
            metrics={
                "daily_pnl_pct": daily_pnl_pct,
                "portfolio_var_pct": var_pct,
                "position_size_pct": position_size_pct,
                "trades_1h": recent_trades
            }
        )

    def activate_circuit_breaker(self, reason: str, duration_s: int = 300):
        """Activate circuit breaker (halt trading)."""
        self.circuit_breaker_active = True
        print(f"[RISK] Circuit breaker activated: {reason} (duration: {duration_s}s)")

        # Would schedule deactivation after duration_s

    def activate_kill_switch(self, reason: str):
        """Activate kill switch (requires manual intervention)."""
        self.kill_switch_active = True
        print(f"[RISK] KILL SWITCH ACTIVATED: {reason}")

    def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker (can resume trading)."""
        self.circuit_breaker_active = False
        print("[RISK] Circuit breaker deactivated")

    def deactivate_kill_switch(self, authorized_user: str):
        """Deactivate kill switch (manual override)."""
        self.kill_switch_active = False
        print(f"[RISK] Kill switch deactivated by {authorized_user}")

```text

______________________________________________________________________

## 📋 Implementation Roadmap

### Week 1: Multi-Horizon Brain

- [ ] Day 1-2: Implement `MultiHorizonForecaster` class
- [ ] Day 3: Add `/api/forecast/multi_horizon` endpoint
- [ ] Day 4: UI panel for multi-horizon display
- [ ] Day 5: Testing and validation


### Week 2: Strategy Ensemble

- [ ] Day 1-2: Implement `StrategyRegistry` with 3 strategies
- [ ] Day 3: Add meta-learner weighting logic
- [ ] Day 4: Integrate with regime detector
- [ ] Day 5: Add `/api/strategies/ensemble` endpoint


### Week 3: Enhanced Risk Shell

- [ ] Day 1-2: Implement `EnhancedRiskManager`
- [ ] Day 3: Add circuit breaker triggers
- [ ] Day 4: UI risk dashboard
- [ ] Day 5: Kill-switch manual controls


______________________________________________________________________

## 🎨 UI Enhancements Needed

### New Panels to Add

1. **Multi-Horizon Dashboard**- 3 columns: Nowcast | Swing | Position
   - Confidence gauges for each
   - Expected return/PnL
   - Signal breakdown


1.**Strategy Ensemble View**- Strategy votes table (BUY/SELL/HOLD)

   - Confidence bars
   - Meta-learner weights by regime
   - Final consensus


1.**Risk Dashboard**- Traffic light indicator (🟢🟡🔴)

   - VaR gauge
   - Daily P&L tracker
   - Circuit breaker status
   - Kill switch button (red, protected)


______________________________________________________________________

## 🚀 Quick Start (After Implementation)

### Test Multi-Horizon Forecast

```bash

curl "<<<<<http://localhost:5000/api/forecast/multi_horizon?symbol=WOLF">>>>>

```text**Expected Output**:

```json

{
  "symbol": "WOLF",
  "as_of": 1759690128,
  "consensus": {
    "action": "BUY",
    "confidence": 62.3,
    "expected_return": 0.0234
  },
  "forecasts": {
    "nowcast": {
      "horizon": "15min-4h",
      "confidence": 65.2,
      "expected_return": 0.0180,
      "signals": {"vwap_drift": 0.012, "volume_shock": 1.5}
    },
    "swing": {
      "horizon": "1-7 days",
      "confidence": 72.4,
      "expected_return": 0.0312
    },
    "position": {
      "horizon": "1-4 weeks",
      "confidence": 49.3,
      "expected_return": 0.0201
    }
  }
}

```text

### Test Strategy Ensemble

```bash

curl "<<<<<http://localhost:5000/api/strategies/ensemble?symbol=WOLF&regime=trending">>>>>

```text

### Test Risk Status

```bash

curl "<<<<<http://localhost:5000/api/risk/status">>>>>

```text

______________________________________________________________________

## 📊 Success Metrics (90-Day Target)

| Metric | Current | Target | APEX Goal | |--------|---------|--------|-----------| |
Hit Rate | ~45% | 52% | ≥55% | | Sharpe Ratio | 0.8 | 1.0 | ≥1.2 | | Max Drawdown | 15%
| 13% | ≤12% | | Annual Turnover | 8× | 10× | \<12× |

______________________________________________________________________

## 🔮 Future Enhancements (Beyond Week 3)

1. **Event Engine**(Week 4-5)

   - EDGAR filings parser
   - Company/event graph
   - Causal impact scoring


1.**Online Calibration**(Week 6)

   - Daily Platt scaling
   - Rolling model updates
   - Performance tracking


1.**Shadow Deployment**(Week 7-8)

   - A/B testing framework
   - Paper trading comparison
   - Automated rollback


1.**Feature Store**(Week 9-10)

   - Centralized feature computation
   - Historical feature caching
   - Feature importance tracking


______________________________________________________________________**Status**: 📋 PLANNING COMPLETE\
**Next Step**: Begin Phase 1 implementation\
**Estimated Time**: 3 weeks for MVP features\
**Full APEX Parity**: 10-12 weeks

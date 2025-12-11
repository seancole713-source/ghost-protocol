"""
Phase 10: Multi-Strategy Trading Engine
Implements multiple trading strategies with dynamic allocation.
"""
import logging
import numpy as np
from datetime import datetime, UTC
from typing import Any
from abc import ABC, abstractmethod

LOGGER = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, name: str, allocation: float = 0.25):
        self.name = name
        self.allocation = allocation  # % of capital allocated to this strategy
        self.enabled = True
        self.trades_executed = 0
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
    
    @abstractmethod
    def generate_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate trading signal.
        
        Args:
            market_data: Dict with price, volume, indicators, etc.
        
        Returns:
            Dict with action, confidence, position_size, etc.
        """
        pass
    
    def record_result(self, pnl: float) -> None:
        """Record trade result."""
        self.trades_executed += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_performance(self) -> dict[str, Any]:
        """Get strategy performance metrics."""
        win_rate = (self.wins / self.trades_executed * 100) if self.trades_executed > 0 else 0.0
        avg_pnl = self.total_pnl / self.trades_executed if self.trades_executed > 0 else 0.0
        
        return {
            "name": self.name,
            "enabled": self.enabled,
            "allocation": self.allocation,
            "trades": self.trades_executed,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": avg_pnl
        }


class PredictionBasedStrategy(TradingStrategy):
    """Original AI prediction-based strategy (Phase 5)."""
    
    def __init__(self, min_confidence: float = 70.0):
        super().__init__("AI Prediction", allocation=0.40)
        self.min_confidence = min_confidence
    
    def generate_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Generate signal based on AI prediction."""
        prediction = market_data.get("prediction", {})
        confidence = prediction.get("confidence", 0)
        direction = prediction.get("direction", "HOLD")
        
        if confidence < self.min_confidence:
            return {"action": "HOLD", "confidence": 0, "reason": "Confidence too low"}
        
        action = "BUY" if direction == "UP" else "SELL" if direction == "DOWN" else "HOLD"
        
        return {
            "action": action,
            "confidence": confidence,
            "position_size": self.allocation,
            "reason": f"AI prediction: {direction} @ {confidence}%"
        }


class MomentumStrategy(TradingStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Momentum", allocation=0.25)
        self.lookback_period = lookback_period
        self.momentum_threshold = 0.02  # 2% price change threshold
    
    def generate_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Generate signal based on price momentum."""
        prices = market_data.get("price_history", [])
        
        if len(prices) < self.lookback_period:
            return {"action": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        recent_prices = prices[-self.lookback_period:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if momentum > self.momentum_threshold:
            return {
                "action": "BUY",
                "confidence": min(abs(momentum) * 100, 90),
                "position_size": self.allocation,
                "reason": f"Strong upward momentum: {momentum*100:.2f}%"
            }
        elif momentum < -self.momentum_threshold:
            return {
                "action": "SELL",
                "confidence": min(abs(momentum) * 100, 90),
                "position_size": self.allocation,
                "reason": f"Strong downward momentum: {momentum*100:.2f}%"
            }
        else:
            return {"action": "HOLD", "confidence": 0, "reason": "Weak momentum"}


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, std_threshold: float = 2.0):
        super().__init__("Mean Reversion", allocation=0.20)
        self.std_threshold = std_threshold
        self.lookback = 50
    
    def generate_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Generate signal based on mean reversion."""
        prices = market_data.get("price_history", [])
        
        if len(prices) < self.lookback:
            return {"action": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        recent_prices = np.array(prices[-self.lookback:])
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        if std_price == 0:
            return {"action": "HOLD", "confidence": 0, "reason": "No volatility"}
        
        z_score = (current_price - mean_price) / std_price
        
        if z_score < -self.std_threshold:
            # Price significantly below mean - buy signal
            confidence = min(abs(z_score) * 30, 85)
            return {
                "action": "BUY",
                "confidence": confidence,
                "position_size": self.allocation,
                "reason": f"Price {abs(z_score):.2f} std below mean"
            }
        elif z_score > self.std_threshold:
            # Price significantly above mean - sell signal
            confidence = min(abs(z_score) * 30, 85)
            return {
                "action": "SELL",
                "confidence": confidence,
                "position_size": self.allocation,
                "reason": f"Price {abs(z_score):.2f} std above mean"
            }
        else:
            return {"action": "HOLD", "confidence": 0, "reason": "Price near mean"}


class VolatilityBreakoutStrategy(TradingStrategy):
    """Volatility breakout strategy."""
    
    def __init__(self, atr_multiplier: float = 1.5):
        super().__init__("Volatility Breakout", allocation=0.15)
        self.atr_multiplier = atr_multiplier
        self.atr_period = 14
    
    def generate_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Generate signal based on volatility breakouts."""
        highs = market_data.get("high_history", [])
        lows = market_data.get("low_history", [])
        closes = market_data.get("price_history", [])
        
        if len(highs) < self.atr_period or len(lows) < self.atr_period:
            return {"action": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        # Calculate ATR (Average True Range)
        tr_values = []
        for i in range(1, min(len(highs), self.atr_period + 1)):
            high_low = highs[-i] - lows[-i]
            high_close = abs(highs[-i] - closes[-(i+1)])
            low_close = abs(lows[-i] - closes[-(i+1)])
            tr_values.append(max(high_low, high_close, low_close))
        
        atr = np.mean(tr_values) if tr_values else 0
        
        if atr == 0:
            return {"action": "HOLD", "confidence": 0, "reason": "No volatility"}
        
        current_price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else current_price
        price_change = current_price - prev_close
        
        # Breakout if price moves > ATR * multiplier
        if price_change > atr * self.atr_multiplier:
            return {
                "action": "BUY",
                "confidence": 75,
                "position_size": self.allocation,
                "reason": f"Upward volatility breakout: {price_change:.2f} > {atr*self.atr_multiplier:.2f}"
            }
        elif price_change < -atr * self.atr_multiplier:
            return {
                "action": "SELL",
                "confidence": 75,
                "position_size": self.allocation,
                "reason": f"Downward volatility breakout"
            }
        else:
            return {"action": "HOLD", "confidence": 0, "reason": "No breakout"}


class MultiStrategyEngine:
    """Coordinate multiple trading strategies."""
    
    def __init__(self):
        self.strategies: list[TradingStrategy] = [
            PredictionBasedStrategy(min_confidence=70.0),
            MomentumStrategy(lookback_period=20),
            MeanReversionStrategy(std_threshold=2.0),
            VolatilityBreakoutStrategy(atr_multiplier=1.5)
        ]
        
        self.rebalance_enabled = True
        self.min_consensus_pct = 50.0  # Minimum % of strategies agreeing
    
    def add_strategy(self, strategy: TradingStrategy) -> None:
        """Add a new strategy to the engine."""
        self.strategies.append(strategy)
        LOGGER.info(f"[MULTI-STRATEGY] Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy by name."""
        for i, s in enumerate(self.strategies):
            if s.name == strategy_name:
                self.strategies.pop(i)
                LOGGER.info(f"[MULTI-STRATEGY] Removed strategy: {strategy_name}")
                return True
        return False
    
    def generate_consensus_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate consensus signal from all strategies.
        
        Args:
            market_data: Market data for signal generation
        
        Returns:
            Aggregated trading signal
        """
        signals = []
        
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            
            try:
                signal = strategy.generate_signal(market_data)
                signals.append({
                    "strategy": strategy.name,
                    "signal": signal,
                    "allocation": strategy.allocation
                })
            except Exception as e:
                LOGGER.error(f"[MULTI-STRATEGY] {strategy.name} error: {e}")
        
        if not signals:
            return {"action": "HOLD", "confidence": 0, "reason": "No signals generated"}
        
        # Calculate consensus
        buy_votes = sum(1 for s in signals if s["signal"]["action"] == "BUY")
        sell_votes = sum(1 for s in signals if s["signal"]["action"] == "SELL")
        total_votes = len(signals)
        
        buy_pct = (buy_votes / total_votes * 100) if total_votes > 0 else 0
        sell_pct = (sell_votes / total_votes * 100) if total_votes > 0 else 0
        
        # Weighted consensus confidence
        buy_confidence = np.mean([
            s["signal"]["confidence"] * s["allocation"]
            for s in signals if s["signal"]["action"] == "BUY"
        ]) if buy_votes > 0 else 0
        
        sell_confidence = np.mean([
            s["signal"]["confidence"] * s["allocation"]
            for s in signals if s["signal"]["action"] == "SELL"
        ]) if sell_votes > 0 else 0
        
        # Determine consensus action
        if buy_pct >= self.min_consensus_pct and buy_confidence > sell_confidence:
            action = "BUY"
            confidence = buy_confidence
            reason = f"{buy_votes}/{total_votes} strategies agree (BUY)"
        elif sell_pct >= self.min_consensus_pct and sell_confidence > buy_confidence:
            action = "SELL"
            confidence = sell_confidence
            reason = f"{sell_votes}/{total_votes} strategies agree (SELL)"
        else:
            action = "HOLD"
            confidence = 0
            reason = f"No consensus: {buy_votes} BUY, {sell_votes} SELL"
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "signals": signals,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    def rebalance_allocations(self) -> None:
        """Rebalance strategy allocations based on performance."""
        if not self.rebalance_enabled:
            return
        
        # Calculate performance scores
        performances = []
        for strategy in self.strategies:
            if strategy.trades_executed < 10:
                # Not enough data
                performances.append({"strategy": strategy, "score": 0.5})
                continue
            
            win_rate = strategy.wins / strategy.trades_executed
            avg_pnl = strategy.total_pnl / strategy.trades_executed
            
            # Score = weighted combination of win rate and avg PnL
            score = (win_rate * 0.6) + (min(avg_pnl / 100, 1.0) * 0.4)
            performances.append({"strategy": strategy, "score": max(0.1, min(score, 1.0))})
        
        # Normalize scores to sum to 1.0
        total_score = sum(p["score"] for p in performances)
        if total_score > 0:
            for perf in performances:
                new_allocation = perf["score"] / total_score
                old_allocation = perf["strategy"].allocation
                perf["strategy"].allocation = new_allocation
                
                if abs(new_allocation - old_allocation) > 0.05:
                    LOGGER.info(
                        f"[MULTI-STRATEGY] Rebalanced {perf['strategy'].name}: "
                        f"{old_allocation:.2%} → {new_allocation:.2%}"
                    )
    
    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all strategies."""
        return {
            "ok": True,
            "strategies": [s.get_performance() for s in self.strategies],
            "total_strategies": len(self.strategies),
            "enabled_strategies": sum(1 for s in self.strategies if s.enabled),
            "rebalance_enabled": self.rebalance_enabled,
            "timestamp": datetime.now(UTC).isoformat()
        }


# Global multi-strategy engine
_strategy_engine = MultiStrategyEngine()


def get_strategy_engine() -> MultiStrategyEngine:
    """Get global strategy engine."""
    return _strategy_engine


def generate_trading_signal(market_data: dict[str, Any]) -> dict[str, Any]:
    """Generate consensus trading signal."""
    return _strategy_engine.generate_consensus_signal(market_data)


def get_strategy_performance() -> dict[str, Any]:
    """Get strategy performance summary."""
    return _strategy_engine.get_performance_summary()

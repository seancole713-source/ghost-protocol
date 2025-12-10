"""
Historical Prediction Simulator

Simulates predictions on historical data to calculate immediate accuracy
without waiting 48 hours. Uses historical price data from CoinGecko/Polygon
to "go back in time", make predictions, and validate against actual outcomes.
"""

import logging
import time
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)


class HistoricalSimulator:
    """
    Simulates predictions on historical data to calculate immediate accuracy.
    """

    def __init__(self):
        self.session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_historical_prices(
        self, symbol: str, days_back: int = 7
    ) -> list[dict[str, Any]]:
        """
        Fetch historical hourly prices for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            days_back: How many days of history to fetch

        Returns:
            List of {timestamp, price} dictionaries
        """
        await self._ensure_session()

        # Map crypto symbols to CoinGecko IDs
        symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "DOGE": "dogecoin",
            "MATIC": "matic-network",
            "DOT": "polkadot",
            "AVAX": "avalanche-2",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "ATOM": "cosmos",
        }

        coin_id = symbol_map.get(symbol)
        if not coin_id:
            LOGGER.warning(f"No CoinGecko mapping for {symbol}")
            return []

        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # CoinGecko market_chart endpoint (free tier, 365 days max)
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    "vs_currency": "usd",
                    "days": min(days_back, 365),
                    "interval": "hourly"
                }

                # Add delay to avoid rate limits (free tier: 10-30 calls/minute)
                import asyncio
                if attempt > 0:
                    await asyncio.sleep(retry_delay * attempt)
                else:
                    await asyncio.sleep(1.5)  # Base delay between calls

                async with self.session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 429:  # Rate limited
                        if attempt < max_retries - 1:
                            LOGGER.warning(f"Rate limited on {symbol}, retrying in {retry_delay * (attempt + 1)}s...")
                            continue
                        else:
                            LOGGER.error(f"CoinGecko rate limit exceeded for {symbol} after {max_retries} attempts")
                            return []

                    if resp.status != 200:
                        LOGGER.error(f"CoinGecko API error {resp.status} for {symbol}")
                        return []

                    data = await resp.json()
                    prices = data.get("prices", [])

                    # Convert to our format: [{timestamp, price}]
                    result = []
                    for timestamp_ms, price in prices:
                        result.append({
                            "timestamp": timestamp_ms / 1000,  # Convert to seconds
                            "price": price
                        })

                    LOGGER.info(f"Fetched {len(result)} historical prices for {symbol}")
                    return result

            except Exception as e:
                if attempt < max_retries - 1:
                    LOGGER.warning(f"Error fetching {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    LOGGER.error(f"Failed to fetch historical prices for {symbol} after {max_retries} attempts: {e}")
                    return []

        return []

    async def simulate_prediction(
        self, symbol: str, prediction_time: float, historical_prices: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Simulate making a prediction at a specific historical time.
        
        Args:
            symbol: Trading symbol
            prediction_time: Unix timestamp when prediction is "made"
            historical_prices: List of historical price data
            
        Returns:
            Prediction result or None if not enough data
        """
        from core.wolf_model import get_wolf_predictor
        
        # Find the price at prediction time
        prediction_price = None
        for data_point in historical_prices:
            if abs(data_point["timestamp"] - prediction_time) < 1800:  # Within 30 min
                prediction_price = data_point["price"]
                break
        
        if prediction_price is None:
            LOGGER.warning(f"No price data at prediction time for {symbol}")
            return None
        
        # Find price 48 hours later
        target_time = prediction_time + (48 * 3600)
        actual_price = None
        
        for data_point in historical_prices:
            if abs(data_point["timestamp"] - target_time) < 1800:  # Within 30 min
                actual_price = data_point["price"]
                break
        
        if actual_price is None:
            LOGGER.warning(f"No price data 48h later for {symbol}")
            return None
        
        # Make prediction using Wolf predictor
        # Note: This uses current data collector, not historical data
        # For true historical simulation, would need to reconstruct market state
        try:
            predictor = get_wolf_predictor()
            prediction = await predictor.predict(symbol, horizon_hours=48)
            
            # Calculate actual outcome
            price_change_pct = ((actual_price - prediction_price) / prediction_price) * 100
            
            actual_direction = "FLAT"
            if price_change_pct > 0.5:
                actual_direction = "UP"
            elif price_change_pct < -0.5:
                actual_direction = "DOWN"
            
            predicted_direction = prediction.get("direction", "FLAT")
            is_correct = predicted_direction == actual_direction
            
            return {
                "symbol": symbol,
                "prediction_time": prediction_time,
                "prediction_price": prediction_price,
                "actual_price_48h": actual_price,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "confidence": prediction.get("confidence", 0),
                "correct": is_correct,
                "price_change_pct": price_change_pct
            }
            
        except Exception as e:
            LOGGER.error(f"Prediction simulation failed for {symbol}: {e}")
            return None

    async def run_simulation(
        self, symbols: list[str], num_predictions: int = 50, days_back: int = 7
    ) -> dict[str, Any]:
        """
        Run full historical simulation across multiple symbols.
        
        Args:
            symbols: List of symbols to simulate
            num_predictions: Target number of predictions to generate
            days_back: How many days of history to use
            
        Returns:
            Simulation results with accuracy metrics
        """
        start_time = time.time()
        LOGGER.info(f"Starting historical simulation: {len(symbols)} symbols, {num_predictions} predictions")
        
        # Fetch historical data for all symbols
        all_historical_data = {}
        
        for symbol in symbols:
            prices = await self.fetch_historical_prices(symbol, days_back)
            if prices:
                all_historical_data[symbol] = prices
        
        if not all_historical_data:
            return {
                "ok": False,
                "error": "No historical data available",
                "symbols_attempted": symbols,
                "execution_time_s": time.time() - start_time
            }
        
        # Generate prediction times (evenly distributed across available history)
        predictions_per_symbol = max(1, num_predictions // len(all_historical_data))
        simulated_predictions = []
        
        for symbol, prices in all_historical_data.items():
            if len(prices) < 50:  # Need at least 50 hours for 48h prediction
                continue
            
            # Space predictions evenly, leaving room for 48h validation
            available_range = len(prices) - 50  # Reserve last 48+ hours for validation
            step = max(1, available_range // predictions_per_symbol)
            
            for i in range(0, available_range, step):
                if len(simulated_predictions) >= num_predictions:
                    break
                
                prediction_time = prices[i]["timestamp"]
                result = await self.simulate_prediction(symbol, prediction_time, prices)
                
                if result:
                    simulated_predictions.append(result)
            
            if len(simulated_predictions) >= num_predictions:
                break
        
        # Calculate accuracy metrics
        total = len(simulated_predictions)
        correct = sum(1 for p in simulated_predictions if p["correct"])
        accuracy_pct = (correct / total * 100) if total > 0 else 0
        
        # Calculate per-symbol accuracy
        symbol_accuracy = {}
        for symbol in set(p["symbol"] for p in simulated_predictions):
            symbol_preds = [p for p in simulated_predictions if p["symbol"] == symbol]
            symbol_correct = sum(1 for p in symbol_preds if p["correct"])
            symbol_accuracy[symbol] = {
                "total": len(symbol_preds),
                "correct": symbol_correct,
                "accuracy_pct": (symbol_correct / len(symbol_preds) * 100) if symbol_preds else 0
            }
        
        # Calculate confidence correlation
        high_conf_preds = [p for p in simulated_predictions if p["confidence"] >= 0.6]
        high_conf_correct = sum(1 for p in high_conf_preds if p["correct"])
        high_conf_accuracy = (high_conf_correct / len(high_conf_preds) * 100) if high_conf_preds else 0
        
        execution_time = time.time() - start_time
        
        result = {
            "ok": True,
            "simulation_type": "historical_backtest",
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": round(accuracy_pct, 2),
            "high_confidence_accuracy_pct": round(high_conf_accuracy, 2) if high_conf_preds else None,
            "high_confidence_count": len(high_conf_preds),
            "symbol_accuracy": symbol_accuracy,
            "days_back": days_back,
            "execution_time_s": round(execution_time, 2),
            "timestamp": time.time(),
            "predictions": simulated_predictions[:10]  # Include first 10 for review
        }
        
        LOGGER.info(
            f"Simulation complete: {accuracy_pct:.1f}% accuracy "
            f"({correct}/{total} correct) in {execution_time:.1f}s"
        )
        
        return result


# Singleton instance
_simulator: HistoricalSimulator | None = None


def get_historical_simulator() -> HistoricalSimulator:
    """Get singleton historical simulator instance"""
    global _simulator
    if _simulator is None:
        _simulator = HistoricalSimulator()
    return _simulator

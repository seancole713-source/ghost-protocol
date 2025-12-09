"""
Ghost Data-Enhanced Prediction System
=====================================
Integrates multi-source data collection into Ghost's prediction pipeline.
Automatically enriches predictions with market data, sentiment, and macro indicators.
"""

import asyncio
import logging
import os
from typing import Any
from datetime import datetime

from core.data_collector import DataCollector, feed_ghost_prediction, VIPCoinData

LOGGER = logging.getLogger("ghost.data_enhanced_predictor")

# VIP coins to track
VIP_COINS = ['WEPE', 'LILPEPE', 'DORKL', 'SLOTH', 'APC']


class DataEnhancedPredictor:
    """
    Wraps Ghost's prediction system with automatic data enrichment.
    """
    
    def __init__(self):
        self.data_enabled = os.getenv("GHOST_DATA_ENRICHMENT", "1") == "1"
        self.collector = None
        
    async def __aenter__(self):
        if self.data_enabled:
            self.collector = DataCollector()
            await self.collector.__aenter__()
        return self
        
    async def __aexit__(self, *args):
        if self.collector:
            await self.collector.__aexit__(*args)
    
    async def predict_with_data(self, symbol: str, horizon_h: int = 48) -> dict[str, Any]:
        """
        Make prediction with full data enrichment.
        
        Returns:
            {
                'symbol': str,
                'direction': 'UP' | 'DOWN' | 'FLAT',
                'confidence': float (0-1),
                'horizon_h': int,
                'data_quality': float (0-1),
                'features': dict,
                'timestamp': float
            }
        """
        LOGGER.info(f"Making data-enhanced prediction for {symbol}")
        
        if not self.data_enabled or not self.collector:
            LOGGER.warning("Data enrichment disabled, using basic prediction")
            return await self._basic_prediction(symbol, horizon_h)
        
        try:
            # Collect comprehensive market data
            features = await feed_ghost_prediction(symbol)
            
            # Make prediction using enriched features
            prediction = await self._make_enriched_prediction(symbol, features, horizon_h)
            
            LOGGER.info(f"Prediction for {symbol}: {prediction['direction']} "
                       f"(confidence: {prediction['confidence']:.2%}, "
                       f"data quality: {prediction['data_quality']:.2%})")
            
            return prediction
            
        except Exception as e:
            LOGGER.error(f"Data-enhanced prediction failed for {symbol}: {e}")
            # Fallback to basic prediction
            return await self._basic_prediction(symbol, horizon_h)
    
    async def _make_enriched_prediction(self, symbol: str, features: dict, horizon_h: int) -> dict:
        """
        Use enriched features to make smarter prediction.
        """
        import time
        
        # Extract key signals (with None handling)
        rsi = features.get('rsi_14') or 50
        trend = features.get('trend') or 'SIDEWAYS'
        sentiment = features.get('sentiment') or 0
        fear_greed = features.get('fear_greed') or 50
        volume = features.get('volume_24h') or 0
        
        # Decision logic (simplified - replace with actual ML model)
        direction = 'FLAT'
        confidence = 0.5
        
        # Strong signals
        bullish_score = 0
        bearish_score = 0
        
        # RSI signals
        if rsi and rsi < 30:  # Oversold
            bullish_score += 2
        elif rsi and rsi > 70:  # Overbought
            bearish_score += 2
        
        # Trend signals
        if trend == 'UP':
            bullish_score += 2
        elif trend == 'DOWN':
            bearish_score += 2
        
        # Sentiment signals
        if sentiment > 0.3:
            bullish_score += 1
        elif sentiment < -0.3:
            bearish_score += 1
        
        # Fear & Greed signals
        if fear_greed:
            if fear_greed < 25:  # Extreme fear = buy opportunity
                bullish_score += 1
            elif fear_greed > 75:  # Extreme greed = sell opportunity
                bearish_score += 1
        
        # Volume confirmation
        if volume > 0:  # Has volume data
            if bullish_score > bearish_score:
                bullish_score += 1
            elif bearish_score > bullish_score:
                bearish_score += 1
        
        # Make decision
        if bullish_score >= bearish_score + 2:
            direction = 'UP'
            confidence = min(0.9, 0.5 + (bullish_score - bearish_score) * 0.1)
        elif bearish_score >= bullish_score + 2:
            direction = 'DOWN'
            confidence = min(0.9, 0.5 + (bearish_score - bullish_score) * 0.1)
        else:
            direction = 'FLAT'
            confidence = 0.5
        
        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'horizon_h': horizon_h,
            'data_quality': features.get('data_quality_score', 0),
            'features': features,
            'signals': {
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'rsi': rsi,
                'trend': trend,
                'sentiment': sentiment,
                'fear_greed': fear_greed
            },
            'timestamp': time.time()
        }
    
    async def _basic_prediction(self, symbol: str, horizon_h: int) -> dict:
        """Fallback to basic prediction without data enrichment"""
        import time
        
        return {
            'symbol': symbol,
            'direction': 'FLAT',
            'confidence': 0.5,
            'horizon_h': horizon_h,
            'data_quality': 0.0,
            'features': {},
            'timestamp': time.time()
        }
    
    async def track_vip_coins(self) -> list[VIPCoinData]:
        """
        Track VIP coins: WEPE, LILPEPE, DORKL, SLOTH, APC
        Returns detailed intelligence for each.
        """
        if not self.collector:
            return []
        
        vip_data = []
        
        for coin in VIP_COINS:
            try:
                data = await self.collector.get_vip_coin_data(coin)
                if data:
                    vip_data.append(data)
                    
                    # Log key metrics
                    LOGGER.info(f"VIP {coin}: ${data.price:.6f} | "
                               f"Vol: ${data.volume_24h:,.0f} | "
                               f"Liq: ${data.liquidity_usd:,.0f} | "
                               f"24h: {data.price_change_24h_pct:+.1f}%")
            except Exception as e:
                LOGGER.error(f"Error tracking VIP coin {coin}: {e}")
        
        return vip_data


# ============================================================================
# API ENDPOINTS
# ============================================================================

async def api_predict_enhanced(symbol: str, horizon_h: int = 48) -> dict:
    """
    API endpoint: Make data-enhanced prediction.
    
    Usage:
        POST /api/v3/predict/enhanced
        {
            "symbol": "BTC",
            "horizon_h": 48
        }
    """
    async with DataEnhancedPredictor() as predictor:
        return await predictor.predict_with_data(symbol, horizon_h)


async def api_vip_coins() -> dict:
    """
    API endpoint: Get VIP coin intelligence.
    
    Usage:
        GET /api/v3/vip-coins
    """
    async with DataEnhancedPredictor() as predictor:
        vip_data = await predictor.track_vip_coins()
        
        return {
            'ok': True,
            'coins': [
                {
                    'symbol': d.symbol,
                    'price': d.price,
                    'market_cap': d.market_cap,
                    'liquidity': d.liquidity_usd,
                    'volume_24h': d.volume_24h,
                    'price_change_24h_pct': d.price_change_24h_pct,
                    'buy_pressure': d.buy_pressure,
                    'trend_strength': d.trend_strength
                }
                for d in vip_data
            ],
            'timestamp': datetime.now().isoformat()
        }


async def api_market_snapshot(symbol: str) -> dict:
    """
    API endpoint: Get complete market snapshot.
    
    Usage:
        GET /api/v3/market/snapshot?symbol=BTC
    """
    async with DataEnhancedPredictor() as predictor:
        if not predictor.collector:
            return {'ok': False, 'error': 'Data collection disabled'}
        
        snapshot = await predictor.collector.get_complete_snapshot(symbol)
        
        return {
            'ok': True,
            'symbol': snapshot.symbol,
            'price': snapshot.price,
            'volume_24h': snapshot.volume_24h,
            'market_cap': snapshot.market_cap,
            'technical': {
                'rsi_14': snapshot.rsi_14,
                'trend': snapshot.trend,
                'volatility': snapshot.volatility
            },
            'sentiment': {
                'score': snapshot.sentiment_score,
                'news_count_24h': snapshot.news_count_24h,
                'fear_greed': snapshot.fear_greed_index
            },
            'liquidity': {
                'liquidity_usd': snapshot.liquidity_usd,
                'whale_txs_24h': snapshot.whale_txs_24h
            },
            'timestamp': snapshot.timestamp
        }


# ============================================================================
# BATCH PREDICTION
# ============================================================================

async def batch_predict_with_data(symbols: list[str], horizon_h: int = 48) -> list[dict]:
    """
    Make predictions for multiple symbols in parallel.
    """
    async with DataEnhancedPredictor() as predictor:
        tasks = [predictor.predict_with_data(sym, horizon_h) for sym in symbols]
        return await asyncio.gather(*tasks, return_exceptions=False)


# ============================================================================
# TESTING
# ============================================================================

async def test_data_enhanced_prediction():
    """Test the data-enhanced prediction system"""
    print("\n" + "="*70)
    print("GHOST DATA-ENHANCED PREDICTION TEST")
    print("="*70)
    
    symbols = ['BTC', 'ETH', 'SOL']
    
    async with DataEnhancedPredictor() as predictor:
        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"Predicting {symbol}")
            print('='*70)
            
            prediction = await predictor.predict_with_data(symbol)
            
            print(f"\nDirection: {prediction['direction']}")
            print(f"Confidence: {prediction['confidence']:.1%}")
            print(f"Data Quality: {prediction['data_quality']:.1%}")
            print(f"Horizon: {prediction['horizon_h']}h")
            
            if 'signals' in prediction:
                signals = prediction['signals']
                print(f"\nSignals:")
                print(f"  Bullish: {signals['bullish_score']}")
                print(f"  Bearish: {signals['bearish_score']}")
                print(f"  RSI: {signals['rsi']:.1f}" if signals['rsi'] else "  RSI: N/A")
                print(f"  Trend: {signals['trend']}")
                print(f"  Sentiment: {signals['sentiment']:.2f}")
    
    # Test VIP coins
    print(f"\n{'='*70}")
    print("VIP COINS TRACKING")
    print('='*70)
    
    vip_response = await api_vip_coins()
    
    for coin in vip_response['coins']:
        print(f"\n{coin['symbol']}:")
        print(f"  Price: ${coin['price']:.6f}")
        print(f"  24h Change: {coin['price_change_24h_pct']:+.2f}%")
        print(f"  Volume: ${coin['volume_24h']:,.0f}")
        print(f"  Liquidity: ${coin['liquidity']:,.0f}")


if __name__ == "__main__":
    asyncio.run(test_data_enhanced_prediction())

"""
Ghost Multi-Source Data Collector
=================================
Automatically feeds Ghost with:
- Market data (prices, volume, liquidity, whale activity)
- Sentiment data (news, social media, fear/greed)
- Technical indicators (RSI, MACD, trends)
- VIP coin intelligence
- Macro/risk indicators (DXY, VIX, bonds, equities)

All data sources are API-based or scrape-safe (legal compliance).
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import requests
from dataclasses import dataclass, asdict

LOGGER = logging.getLogger("ghost.data_collector")


@dataclass
class MarketSnapshot:
    """Complete market snapshot for a symbol"""
    symbol: str
    timestamp: float
    
    # Price data
    price: float
    volume_24h: float
    market_cap: Optional[float]
    
    # Technical indicators
    rsi_14: Optional[float]
    macd: Optional[float]
    trend: Optional[str]  # UP/DOWN/SIDEWAYS
    volatility: Optional[float]
    
    # Sentiment
    sentiment_score: Optional[float]  # -1 to +1
    news_count_24h: int
    social_mentions_24h: int
    
    # Liquidity & whale activity
    liquidity_usd: Optional[float]
    whale_txs_24h: int
    holder_count: Optional[int]
    
    # Macro context
    fear_greed_index: Optional[int]
    btc_dominance: Optional[float]


@dataclass
class VIPCoinData:
    """VIP coin intelligence (WEPE, LILPEPE, DORKL, SLOTH, APC)"""
    symbol: str
    timestamp: float
    
    # Core metrics
    price: float
    market_cap: float
    liquidity_usd: float
    holder_count: int
    
    # Activity
    transfers_24h: int
    buy_pressure: float  # buy volume / total volume
    sell_pressure: float
    
    # Momentum
    price_change_24h_pct: float
    volume_24h: float
    trend_strength: float  # 0-100
    
    # Social
    reddit_mentions: int
    twitter_mentions: int


class DataCollector:
    """
    Central data collection hub.
    Gathers data from all allowed sources and provides unified interface.
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def __aenter__(self):
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Connection pooling with limits to prevent exhaustion
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,  # Total connections across all hosts
            limit_per_host=30,  # Max connections per host (CoinGecko, Binance, etc.)
            ttl_dns_cache=300,  # Cache DNS for 5 minutes
            force_close=False,  # Reuse connections
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=10, connect=3),
        )
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    # ========================================================================
    # MARKET DATA
    # ========================================================================
    
    async def get_coingecko_data(self, symbol: str) -> Dict[str, Any]:
        """CoinGecko API - price, volume, market cap"""
        cache_key = f"coingecko_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        try:
            # Map common symbols to CoinGecko IDs
            coin_ids = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
                'SOL': 'solana', 'ADA': 'cardano', 'XRP': 'ripple',
                'DOGE': 'dogecoin', 'AVAX': 'avalanche-2', 'DOT': 'polkadot',
                'MATIC': 'matic-network', 'SHIB': 'shiba-inu', 'UNI': 'uniswap'
            }
            
            coin_id = coin_ids.get(symbol, symbol.lower())
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = {
                        'price': data['market_data']['current_price']['usd'],
                        'volume_24h': data['market_data']['total_volume']['usd'],
                        'market_cap': data['market_data']['market_cap']['usd'],
                        'price_change_24h': data['market_data']['price_change_percentage_24h'],
                        'liquidity_score': data['liquidity_score'] if 'liquidity_score' in data else None
                    }
                    self.cache[cache_key] = (time.time(), result)
                    return result
        except Exception as e:
            LOGGER.warning(f"CoinGecko error for {symbol}: {e}")
        
        return {}
    
    async def get_binance_data(self, symbol: str) -> Dict[str, Any]:
        """Binance API - real-time price, volume, order book"""
        try:
            pair = f"{symbol}USDT"
            
            # 24h ticker
            ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={pair}"
            async with self.session.get(ticker_url) as resp:
                if resp.status == 200:
                    ticker = await resp.json()
                    return {
                        'price': float(ticker['lastPrice']),
                        'volume_24h': float(ticker['quoteVolume']),
                        'price_change_pct': float(ticker['priceChangePercent']),
                        'high_24h': float(ticker['highPrice']),
                        'low_24h': float(ticker['lowPrice']),
                        'trade_count': int(ticker['count'])
                    }
        except Exception as e:
            LOGGER.warning(f"Binance error for {symbol}: {e}")
        
        return {}
    
    async def get_dexscreener_data(self, symbol: str) -> Dict[str, Any]:
        """DEXScreener API - DEX liquidity, price, volume"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('pairs'):
                        # Get pair with highest liquidity
                        pairs = sorted(data['pairs'], 
                                     key=lambda x: float(x.get('liquidity', {}).get('usd', 0)),
                                     reverse=True)
                        
                        if pairs:
                            pair = pairs[0]
                            return {
                                'price': float(pair['priceUsd']),
                                'liquidity_usd': float(pair['liquidity']['usd']),
                                'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                                'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                                'txns_24h': int(pair.get('txns', {}).get('h24', {}).get('buys', 0)) + int(pair.get('txns', {}).get('h24', {}).get('sells', 0)),
                                'dex': pair.get('dexId', 'unknown')
                            }
        except Exception as e:
            LOGGER.warning(f"DEXScreener error for {symbol}: {e}")
        
        return {}
    
    # ========================================================================
    # SENTIMENT DATA
    # ========================================================================
    
    async def get_cryptopanic_sentiment(self, symbol: str) -> Dict[str, Any]:
        """CryptoPanic API - news sentiment"""
        try:
            # Get API key from environment (fallback to 'free' if not set)
            api_key = os.getenv("CRYPTOPANIC_API_KEY", "free")
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={symbol}&filter=rising"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    posts = data.get('results', [])
                    
                    # Analyze sentiment from votes
                    positive = sum(1 for p in posts if p.get('votes', {}).get('positive', 0) > p.get('votes', {}).get('negative', 0))
                    negative = sum(1 for p in posts if p.get('votes', {}).get('negative', 0) > p.get('votes', {}).get('positive', 0))
                    total = len(posts)
                    
                    sentiment_score = (positive - negative) / total if total > 0 else 0
                    
                    return {
                        'news_count_24h': total,
                        'sentiment_score': sentiment_score,
                        'positive_news': positive,
                        'negative_news': negative
                    }
        except Exception as e:
            LOGGER.warning(f"CryptoPanic error for {symbol}: {e}")
        
        return {'news_count_24h': 0, 'sentiment_score': 0}
    
    async def get_fear_greed_index(self) -> Optional[int]:
        """Alternative.me Fear & Greed Index"""
        cache_key = "fear_greed"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < 3600:  # 1 hour cache
                return cached_data
        
        try:
            url = "https://api.alternative.me/fng/"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    index = int(data['data'][0]['value'])
                    self.cache[cache_key] = (time.time(), index)
                    return index
        except Exception as e:
            LOGGER.warning(f"Fear & Greed error: {e}")
        
        return None
    
    # ========================================================================
    # TECHNICAL INDICATORS
    # ========================================================================
    
    async def get_coinbase_candles(self, symbol: str, granularity: int = 3600, limit: int = 50) -> Optional[List[List[float]]]:
        """
        Get historical candles from Coinbase Pro.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, SOL, etc.)
            granularity: Candle size in seconds (3600 = 1h, 86400 = 1d)
            limit: Number of candles to fetch (max 300)
        
        Returns:
            List of candles: [[timestamp, low, high, open, close, volume], ...]
            Or None if failed
        """
        try:
            # Coinbase Pro uses product IDs like BTC-USD
            product_id = f"{symbol}-USD"
            
            # Coinbase requires explicit start/end times
            end_time = int(time.time())
            start_time = end_time - (granularity * limit)
            
            url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            params = {
                'start': start_time,
                'end': end_time,
                'granularity': granularity
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    candles = await resp.json()
                    # Coinbase returns: [timestamp, low, high, open, close, volume]
                    # Sort by timestamp ascending (Coinbase returns descending)
                    candles.sort(key=lambda x: x[0])
                    return candles
                elif resp.status == 404:
                    LOGGER.warning(f"Coinbase candles: {product_id} not found")
                else:
                    LOGGER.warning(f"Coinbase candles error: HTTP {resp.status}")
        except Exception as e:
            LOGGER.warning(f"Coinbase candles error for {symbol}: {e}")
        
        return None
    
    async def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate RSI from recent price data.
        Now uses Coinbase Pro as primary source (Binance geoblocked).
        """
        try:
            # Try Coinbase Pro first (US-friendly)
            candles = await self.get_coinbase_candles(symbol, granularity=3600, limit=period + 1)
            
            if candles and len(candles) >= period + 1:
                # Extract close prices (index 4)
                closes = [float(candle[4]) for candle in candles]
                
                # Calculate RSI
                deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                gains = [d if d > 0 else 0 for d in deltas]
                losses = [-d if d < 0 else 0 for d in deltas]
                
                avg_gain = sum(gains) / period
                avg_loss = sum(losses) / period
                
                if avg_loss == 0:
                    return 100.0
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                return rsi
        except Exception as e:
            LOGGER.warning(f"RSI calculation error for {symbol}: {e}")
        
        return None
    
    async def detect_trend(self, symbol: str) -> Optional[str]:
        """
        Detect trend using EMA crossover.
        Now uses Coinbase Pro as primary source (Binance geoblocked).
        """
        try:
            # Get hourly candles for last 50 hours
            candles = await self.get_coinbase_candles(symbol, granularity=3600, limit=50)
            
            if candles and len(candles) >= 21:
                # Extract close prices (index 4)
                closes = [float(candle[4]) for candle in candles]
                
                # Simple EMA calculation
                def ema(data, period):
                    multiplier = 2 / (period + 1)
                    ema_vals = [data[0]]
                    for price in data[1:]:
                        ema_vals.append((price - ema_vals[-1]) * multiplier + ema_vals[-1])
                    return ema_vals[-1]
                
                ema_9 = ema(closes, 9)
                ema_21 = ema(closes, 21)
                
                if ema_9 > ema_21 * 1.01:  # 1% threshold
                    return "UP"
                elif ema_9 < ema_21 * 0.99:
                    return "DOWN"
                else:
                    return "SIDEWAYS"
        except Exception as e:
            LOGGER.warning(f"Trend detection error for {symbol}: {e}")
        
        return None
    
    # ========================================================================
    # MACRO INDICATORS
    # ========================================================================
    
    async def get_macro_indicators(self) -> Dict[str, float]:
        """Get macro indicators: DXY, VIX, TLT, SPY"""
        # Using Yahoo Finance (allowed for price data)
        indicators = {}
        
        try:
            symbols = {
                'DXY': 'DX-Y.NYB',  # Dollar Index
                'VIX': '^VIX',      # Volatility Index
                'SPY': 'SPY',       # S&P 500
                'TLT': 'TLT'        # 20+ Year Treasury Bond
            }
            
            for name, ticker in symbols.items():
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                    async with self.session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price = data['chart']['result'][0]['meta']['regularMarketPrice']
                            indicators[name] = price
                except Exception as e:
                    LOGGER.warning(f"Error fetching {name}: {e}")
        except Exception as e:
            LOGGER.warning(f"Macro indicators error: {e}")
        
        return indicators
    
    # ========================================================================
    # VIP COIN INTELLIGENCE
    # ========================================================================
    
    async def get_vip_coin_data(self, symbol: str) -> Optional[VIPCoinData]:
        """
        Get comprehensive data for VIP coins: WEPE, LILPEPE, DORKL, SLOTH, APC
        """
        try:
            # Get DEXScreener data (best for small cap tokens)
            dex_data = await self.get_dexscreener_data(symbol)
            
            if not dex_data:
                return None
            
            # Get sentiment
            sentiment = await self.get_cryptopanic_sentiment(symbol)
            
            # Calculate buy/sell pressure from DEXScreener
            # (This would require more detailed order book data in production)
            buy_pressure = 0.5  # Placeholder
            sell_pressure = 0.5
            
            return VIPCoinData(
                symbol=symbol,
                timestamp=time.time(),
                price=dex_data.get('price', 0),
                market_cap=dex_data.get('liquidity_usd', 0) * 10,  # Estimate
                liquidity_usd=dex_data.get('liquidity_usd', 0),
                holder_count=0,  # Would need blockchain scan
                transfers_24h=dex_data.get('txns_24h', 0),
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                price_change_24h_pct=dex_data.get('price_change_24h', 0),
                volume_24h=dex_data.get('volume_24h', 0),
                trend_strength=abs(dex_data.get('price_change_24h', 0)),
                reddit_mentions=0,  # Would need Reddit API
                twitter_mentions=sentiment.get('news_count_24h', 0)
            )
        except Exception as e:
            LOGGER.error(f"VIP coin data error for {symbol}: {e}")
            return None
    
    # ========================================================================
    # UNIFIED SNAPSHOT
    # ========================================================================
    
    async def get_complete_snapshot(self, symbol: str) -> MarketSnapshot:
        """
        Get complete market snapshot combining all data sources.
        This is the main entry point for prediction system.
        """
        LOGGER.info(f"Gathering complete snapshot for {symbol}")
        
        # Gather data from multiple sources in parallel
        results = await asyncio.gather(
            self.get_coingecko_data(symbol),
            self.get_binance_data(symbol),
            self.get_dexscreener_data(symbol),
            self.get_cryptopanic_sentiment(symbol),
            self.get_fear_greed_index(),
            self.calculate_rsi(symbol),
            self.detect_trend(symbol),
            return_exceptions=True
        )
        
        coingecko_data, binance_data, dex_data, sentiment_data, fear_greed, rsi, trend = results
        
        # Combine data (prioritize most reliable source for each field)
        price = (binance_data.get('price') or 
                coingecko_data.get('price') or 
                dex_data.get('price') or 0)
        
        volume = (binance_data.get('volume_24h') or 
                 coingecko_data.get('volume_24h') or 
                 dex_data.get('volume_24h') or 0)
        
        return MarketSnapshot(
            symbol=symbol,
            timestamp=time.time(),
            price=price,
            volume_24h=volume,
            market_cap=coingecko_data.get('market_cap'),
            rsi_14=rsi,
            macd=None,  # Could add MACD calculation
            trend=trend,
            volatility=None,  # Could calculate from price history
            sentiment_score=sentiment_data.get('sentiment_score', 0),
            news_count_24h=sentiment_data.get('news_count_24h', 0),
            social_mentions_24h=0,  # Would need Twitter/Reddit API
            liquidity_usd=dex_data.get('liquidity_usd'),
            whale_txs_24h=0,  # Would need blockchain scan
            holder_count=None,
            fear_greed_index=fear_greed,
            btc_dominance=None  # Could add from CoinGecko
        )


# ============================================================================
# INTEGRATION WITH GHOST PREDICTION SYSTEM
# ============================================================================

async def feed_ghost_prediction(symbol: str) -> Dict[str, Any]:
    """
    Collect all data and format for Ghost prediction system.
    Returns enriched feature set for ML model.
    """
    async with DataCollector() as collector:
        # Get complete snapshot
        snapshot = await collector.get_complete_snapshot(symbol)
        
        # Get macro context
        macro = await collector.get_macro_indicators()
        
        # Format for prediction system
        features = {
            # Core price/volume
            'price': snapshot.price,
            'volume_24h': snapshot.volume_24h,
            'market_cap': snapshot.market_cap,
            
            # Technical indicators
            'rsi_14': snapshot.rsi_14,
            'trend': snapshot.trend,
            
            # Sentiment
            'sentiment': snapshot.sentiment_score,
            'news_buzz': snapshot.news_count_24h,
            
            # Market context
            'fear_greed': snapshot.fear_greed_index,
            'liquidity': snapshot.liquidity_usd,
            
            # Macro
            'dxy': macro.get('DXY'),
            'vix': macro.get('VIX'),
            'spy': macro.get('SPY'),
            
            # Metadata
            'timestamp': snapshot.timestamp,
            'data_quality_score': calculate_data_quality(snapshot)
        }
        
        return features


def calculate_data_quality(snapshot: MarketSnapshot) -> float:
    """
    Calculate data quality score (0-1).
    Higher score = more complete data.
    """
    fields = [
        snapshot.price > 0,
        snapshot.volume_24h > 0,
        snapshot.rsi_14 is not None,
        snapshot.trend is not None,
        snapshot.sentiment_score is not None,
        snapshot.fear_greed_index is not None,
        snapshot.liquidity_usd is not None
    ]
    
    return sum(fields) / len(fields)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage"""
    symbols = ['BTC', 'ETH', 'SOL']
    
    async with DataCollector() as collector:
        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"Collecting data for {symbol}")
            print('='*70)
            
            snapshot = await collector.get_complete_snapshot(symbol)
            
            print(f"\nPrice: ${snapshot.price:,.2f}")
            print(f"Volume 24h: ${snapshot.volume_24h:,.0f}")
            print(f"RSI(14): {snapshot.rsi_14:.1f}" if snapshot.rsi_14 else "RSI: N/A")
            print(f"Trend: {snapshot.trend}")
            print(f"Sentiment: {snapshot.sentiment_score:.2f}" if snapshot.sentiment_score else "Sentiment: N/A")
            print(f"Fear & Greed: {snapshot.fear_greed_index}" if snapshot.fear_greed_index else "Fear & Greed: N/A")
            print(f"Data Quality: {calculate_data_quality(snapshot):.0%}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Ghost Protocol - Advanced Crypto Market Structure Analysis
==========================================================
Funding rates, on-chain metrics, liquidity analysis

Better crypto alpha, fewer false signals
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

# API Keys
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")


@dataclass
class FundingRateData:
    """Perpetual funding rate info"""
    symbol: str
    funding_rate: float  # 8h funding rate
    annual_rate_pct: float  # Annualized
    sentiment: str  # BULLISH/BEARISH/NEUTRAL
    timestamp: float


@dataclass
class OnChainMetrics:
    """On-chain blockchain metrics"""
    symbol: str
    whale_movements: int  # Large transfers (>$1M)
    exchange_inflow: float  # $ flowing into exchanges (bearish)
    exchange_outflow: float  # $ leaving exchanges (bullish)
    active_addresses: int
    network_activity_score: float  # 0-100
    timestamp: float


@dataclass
class LiquidityAnalysis:
    """Order book liquidity depth"""
    symbol: str
    bid_depth_1pct: float  # $ within 1% below price
    ask_depth_1pct: float  # $ within 1% above price
    spread_bps: float  # Bid-ask spread in basis points
    liquidity_score: float  # 0-100
    timestamp: float


class CryptoAnalyzer:
    """Advanced crypto market structure analyzer"""
    
    def __init__(self):
        self.funding_cache = {}
        self.onchain_cache = {}
        self.liquidity_cache = {}
        
    def get_funding_rate(self, symbol: str) -> FundingRateData:
        """
        Get perpetual futures funding rate
        
        Positive = longs pay shorts (bullish sentiment)
        Negative = shorts pay longs (bearish sentiment)
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
        
        Returns:
            FundingRateData with current rate
        """
        # Check cache (5 min TTL)
        cached = self.funding_cache.get(symbol)
        if cached and time.time() - cached.timestamp < 300:
            return cached
        
        # Check if we've already detected geo-blocking
        if getattr(self, '_binance_futures_blocked', False):
            return FundingRateData(
                symbol=symbol,
                funding_rate=0.0,
                annual_rate_pct=0.0,
                sentiment="NEUTRAL",
                timestamp=time.time()
            )
        
        try:
            # Binance perpetuals funding rate
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {
                "symbol": f"{symbol}USDT",
                "limit": 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            # Handle geo-blocking silently (451 = Unavailable For Legal Reasons)
            if response.status_code == 451:
                self._binance_futures_blocked = True
                logger.debug(f"Binance Futures geo-blocked (451) - using neutral funding")
                return FundingRateData(
                    symbol=symbol,
                    funding_rate=0.0,
                    annual_rate_pct=0.0,
                    sentiment="NEUTRAL",
                    timestamp=time.time()
                )
            
            if response.status_code != 200:
                raise Exception(f"Binance API error: {response.status_code}")
            
            data = response.json()
            if not data:
                raise Exception("No funding rate data")
            
            latest = data[0]
            funding_rate = float(latest.get("fundingRate", 0))
            
            # Annualize (3 funding periods per day = 1095/year)
            annual_rate_pct = funding_rate * 1095 * 100
            
            # Determine sentiment
            if funding_rate > 0.01:  # >1% per period = very bullish
                sentiment = "BULLISH"
            elif funding_rate < -0.01:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            result = FundingRateData(
                symbol=symbol,
                funding_rate=funding_rate,
                annual_rate_pct=annual_rate_pct,
                sentiment=sentiment,
                timestamp=time.time()
            )
            
            self.funding_cache[symbol] = result
            return result
            
        except Exception as e:
            logger.error(f"Funding rate fetch failed for {symbol}: {e}")
            return FundingRateData(
                symbol=symbol,
                funding_rate=0.0,
                annual_rate_pct=0.0,
                sentiment="NEUTRAL",
                timestamp=time.time()
            )
    
    def get_onchain_metrics(self, symbol: str) -> OnChainMetrics:
        """
        Get on-chain blockchain metrics
        
        Uses Glassnode API for whale movements, exchange flows
        
        Args:
            symbol: Crypto symbol
        
        Returns:
            OnChainMetrics with network activity
        """
        # Check cache (1 hour TTL for on-chain data)
        cached = self.onchain_cache.get(symbol)
        if cached and time.time() - cached.timestamp < 3600:
            return cached
        
        if not GLASSNODE_API_KEY:
            logger.warning("GLASSNODE_API_KEY not set, using mock data")
            return OnChainMetrics(
                symbol=symbol,
                whale_movements=0,
                exchange_inflow=0.0,
                exchange_outflow=0.0,
                active_addresses=0,
                network_activity_score=50.0,
                timestamp=time.time()
            )
        
        try:
            # Glassnode exchange flows
            base_url = "https://api.glassnode.com/v1/metrics"
            headers = {"X-Api-Key": GLASSNODE_API_KEY}
            
            # Exchange inflow (bearish - selling pressure)
            inflow_url = f"{base_url}/transactions/transfers_volume_exchanges_net"
            inflow_params = {
                "a": symbol,
                "i": "24h",
                "c": "USD"
            }
            
            inflow_response = requests.get(inflow_url, headers=headers, params=inflow_params, timeout=10)
            exchange_net_flow = 0.0
            
            if inflow_response.status_code == 200:
                inflow_data = inflow_response.json()
                if inflow_data:
                    exchange_net_flow = float(inflow_data[-1].get("v", 0))
            
            # Active addresses
            active_url = f"{base_url}/addresses/active_count"
            active_params = {
                "a": symbol,
                "i": "24h"
            }
            
            active_response = requests.get(active_url, headers=headers, params=active_params, timeout=10)
            active_addresses = 0
            
            if active_response.status_code == 200:
                active_data = active_response.json()
                if active_data:
                    active_addresses = int(active_data[-1].get("v", 0))
            
            # Network activity score (0-100)
            # High activity + net outflow = bullish
            network_score = min((active_addresses / 10000) * 100, 100)
            
            # Whale movements (mock for now - requires complex on-chain analysis)
            whale_movements = 0
            
            result = OnChainMetrics(
                symbol=symbol,
                whale_movements=whale_movements,
                exchange_inflow=max(exchange_net_flow, 0),
                exchange_outflow=abs(min(exchange_net_flow, 0)),
                active_addresses=active_addresses,
                network_activity_score=network_score,
                timestamp=time.time()
            )
            
            self.onchain_cache[symbol] = result
            return result
            
        except Exception as e:
            logger.error(f"On-chain metrics fetch failed for {symbol}: {e}")
            return OnChainMetrics(
                symbol=symbol,
                whale_movements=0,
                exchange_inflow=0.0,
                exchange_outflow=0.0,
                active_addresses=0,
                network_activity_score=50.0,
                timestamp=time.time()
            )
    
    def get_liquidity_depth(self, symbol: str) -> LiquidityAnalysis:
        """
        Analyze order book liquidity
        
        Args:
            symbol: Crypto symbol
        
        Returns:
            LiquidityAnalysis with depth metrics
        """
        # Check cache (1 min TTL)
        cached = self.liquidity_cache.get(symbol)
        if cached and time.time() - cached.timestamp < 60:
            return cached
        
        try:
            # Binance order book depth
            url = "https://api.binance.com/api/v3/depth"
            params = {
                "symbol": f"{symbol}USDT",
                "limit": 100
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                raise Exception(f"Binance API error: {response.status_code}")
            
            data = response.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            if not bids or not asks:
                raise Exception("No order book data")
            
            # Current price (mid-market)
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate depth within 1%
            price_1pct_below = mid_price * 0.99
            price_1pct_above = mid_price * 1.01
            
            bid_depth = sum(
                float(price) * float(qty)
                for price, qty in bids
                if float(price) >= price_1pct_below
            )
            
            ask_depth = sum(
                float(price) * float(qty)
                for price, qty in asks
                if float(price) <= price_1pct_above
            )
            
            # Spread in basis points
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
            
            # Liquidity score (0-100)
            total_depth = bid_depth + ask_depth
            liquidity_score = min((total_depth / 1000000) * 100, 100)  # $1M = 100 score
            
            result = LiquidityAnalysis(
                symbol=symbol,
                bid_depth_1pct=bid_depth,
                ask_depth_1pct=ask_depth,
                spread_bps=spread_bps,
                liquidity_score=liquidity_score,
                timestamp=time.time()
            )
            
            self.liquidity_cache[symbol] = result
            return result
            
        except Exception as e:
            logger.error(f"Liquidity depth fetch failed for {symbol}: {e}")
            return LiquidityAnalysis(
                symbol=symbol,
                bid_depth_1pct=0.0,
                ask_depth_1pct=0.0,
                spread_bps=0.0,
                liquidity_score=0.0,
                timestamp=time.time()
            )


# Global instance
_crypto_analyzer = None


def get_crypto_analyzer() -> CryptoAnalyzer:
    """Get or create global crypto analyzer"""
    global _crypto_analyzer
    if _crypto_analyzer is None:
        _crypto_analyzer = CryptoAnalyzer()
        logger.info("✅ Crypto analyzer initialized")
    return _crypto_analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("₿ Testing Crypto Analyzer")
    print("=" * 60)
    
    analyzer = get_crypto_analyzer()
    
    # Test funding rates
    btc_funding = analyzer.get_funding_rate("BTC")
    print("\n💰 BTC Funding Rate:")
    print(f"  8h Rate: {btc_funding.funding_rate:.4%}")
    print(f"  Annual: {btc_funding.annual_rate_pct:+.1f}%")
    print(f"  Sentiment: {btc_funding.sentiment}")
    
    # Test on-chain metrics
    btc_onchain = analyzer.get_onchain_metrics("BTC")
    print("\n⛓️ BTC On-Chain Metrics:")
    print(f"  Whale Movements: {btc_onchain.whale_movements}")
    print(f"  Exchange Inflow: ${btc_onchain.exchange_inflow:,.0f}")
    print(f"  Exchange Outflow: ${btc_onchain.exchange_outflow:,.0f}")
    print(f"  Active Addresses: {btc_onchain.active_addresses:,}")
    print(f"  Network Score: {btc_onchain.network_activity_score:.0f}/100")
    
    # Test liquidity
    btc_liquidity = analyzer.get_liquidity_depth("BTC")
    print("\n📊 BTC Liquidity:")
    print(f"  Bid Depth (1%): ${btc_liquidity.bid_depth_1pct:,.0f}")
    print(f"  Ask Depth (1%): ${btc_liquidity.ask_depth_1pct:,.0f}")
    print(f"  Spread: {btc_liquidity.spread_bps:.1f} bps")
    print(f"  Liquidity Score: {btc_liquidity.liquidity_score:.0f}/100")
    
    print("\n✅ Crypto analyzer test complete")

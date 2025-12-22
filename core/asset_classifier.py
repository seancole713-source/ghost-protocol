"""
Ghost Protocol - Asset Classifier
Classifies assets and returns appropriate target/stop percentages.

Created Dec 22, 2025 - Stock targets were 6-7% (too aggressive for large caps)
- AAPL moving 6% in 48h is rare (2-3 times/year)
- Need different targets for: crypto, large cap stocks, volatile stocks

Realistic Move Ranges (48 hours):
| Asset Type        | Daily Avg | 48h Expected | Target | Stop |
|-------------------|-----------|--------------|--------|------|
| Crypto            | 3-5%      | 5-10%        | 6%     | 4.5% |
| Stock (Large Cap) | 0.5-1.5%  | 1-3%         | 2.5%   | 2%   |
| Stock (Mid Cap)   | 1-2%      | 2-4%         | 3.5%   | 2.5% |
| Stock (Volatile)  | 2-5%      | 4-8%         | 5%     | 4%   |
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AssetClassifier:
    """
    Classifies assets and returns appropriate target/stop percentages.
    """
    
    # Large cap stocks (low volatility) - typically move 0.5-1.5% daily
    LARGE_CAP_STOCKS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'BRK.A', 'BRK.B',
        'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'VZ',
        'NFLX', 'INTC', 'CSCO', 'PFE', 'MRK', 'T', 'WMT', 'KO', 'PEP',
        'ABT', 'CVX', 'XOM', 'BAC', 'WFC', 'C', 'ORCL', 'CRM', 'ADBE',
        'ACN', 'COST', 'NKE', 'MCD', 'TMO', 'LLY', 'AVGO', 'TXN', 'QCOM',
        'IBM', 'GE', 'CAT', 'HON', 'MMM', 'BA', 'RTX', 'LMT', 'GS', 'MS',
    }
    
    # Volatile stocks (high volatility even though some are large cap)
    VOLATILE_STOCKS = {
        'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'RIVN',
        'LCID', 'NIO', 'PLTR', 'SOFI', 'HOOD', 'RBLX', 'SNOW', 'CRWD',
        'NET', 'DDOG', 'ZS', 'OKTA', 'MDB', 'U', 'ROKU', 'SQ', 'SHOP',
        'AFRM', 'UPST', 'PATH', 'IONQ', 'SMCI', 'ARM', 'MBLY', 'CELH',
    }
    
    # Crypto assets (highest volatility)
    CRYPTO = {
        'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT', 'MATIC',
        'AVAX', 'LINK', 'UNI', 'AAVE', 'LTC', 'BCH', 'ATOM', 'FIL',
        'NEAR', 'APT', 'ARB', 'OP', 'IMX', 'INJ', 'SUI', 'SEI',
        'PEPE', 'SHIB', 'BONK', 'WIF', 'FLOKI', 'MEME',
        'MKR', 'SNX', 'CRV', 'COMP', 'YFI', 'SUSHI', '1INCH',
        'LRC', 'ENJ', 'SAND', 'MANA', 'AXS', 'GALA', 'ILV',
        'FTM', 'ALGO', 'HBAR', 'VET', 'EOS', 'XLM', 'TRX',
        'KAVA', 'ZEN', 'ZEC', 'DASH', 'XMR', 'ETC',
        'BNB', 'TON', 'LEO', 'OKB', 'CRO',  # Exchange tokens
        # Common trading pairs suffixes
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    }
    
    # Target/stop percentages by asset type and horizon
    # Format: {asset_type: {horizon: (target_pct, stop_pct)}}
    TARGET_MATRIX = {
        'crypto': {
            6: (1.5, 1.5),    # 6h: validated Dec 21-22
            24: (3.5, 3.0),   # 24h: scaled
            48: (6.0, 4.5),   # 48h: production
        },
        'stock_large': {
            6: (0.8, 0.7),    # Large caps barely move in 6h
            24: (1.5, 1.2),
            48: (2.5, 2.0),   # 2.5% is realistic for AAPL in 48h
        },
        'stock_volatile': {
            6: (1.2, 1.0),    # TSLA/NVDA move more
            24: (2.5, 2.0),
            48: (5.0, 4.0),   # 5% is realistic for TSLA in 48h
        },
        'stock_mid': {
            6: (1.0, 0.8),
            24: (2.0, 1.5),
            48: (3.5, 2.5),   # Mid caps between large and volatile
        },
    }
    
    @classmethod
    def get_asset_type(cls, symbol: str) -> str:
        """
        Returns asset type: 'crypto', 'stock_large', 'stock_volatile', 'stock_mid'
        """
        # Clean symbol (remove USDT, USD, etc.)
        clean = symbol.upper().replace('USDT', '').replace('USD', '').replace('/USD', '')
        
        if clean in cls.CRYPTO or symbol.upper() in cls.CRYPTO:
            return 'crypto'
        elif clean in cls.VOLATILE_STOCKS:
            return 'stock_volatile'
        elif clean in cls.LARGE_CAP_STOCKS:
            return 'stock_large'
        else:
            # Default to mid cap for unknown stocks
            return 'stock_mid'
    
    @classmethod
    def is_crypto(cls, symbol: str) -> bool:
        """Check if symbol is a crypto asset"""
        return cls.get_asset_type(symbol) == 'crypto'
    
    @classmethod
    def is_stock(cls, symbol: str) -> bool:
        """Check if symbol is a stock"""
        return cls.get_asset_type(symbol) != 'crypto'
    
    @classmethod
    def get_target_stop(cls, symbol: str, horizon_hours: int = 48) -> Dict:
        """
        Returns appropriate target and stop percentages based on asset type and horizon.
        
        Args:
            symbol: The ticker symbol (e.g., 'BTC', 'AAPL', 'TSLA')
            horizon_hours: Prediction horizon in hours (6, 24, or 48)
            
        Returns:
            dict with: target_pct, stop_pct, asset_type
        """
        asset_type = cls.get_asset_type(symbol)
        
        # Get the matrix for this asset type
        matrix = cls.TARGET_MATRIX.get(asset_type, cls.TARGET_MATRIX['stock_mid'])
        
        # Find the closest horizon bucket
        if horizon_hours <= 6:
            target, stop = matrix.get(6, (1.5, 1.5))
        elif horizon_hours <= 24:
            target, stop = matrix.get(24, (3.5, 3.0))
        else:
            target, stop = matrix.get(48, (6.0, 4.5))
        
        result = {
            'target_pct': target,
            'stop_pct': stop,
            'asset_type': asset_type,
            'symbol': symbol.upper(),
            'horizon_hours': horizon_hours,
        }
        
        logger.debug(
            f"[{symbol}] Asset classification: {asset_type}, "
            f"target={target}%, stop={stop}% for {horizon_hours}h"
        )
        
        return result


# Singleton instance
_classifier = None

def get_asset_classifier() -> AssetClassifier:
    """Get the singleton AssetClassifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = AssetClassifier()
    return _classifier


def get_target_stop(symbol: str, horizon_hours: int = 48) -> Dict:
    """Convenience function to get target/stop for a symbol"""
    return AssetClassifier.get_target_stop(symbol, horizon_hours)


def get_asset_type(symbol: str) -> str:
    """Convenience function to get asset type"""
    return AssetClassifier.get_asset_type(symbol)


def is_crypto(symbol: str) -> bool:
    """Convenience function to check if crypto"""
    return AssetClassifier.is_crypto(symbol)


def is_stock(symbol: str) -> bool:
    """Convenience function to check if stock"""
    return AssetClassifier.is_stock(symbol)

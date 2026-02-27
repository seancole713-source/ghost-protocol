"""
Ghost Protocol - Asset Classifier
Classifies assets and returns appropriate target/stop percentages.

Created Dec 22, 2025 - Stock targets were 6-7% (too aggressive for large caps)
- AAPL moving 6% in 48h is rare (2-3 times/year)
- Need different targets for: crypto, large cap stocks, volatile stocks

Realistic Move Ranges (48 hours):
| Asset Type          | Daily Avg | 48h Expected | Target | Stop |
|---------------------|-----------|--------------|--------|------|
| Crypto Major (BTC)  | 3-5%      | 5-10%        | 6%     | 4.5% |
| Crypto Mid (UNI)    | 5-8%      | 8-15%        | 8%     | 6%   |
| Crypto Micro (ICP)  | 8-20%     | 12-25%       | 12%    | 8.5% |
| Stock (Large Cap)   | 0.5-1.5%  | 1-3%         | 2.5%   | 2%   |
| Stock (Mid Cap)     | 1-2%      | 2-4%         | 3.5%   | 2.5% |
| Stock (Volatile)    | 2-5%      | 4-8%         | 5%     | 4%   |

ICP FIX (Feb 27, 2026): ICP at $2.44 was getting 4.5% stop ($0.11 room).
Daily vol of ICP is 8-15%. Stop was eaten by noise in 30 minutes.
Split crypto into major/mid/micro tiers with appropriate stops.
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AssetClassifier:
    """
    Classifies assets and returns appropriate target/stop percentages.
    """
    
    # STABLECOINS - EXCLUDE from all predictions (pegged to $1, no movement)
    # These should NEVER appear in TOP 10 picks
    STABLECOINS = {
        'USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'FRAX',
        'LUSD', 'SUSD', 'USDD', 'USTC', 'MIM', 'CUSD', 'EURC', 'PAXG',
        'FDUSD', 'PYUSD', 'GHO', 'CRVUSD', 'DOLA', 'EURT', 'EURS',
        # Wrapped/Synthetic versions
        'WETH', 'WBTC', 'STETH', 'CBETH', 'RETH', 'WSTETH',
    }
    
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
    
    # Crypto assets — tiered by volatility/market cap
    # Major cryptos (daily vol ~3-5%, like BTC/ETH/SOL)
    CRYPTO_MAJOR = {
        'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'TON', 'ADA', 'DOT',
        'AVAX', 'LINK', 'MATIC', 'LTC', 'BCH',
    }

    # Mid-cap cryptos (daily vol ~5-8%)
    CRYPTO_MID = {
        'UNI', 'AAVE', 'ATOM', 'FIL', 'NEAR', 'APT', 'ARB', 'OP',
        'IMX', 'INJ', 'SUI', 'SEI', 'MKR', 'LDO', 'RNDR', 'FET',
        'GRT', 'STX', 'TIA', 'HBAR', 'VET', 'ALGO', 'FTM',
        'DYDX', 'ETC', 'XLM', 'TRX', 'RUNE', 'AR',
    }

    # Micro/meme cryptos (daily vol ~8-20%, very noisy)
    CRYPTO_MICRO = {
        # Low-cap altcoins
        'ICP', 'THETA', 'EGLD', 'QNT', 'QTUM', 'XTZ',
        'ROSE', 'CELO', 'ONE', 'FLOW', 'MINA', 'CFX', 'KAS',
        'KAVA', 'ZEN', 'ZEC', 'DASH', 'XMR', 'EOS',
        'CHZ', 'ENJ', 'SAND', 'MANA', 'AXS', 'GALA', 'ILV',
        'MAGIC', 'PRIME', 'BEAM', 'PIXEL', 'PORTAL', 'LRC',
        'ONDO', 'RLC', 'BAT', 'ZRX', 'ANT', 'LOOM', 'OMG',
        'STORJ', 'OCEAN', 'AGIX', 'APE', 'BLUR', 'ID', 'MASK',
        'ENS', 'LPT', 'SSV', 'ORDI', 'SATS', 'RATS', 'TRAC',
        'PYTH', 'JTO', 'JUP', 'W', 'STRK', 'ETHFI', 'ENA', 'PENDLE',
        'RSR', 'ANKR', 'API3', 'BAND', 'DIA', 'TRB', 'UMA',
        'SKL', 'CTSI', 'NMR', 'RAD', 'MLN', 'REN', 'KNC',
        'ZIL', 'ICX', 'ONT', 'NEO', 'WAVES', 'LSK', 'ARK',
        'METIS', 'BOBA', 'CELR', 'ACH', 'ALICE', 'TLM', 'SLP',
        'CLV', 'ORCA', 'SRM', 'HNT', 'IOTX', 'GLM', 'VOXEL',
        'SANTOS', 'PSG', 'BAR', 'JUV', 'CITY', 'ASR',
        # DeFi mid/small
        'SNX', 'CRV', 'COMP', 'YFI', 'SUSHI', '1INCH',
        'CAKE', 'JOE', 'GMX', 'BAL', 'CVX', 'FXS', 'RPL', 'LQTY', 'VELO',
        # Meme coins (widest stops — pure noise)
        'PEPE', 'SHIB', 'BONK', 'WIF', 'FLOKI', 'MEME',
        'TURBO', 'SAMO', 'ELON', 'LADYS', 'WOJAK', 'CHAD',
        'NEIRO', 'TOSHI', 'POPCAT', 'PNUT', 'MOODENG', 'GIGA',
        'SPX', 'BRETT', 'DOGE',
    }

    # Combined set for is_crypto checks (union of all tiers + common suffixes)
    CRYPTO = CRYPTO_MAJOR | CRYPTO_MID | CRYPTO_MICRO | {
        # Exchange tokens not in tiers above
        'LEO', 'OKB', 'CRO', 'FTT', 'HT', 'GT',
        # Common trading pairs suffixes
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    }
    
    # Target/stop percentages by asset type and horizon
    # Format: {asset_type: {horizon: (target_pct, stop_pct)}}
    TARGET_MATRIX = {
        'crypto_major': {
            6: (1.5, 1.5),    # BTC/ETH: tight stops OK
            24: (3.5, 3.0),
            48: (6.0, 4.5),
        },
        'crypto_mid': {
            6: (2.5, 2.5),    # Mid-caps: wider stops
            24: (5.0, 4.5),
            48: (8.0, 6.0),
        },
        'crypto_micro': {
            6: (4.0, 4.0),    # Micro/meme: WIDE stops — daily vol 8-20%
            24: (8.0, 7.0),   # ICP at $2.44 needs $0.17+ room, not $0.11
            48: (12.0, 8.5),  # A 4.5% stop gets eaten by noise
        },
        # Backward compat alias
        'crypto': {
            6: (1.5, 1.5),
            24: (3.5, 3.0),
            48: (6.0, 4.5),
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
    def is_stablecoin(cls, symbol: str) -> bool:
        """
        Check if symbol is a stablecoin (should be EXCLUDED from predictions).
        Stablecoins are pegged to $1 and don't move meaningfully.
        """
        clean = symbol.upper().replace('USDT', '').replace('USD', '').replace('/USD', '')
        return clean in cls.STABLECOINS or symbol.upper() in cls.STABLECOINS
    
    @classmethod
    def get_asset_type(cls, symbol: str) -> str:
        """
        Returns asset type: 'crypto_major', 'crypto_mid', 'crypto_micro',
        'stock_large', 'stock_volatile', 'stock_mid', 'stablecoin'
        """
        # Clean symbol (remove USDT, USD, etc.)
        clean = symbol.upper().replace('USDT', '').replace('USD', '').replace('/USD', '')
        
        # Check stablecoins FIRST (should be excluded from predictions)
        if clean in cls.STABLECOINS or symbol.upper() in cls.STABLECOINS:
            return 'stablecoin'  # Special type - should be filtered out
        
        # Tiered crypto classification
        if clean in cls.CRYPTO_MAJOR or symbol.upper() in cls.CRYPTO_MAJOR:
            return 'crypto_major'
        elif clean in cls.CRYPTO_MID or symbol.upper() in cls.CRYPTO_MID:
            return 'crypto_mid'
        elif clean in cls.CRYPTO_MICRO or symbol.upper() in cls.CRYPTO_MICRO:
            return 'crypto_micro'
        elif clean in cls.CRYPTO or symbol.upper() in cls.CRYPTO:
            return 'crypto_mid'  # Unknown crypto → default to mid
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
        return cls.get_asset_type(symbol).startswith('crypto')
    
    @classmethod
    def is_stock(cls, symbol: str) -> bool:
        """Check if symbol is a stock"""
        return not cls.get_asset_type(symbol).startswith('crypto')
    
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


def is_stablecoin(symbol: str) -> bool:
    """Convenience function to check if stablecoin (should be EXCLUDED from predictions)"""
    return AssetClassifier.is_stablecoin(symbol)

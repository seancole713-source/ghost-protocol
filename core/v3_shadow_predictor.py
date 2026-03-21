#!/usr/bin/env python3
"""
🔮 GHOST V3 - SHADOW PREDICTOR

Makes predictions for ALL assets (not just TOP 10) to build competition data.
Shadow predictions are tracked but NOT sent to Telegram.

This allows pending assets to "prove themselves" and compete for TOP 10.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

LOGGER = logging.getLogger("ghost.v3_shadow")


# Default asset pools to compete
DEFAULT_STOCKS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA", "NFLX", "CRM",
    # Semiconductors (REMOVED: PANW 5.4% accuracy - chronically wrong)
    "INTC", "MU", "LRCX", "AMAT", "QCOM", "AVGO", "TXN", "MRVL", "ON", "SNDK",
    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW", "AXP", "V",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "DHR", "BMY", "AMGN",
    # Consumer
    "WMT", "COST", "HD", "TGT", "LOW", "MCD", "SBUX", "NKE", "DIS", "CMCSA",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO", "HAL",
    # Gold/Mining
    "NEM", "GOLD", "GDX", "AEM", "FNV", "WPM", "KGC", "AGI", "AU", "HL",
    # Storage/Memory (your hot themes)
    "WDC", "STX", "NTAP", "PSTG",
    # Meme/Volatile
    "GME", "AMC", "BBBY", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "XPEV",
    # REITs
    "O", "AMT", "PLD", "CCI", "EQIX", "DLR", "PSA", "SPG", "VICI", "AVB"
]

DEFAULT_CRYPTO = [
    # Major
    "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "MATIC", "LTC",
    # Layer 2 / DeFi
    "AVAX", "LINK", "UNI", "AAVE", "MKR", "CRV", "COMP", "SNX", "LDO", "RPL",
    # Gaming/Metaverse
    "MANA", "SAND", "AXS", "GALA", "ENJ", "IMX", "ILV", "ALICE", "YGG", "MAGIC",
    # AI/Compute
    "RNDR", "FET", "OCEAN", "AGIX", "ROSE", "TAO", "NEAR", "ICP", "FIL", "AR",
    # Meme (REMOVED: SHIB 0% accuracy)
    "DOGE", "PEPE", "FLOKI", "BONK", "WIF", "TURBO", "BRETT",
    # Privacy
    "XMR", "ZEC", "DASH", "SCRT", "OASIS",
    # Exchange Tokens
    "BNB", "FTT", "CRO", "OKB", "KCS", "GT", "HT",
    # Your proven performers
    "CHZ", "EGLD", "RLC",
    # Emerging
    "SEI", "TIA", "SUI", "APT", "INJ", "STX", "ATOM", "OSMO", "RUNE", "KAVA"
]


class ShadowPredictor:
    """
    Makes shadow predictions for competition tracking.
    Uses existing prediction engines but doesn't send alerts.
    """
    
    def __init__(self):
        self.stocks = list(set(DEFAULT_STOCKS))  # Dedupe
        self.crypto = list(set(DEFAULT_CRYPTO))
        LOGGER.info(f"[SHADOW] Initialized with {len(self.stocks)} stocks, {len(self.crypto)} crypto")
    
    async def run_shadow_cycle(self) -> Dict:
        """
        Run shadow predictions for ALL assets in the pool.
        This builds competition data without sending alerts.
        
        Returns summary of predictions made.
        """
        from core.v3_competition import get_competition_system
        
        competition = get_competition_system()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stocks_predicted": 0,
            "crypto_predicted": 0,
            "errors": []
        }
        
        # Run stock predictions
        LOGGER.info(f"[SHADOW] 📊 Running shadow predictions for {len(self.stocks)} stocks...")
        for symbol in self.stocks:
            try:
                pred = await self._get_stock_prediction(symbol)
                if pred:
                    competition.record_shadow_prediction(
                        symbol=symbol,
                        asset_type="stock",
                        direction=pred["direction"],
                        entry_price=pred["entry_price"],
                        target_price=pred["target_price"],
                        confidence=pred["confidence"],
                        target_time=pred["target_time"]
                    )
                    results["stocks_predicted"] += 1
            except Exception as e:
                results["errors"].append(f"{symbol}: {str(e)[:50]}")
        
        # Run crypto predictions
        LOGGER.info(f"[SHADOW] 📊 Running shadow predictions for {len(self.crypto)} crypto...")
        for symbol in self.crypto:
            try:
                pred = await self._get_crypto_prediction(symbol)
                if pred:
                    competition.record_shadow_prediction(
                        symbol=symbol,
                        asset_type="crypto",
                        direction=pred["direction"],
                        entry_price=pred["entry_price"],
                        target_price=pred["target_price"],
                        confidence=pred["confidence"],
                        target_time=pred["target_time"]
                    )
                    results["crypto_predicted"] += 1
            except Exception as e:
                results["errors"].append(f"{symbol}: {str(e)[:50]}")
        
        LOGGER.info(f"[SHADOW] ✅ Shadow cycle complete: {results['stocks_predicted']} stocks, {results['crypto_predicted']} crypto")
        return results
    
    async def _get_stock_prediction(self, symbol: str) -> Optional[Dict]:
        """Get prediction for a stock using existing engine"""
        try:
            # Import here to avoid circular imports
            from core.prediction_engine import get_stock_prediction
            
            pred = await get_stock_prediction(symbol)
            if not pred or pred.get("error"):
                return None
            
            return {
                "direction": pred.get("direction", "BUY"),
                "entry_price": pred.get("entry_price", pred.get("current_price", 0)),
                "target_price": pred.get("target_price", 0),
                "confidence": pred.get("confidence", 0.5),
                "target_time": datetime.utcnow() + timedelta(hours=48)
            }
        except Exception as e:
            LOGGER.debug(f"[SHADOW] Stock prediction failed for {symbol}: {e}")
            return None
    
    async def _get_crypto_prediction(self, symbol: str) -> Optional[Dict]:
        """Get prediction for crypto using existing engine"""
        try:
            # Import here to avoid circular imports
            from core.prediction_engine import get_crypto_prediction
            
            pred = await get_crypto_prediction(symbol)
            if not pred or pred.get("error"):
                return None
            
            return {
                "direction": pred.get("direction", "BUY"),
                "entry_price": pred.get("entry_price", pred.get("current_price", 0)),
                "target_price": pred.get("target_price", 0),
                "confidence": pred.get("confidence", 0.5),
                "target_time": datetime.utcnow() + timedelta(hours=48)
            }
        except Exception as e:
            LOGGER.debug(f"[SHADOW] Crypto prediction failed for {symbol}: {e}")
            return None
    
    def add_stock(self, symbol: str):
        """Add a stock to the shadow pool"""
        symbol = symbol.upper()
        if symbol not in self.stocks:
            self.stocks.append(symbol)
            LOGGER.info(f"[SHADOW] Added stock: {symbol}")
    
    def add_crypto(self, symbol: str):
        """Add a crypto to the shadow pool"""
        symbol = symbol.upper()
        if symbol not in self.crypto:
            self.crypto.append(symbol)
            LOGGER.info(f"[SHADOW] Added crypto: {symbol}")
    
    def get_pool_status(self) -> Dict:
        """Get current shadow pool status"""
        return {
            "stocks": len(self.stocks),
            "crypto": len(self.crypto),
            "total": len(self.stocks) + len(self.crypto),
            "stock_list": sorted(self.stocks),
            "crypto_list": sorted(self.crypto)
        }


# Singleton
_shadow_predictor: Optional[ShadowPredictor] = None


def get_shadow_predictor() -> ShadowPredictor:
    """Get or create shadow predictor singleton"""
    global _shadow_predictor
    if _shadow_predictor is None:
        _shadow_predictor = ShadowPredictor()
    return _shadow_predictor


async def run_shadow_predictions() -> Dict:
    """Convenience function to run shadow prediction cycle"""
    predictor = get_shadow_predictor()
    return await predictor.run_shadow_cycle()

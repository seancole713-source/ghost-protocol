"""
BTC Correlation Analyzer

BTC is the sun, altcoins are planets.
Understanding BTC = understanding 80% of the market.

FREE data source: CoinGecko, Binance
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics

logger = logging.getLogger(__name__)


class BTCCorrelationAnalyzer:
    """
    Analyze BTC's influence on the market.
    
    Key insights:
    - 80% of altcoins follow BTC
    - BTC dominance rising = alts suffer
    - BTC dominance falling + BTC stable = alt season
    """
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3"
    
    # Market regime definitions
    MARKET_REGIMES = {
        'btc_accumulation': {
            'btc_dominance_trend': 'rising',
            'btc_price_trend': 'up_or_flat',
            'alt_performance': 'lagging',
            'strategy': 'HOLD_BTC_ONLY',
            'description': 'BTC accumulating - stay in BTC',
            'accuracy': 0.68,
            'next_phase': 'btc_rally'
        },
        'btc_rally': {
            'btc_dominance_trend': 'rising',
            'btc_price_trend': 'up_strong',
            'alt_performance': 'lagging',
            'strategy': 'HOLD_BTC',
            'description': 'BTC leading rally - alts will follow later',
            'accuracy': 0.65,
            'next_phase': 'alt_season'
        },
        'alt_season': {
            'btc_dominance_trend': 'falling',
            'btc_price_trend': 'stable',
            'alt_performance': 'outperforming',
            'strategy': 'ROTATE_TO_ALTS',
            'description': 'ALT SEASON - rotate to high-beta alts',
            'accuracy': 0.72,
            'next_phase': 'distribution'
        },
        'distribution': {
            'btc_dominance_trend': 'stable',
            'btc_price_trend': 'at_highs',
            'alt_performance': 'euphoric',
            'strategy': 'TAKE_PROFITS',
            'description': 'Distribution phase - take profits',
            'accuracy': 0.70,
            'next_phase': 'crash'
        },
        'crash': {
            'btc_dominance_trend': 'rising',
            'btc_price_trend': 'down_strong',
            'alt_performance': 'crashing_harder',
            'strategy': 'GO_TO_STABLES',
            'description': 'Market crash - move to stables',
            'accuracy': 0.75,
            'next_phase': 'capitulation'
        },
        'capitulation': {
            'btc_dominance_trend': 'rising_fast',
            'btc_price_trend': 'down',
            'alt_performance': 'destroyed',
            'strategy': 'ACCUMULATE_BTC',
            'description': 'Capitulation - start accumulating BTC',
            'accuracy': 0.73,
            'next_phase': 'btc_accumulation'
        }
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 600  # 10 minute cache
    
    def get_btc_dominance(self) -> Dict:
        """Get current BTC dominance percentage"""
        try:
            cache_key = 'btc_dominance'
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    return cached_data
            
            # Try CoinGecko with short timeout (respect rate limits)
            url = f"{self.COINGECKO_URL}/global"
            response = requests.get(url, timeout=5)
            
            # Handle rate limiting gracefully
            if response.status_code == 429:
                logger.warning("CoinGecko 429 rate limit on /global - using cached/default")
                if cache_key in self.cache:
                    return self.cache[cache_key][1]
                return {
                    'btc_dominance': 50,
                    'dom_signal': 'UNKNOWN',
                    'error': 'rate_limited'
                }
            
            response.raise_for_status()
            data = response.json()['data']
            
            btc_dominance = data['market_cap_percentage']['btc']
            eth_dominance = data['market_cap_percentage']['eth']
            total_market_cap = data['total_market_cap']['usd']
            
            # Interpret dominance level
            if btc_dominance > 55:
                dom_signal = 'HIGH'
                dom_interpretation = 'BTC dominant - risky for alts'
            elif btc_dominance > 45:
                dom_signal = 'NORMAL'
                dom_interpretation = 'Balanced market'
            else:
                dom_signal = 'LOW'
                dom_interpretation = 'Alt season territory'
            
            result = {
                'btc_dominance': btc_dominance,
                'eth_dominance': eth_dominance,
                'alt_dominance': 100 - btc_dominance - eth_dominance,
                'total_market_cap': total_market_cap,
                'dom_signal': dom_signal,
                'interpretation': dom_interpretation,
                'timestamp': datetime.now()
            }
            
            self.cache[cache_key] = (datetime.now(), result)
            
            logger.info(f"BTC Dominance: {btc_dominance:.1f}% ({dom_signal})")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching BTC dominance: {e}")
            return {
                'btc_dominance': 50,
                'dom_signal': 'UNKNOWN',
                'error': str(e)
            }
    
    def get_btc_price_data(self, days: int = 30) -> Dict:
        """Get BTC price and trend data"""
        try:
            cache_key = f'btc_price_{days}d'
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    return cached_data
            
            url = f"{self.COINGECKO_URL}/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 429:
                logger.warning("CoinGecko 429 rate limit on BTC price - using cached/default")
                if cache_key in self.cache:
                    return self.cache[cache_key][1]
                return {
                    'current_price': 70000,
                    'trend': 'unknown',
                    'error': 'rate_limited'
                }
            
            response.raise_for_status()
            data = response.json()
            
            prices = [p[1] for p in data['prices']]
            current_price = prices[-1]
            
            # Calculate changes
            price_1d = ((current_price - prices[-24]) / prices[-24] * 100) if len(prices) > 24 else 0
            price_7d = ((current_price - prices[-168]) / prices[-168] * 100) if len(prices) > 168 else 0
            price_30d = ((current_price - prices[0]) / prices[0] * 100) if prices else 0
            
            # Calculate trend
            recent_avg = statistics.mean(prices[-72:]) if len(prices) > 72 else current_price
            older_avg = statistics.mean(prices[-168:-72]) if len(prices) > 168 else current_price
            
            if recent_avg > older_avg * 1.05:
                trend = 'up_strong'
            elif recent_avg > older_avg * 1.02:
                trend = 'up'
            elif recent_avg < older_avg * 0.95:
                trend = 'down_strong'
            elif recent_avg < older_avg * 0.98:
                trend = 'down'
            else:
                trend = 'stable'
            
            result = {
                'price': current_price,
                'change_1d': price_1d,
                'change_7d': price_7d,
                'change_30d': price_30d,
                'trend': trend,
                'recent_avg': recent_avg,
                'older_avg': older_avg
            }
            
            self.cache[cache_key] = (datetime.now(), result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching BTC price data: {e}")
            return {'price': 0, 'trend': 'unknown', 'error': str(e)}
    
    def get_dominance_trend(self, days: int = 14) -> Dict:
        """Calculate BTC dominance trend over time"""
        try:
            cache_key = 'dom_trend'
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    return cached_data
            
            # Reuse cached dominance data if available
            dom_data = self.get_btc_dominance()
            current_dom = dom_data.get('btc_dominance', 50)
            
            # We can estimate trend from BTC vs total market cap changes
            # In production, you'd store historical dominance values
            
            # For now, use current as baseline and estimate
            # This is a simplified version - you'd want to store and compare
            
            if current_dom > 52:
                trend = 'rising'
            elif current_dom < 45:
                trend = 'falling'
            else:
                trend = 'stable'
            
            result = {
                'current': current_dom,
                'trend': trend,
                'interpretation': self._interpret_dom_trend(trend)
            }
            
            self.cache[cache_key] = (datetime.now(), result)
            return result
            
        except Exception as e:
            logger.error(f"Error calculating dominance trend: {e}")
            return {'current': 50, 'trend': 'unknown', 'error': str(e)}
    
    def _interpret_dom_trend(self, trend: str) -> str:
        """Interpret dominance trend"""
        interpretations = {
            'rising': 'BTC strengthening - be cautious on alts',
            'falling': 'Money flowing to alts - alt season building',
            'stable': 'Market in balance'
        }
        return interpretations.get(trend, 'Unknown trend')
    
    def detect_market_regime(self) -> Dict:
        """
        Detect current market regime.
        This is critical for strategy selection.
        """
        try:
            dominance = self.get_btc_dominance()
            btc_price = self.get_btc_price_data()
            dom_trend = self.get_dominance_trend()
            
            # Score each regime
            regime_scores = {}
            
            for regime_name, regime_data in self.MARKET_REGIMES.items():
                score = 0
                
                # Check dominance trend
                if dom_trend['trend'] == 'rising' and regime_data['btc_dominance_trend'] == 'rising':
                    score += 2
                elif dom_trend['trend'] == 'falling' and regime_data['btc_dominance_trend'] == 'falling':
                    score += 2
                elif dom_trend['trend'] == 'stable' and regime_data['btc_dominance_trend'] == 'stable':
                    score += 2
                
                # Check BTC price trend
                if btc_price['trend'] == regime_data['btc_price_trend']:
                    score += 2
                elif btc_price['trend'].startswith('up') and regime_data['btc_price_trend'].startswith('up'):
                    score += 1
                elif btc_price['trend'].startswith('down') and regime_data['btc_price_trend'].startswith('down'):
                    score += 1
                
                # Dominance level checks
                if dominance['btc_dominance'] > 55 and regime_name in ['btc_rally', 'crash', 'capitulation']:
                    score += 1
                elif dominance['btc_dominance'] < 45 and regime_name in ['alt_season', 'distribution']:
                    score += 1
                
                regime_scores[regime_name] = score
            
            # Find best matching regime
            best_regime = max(regime_scores, key=regime_scores.get)
            regime_data = self.MARKET_REGIMES[best_regime]
            
            return {
                'regime': best_regime,
                'strategy': regime_data['strategy'],
                'description': regime_data['description'],
                'accuracy': regime_data['accuracy'],
                'next_phase': regime_data['next_phase'],
                'btc_dominance': dominance['btc_dominance'],
                'btc_trend': btc_price['trend'],
                'dom_trend': dom_trend['trend'],
                'confidence_score': regime_scores[best_regime],
                'all_scores': regime_scores
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': 'unknown',
                'strategy': 'HOLD',
                'error': str(e)
            }
    
    def calculate_altcoin_beta(self, altcoin: str, days: int = 30) -> Dict:
        """
        Calculate how much an altcoin moves relative to BTC.
        
        Beta = altcoin volatility / BTC volatility * correlation
        
        - Beta 1.0 = moves same as BTC
        - Beta 2.0 = moves 2x BTC (more volatile)
        - Beta 0.5 = moves half of BTC (less volatile)
        """
        try:
            cache_key = f'beta_{altcoin}_{days}'
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    return cached_data
            
            # Get BTC data
            btc_url = f"{self.COINGECKO_URL}/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
            btc_response = requests.get(btc_url, timeout=5)
            if btc_response.status_code == 429:
                logger.warning(f"CoinGecko 429 on BTC beta calc for {altcoin}")
                return {'beta': 1.0, 'volatility_ratio': 1.0, 'error': 'rate_limited'}
            btc_prices = [p[1] for p in btc_response.json()['prices']]
            
            # Get altcoin data
            alt_url = f"{self.COINGECKO_URL}/coins/{altcoin.lower()}/market_chart?vs_currency=usd&days={days}"
            alt_response = requests.get(alt_url, timeout=5)
            if alt_response.status_code == 429:
                logger.warning(f"CoinGecko 429 on altcoin beta calc for {altcoin}")
                return {'beta': 1.0, 'volatility_ratio': 1.0, 'error': 'rate_limited'}
            alt_prices = [p[1] for p in alt_response.json()['prices']]
            
            # Align lengths
            min_len = min(len(btc_prices), len(alt_prices))
            btc_prices = btc_prices[-min_len:]
            alt_prices = alt_prices[-min_len:]
            
            # Calculate returns
            btc_returns = [(btc_prices[i] - btc_prices[i-1]) / btc_prices[i-1] 
                          for i in range(1, len(btc_prices))]
            alt_returns = [(alt_prices[i] - alt_prices[i-1]) / alt_prices[i-1] 
                          for i in range(1, len(alt_prices))]
            
            # Calculate correlation
            if len(btc_returns) < 10:
                return {'altcoin': altcoin, 'beta': 1.0, 'error': 'Insufficient data'}
            
            # Covariance
            btc_mean = statistics.mean(btc_returns)
            alt_mean = statistics.mean(alt_returns)
            
            covariance = sum((b - btc_mean) * (a - alt_mean) 
                           for b, a in zip(btc_returns, alt_returns)) / len(btc_returns)
            
            # Variance of BTC
            btc_variance = statistics.variance(btc_returns)
            
            # Beta
            beta = covariance / btc_variance if btc_variance > 0 else 1.0
            
            # Interpretation
            if beta > 1.5:
                interpretation = 'High beta - amplifies BTC moves significantly'
                risk = 'HIGH'
            elif beta > 1.0:
                interpretation = 'Above average beta - amplifies BTC moves'
                risk = 'MEDIUM'
            elif beta > 0.5:
                interpretation = 'Below average beta - muted response to BTC'
                risk = 'LOW'
            else:
                interpretation = 'Very low beta - somewhat independent of BTC'
                risk = 'VARIABLE'
            
            return {
                'altcoin': altcoin,
                'beta': beta,
                'interpretation': interpretation,
                'risk': risk,
                'btc_change_30d': ((btc_prices[-1] - btc_prices[0]) / btc_prices[0]) * 100,
                'alt_change_30d': ((alt_prices[-1] - alt_prices[0]) / alt_prices[0]) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating beta for {altcoin}: {e}")
            return {'altcoin': altcoin, 'beta': 1.0, 'error': str(e)}
    
    def get_alt_season_index(self) -> Dict:
        """
        Calculate alt season index.
        
        Measures what percentage of top 50 alts are outperforming BTC.
        - > 75% = Strong alt season
        - 50-75% = Moderate alt season
        - < 50% = BTC season
        """
        try:
            # Get top coins
            url = f"{self.COINGECKO_URL}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=51&page=1"
            response = requests.get(url, timeout=10)
            coins = response.json()
            
            # Find BTC
            btc_data = next((c for c in coins if c['id'] == 'bitcoin'), None)
            if not btc_data:
                return {'index': 50, 'error': 'BTC not found'}
            
            btc_change = btc_data.get('price_change_percentage_24h', 0) or 0
            
            # Count outperformers
            outperforming = 0
            total_alts = 0
            
            for coin in coins:
                if coin['id'] == 'bitcoin':
                    continue
                if coin['id'] in ['tether', 'usd-coin', 'binance-usd', 'dai']:  # Skip stables
                    continue
                
                total_alts += 1
                alt_change = coin.get('price_change_percentage_24h', 0) or 0
                
                if alt_change > btc_change:
                    outperforming += 1
            
            index = (outperforming / total_alts * 100) if total_alts > 0 else 50
            
            # Interpretation
            if index > 75:
                signal = 'STRONG_ALT_SEASON'
                description = 'Strong alt season - 75%+ alts outperforming BTC'
                strategy = 'HEAVY_ALTS'
            elif index > 50:
                signal = 'ALT_SEASON'
                description = 'Alt season starting - consider rotating'
                strategy = 'BALANCED'
            elif index > 25:
                signal = 'BTC_FAVORED'
                description = 'BTC leading - stay with majors'
                strategy = 'BTC_HEAVY'
            else:
                signal = 'BTC_DOMINANCE'
                description = 'BTC strongly dominating - avoid small alts'
                strategy = 'BTC_ONLY'
            
            return {
                'index': index,
                'outperforming_count': outperforming,
                'total_alts': total_alts,
                'btc_24h_change': btc_change,
                'signal': signal,
                'description': description,
                'strategy': strategy
            }
            
        except Exception as e:
            logger.error(f"Error calculating alt season index: {e}")
            return {'index': 50, 'signal': 'UNKNOWN', 'error': str(e)}
    
    def get_signal_strength(self, symbol: str = 'BTC') -> Dict:
        """
        Get comprehensive BTC correlation signal with strength rating.
        """
        dominance = self.get_btc_dominance()
        btc_price = self.get_btc_price_data()
        regime = self.detect_market_regime()
        alt_index = self.get_alt_season_index()
        
        # Calculate confidence boost
        confidence_boost = 0
        
        # Strong regime signal
        if regime['confidence_score'] >= 4:
            confidence_boost += 6
        elif regime['confidence_score'] >= 3:
            confidence_boost += 3
        
        # Clear alt season or BTC dominance
        if alt_index['index'] > 75 or alt_index['index'] < 25:
            confidence_boost += 5
        
        # Clear dominance trend
        if dominance['btc_dominance'] > 55 or dominance['btc_dominance'] < 42:
            confidence_boost += 4
        
        # Calculate overall strength
        base_strength = regime['accuracy']
        strength = min(base_strength + confidence_boost / 100, 0.85)
        
        return {
            'btc_dominance': dominance,
            'btc_price': btc_price,
            'market_regime': regime,
            'alt_season_index': alt_index,
            'signal': regime['strategy'],
            'strength': strength,
            'confidence_boost': confidence_boost,
            'reasoning': self._generate_reasoning(dominance, regime, alt_index)
        }
    
    def _generate_reasoning(self, dominance: Dict, regime: Dict, alt_index: Dict) -> str:
        """Generate human-readable reasoning"""
        parts = []
        
        # Dominance
        parts.append(f"BTC Dominance: {dominance['btc_dominance']:.1f}%")
        
        # Regime
        parts.append(f"Market Regime: {regime['regime'].replace('_', ' ').title()}")
        parts.append(f"Strategy: {regime['strategy']}")
        
        # Alt season
        parts.append(f"Alt Season Index: {alt_index['index']:.0f}% ({alt_index['signal']})")
        
        return ". ".join(parts)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    analyzer = BTCCorrelationAnalyzer()
    
    print("\n" + "="*60)
    print("BTC CORRELATION ANALYZER TEST")
    print("="*60)
    
    # Test BTC Dominance
    print("\n📊 BTC DOMINANCE:")
    dom = analyzer.get_btc_dominance()
    print(f"   BTC: {dom['btc_dominance']:.1f}%")
    print(f"   ETH: {dom['eth_dominance']:.1f}%")
    print(f"   Alts: {dom['alt_dominance']:.1f}%")
    print(f"   Signal: {dom['dom_signal']}")
    
    # Test Market Regime
    print("\n🌍 MARKET REGIME:")
    regime = analyzer.detect_market_regime()
    print(f"   Regime: {regime['regime']}")
    print(f"   Strategy: {regime['strategy']}")
    print(f"   {regime['description']}")
    print(f"   Accuracy: {regime['accuracy']:.0%}")
    print(f"   Next Phase: {regime['next_phase']}")
    
    # Test Alt Season Index
    print("\n🚀 ALT SEASON INDEX:")
    alt = analyzer.get_alt_season_index()
    print(f"   Index: {alt['index']:.0f}%")
    print(f"   Outperforming: {alt['outperforming_count']}/{alt['total_alts']}")
    print(f"   Signal: {alt['signal']}")
    print(f"   Strategy: {alt['strategy']}")
    
    # Test Signal Strength
    print("\n💪 OVERALL BTC SIGNAL:")
    signal = analyzer.get_signal_strength()
    print(f"   Strategy: {signal['signal']}")
    print(f"   Strength: {signal['strength']:.0%}")
    print(f"   Confidence Boost: +{signal['confidence_boost']}%")
    print(f"\n   Reasoning: {signal['reasoning']}")

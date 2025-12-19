"""
Signal Aggregator - Combines All Signal Sources

This is the unified interface to the pattern intelligence system.
It provides a single entry point to get comprehensive market analysis.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

from .pattern_matcher import PatternMatcher
from .fear_greed import FearGreedAnalyzer
from .funding_rates import FundingRateAnalyzer
from .social_sentiment import SocialSentimentAnalyzer
from .btc_correlation import BTCCorrelationAnalyzer

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Unified interface to all signal sources.
    
    Provides:
    - Individual signal analysis
    - Combined pattern prediction
    - Confidence scoring
    - Trade recommendations
    """
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.fear_greed = FearGreedAnalyzer()
        self.funding = FundingRateAnalyzer()
        self.social = SocialSentimentAnalyzer()
        self.btc = BTCCorrelationAnalyzer()
    
    def get_market_pulse(self) -> Dict:
        """
        Quick market overview - the "pulse" of the market.
        Fast, lightweight check of key indicators.
        """
        pulse = {
            'timestamp': datetime.now(),
            'indicators': {}
        }
        
        # Fear & Greed (most important)
        try:
            fg = self.fear_greed.get_current()
            pulse['indicators']['fear_greed'] = {
                'value': fg['value'],
                'zone': fg['zone'],
                'signal': fg['signal']
            }
        except:
            pulse['indicators']['fear_greed'] = {'value': 50, 'signal': 'HOLD'}
        
        # BTC dominance
        try:
            dom = self.btc.get_btc_dominance()
            pulse['indicators']['btc_dominance'] = {
                'value': dom['btc_dominance'],
                'signal': dom['dom_signal']
            }
        except:
            pulse['indicators']['btc_dominance'] = {'value': 50, 'signal': 'NORMAL'}
        
        # Quick overall assessment
        fg_val = pulse['indicators']['fear_greed']['value']
        
        if fg_val < 25:
            pulse['market_state'] = 'EXTREME_FEAR'
            pulse['quick_take'] = 'Historically good buying opportunity'
        elif fg_val > 75:
            pulse['market_state'] = 'EXTREME_GREED'
            pulse['quick_take'] = 'Historically good time to take profits'
        elif fg_val < 40:
            pulse['market_state'] = 'FEAR'
            pulse['quick_take'] = 'Cautious optimism'
        elif fg_val > 60:
            pulse['market_state'] = 'GREED'
            pulse['quick_take'] = 'Be careful with new positions'
        else:
            pulse['market_state'] = 'NEUTRAL'
            pulse['quick_take'] = 'Wait for clearer signals'
        
        return pulse
    
    def get_full_analysis(self, symbol: str = 'BTC') -> Dict:
        """
        Complete analysis for a symbol.
        Uses all signal sources and pattern matching.
        """
        logger.info(f"Running full analysis for {symbol}")
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signals': {},
            'prediction': None,
            'risk_assessment': None,
            'trade_recommendation': None
        }
        
        # Get prediction from pattern matcher
        try:
            prediction = self.pattern_matcher.generate_prediction(symbol)
            analysis['prediction'] = prediction
            analysis['signals'] = prediction.get('signals', {})
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            analysis['prediction'] = {
                'direction': 'HOLD',
                'confidence': 0.5,
                'conviction': 'INSUFFICIENT',
                'error': str(e)
            }
        
        # Risk assessment
        analysis['risk_assessment'] = self._assess_risk(analysis)
        
        # Trade recommendation
        analysis['trade_recommendation'] = self._generate_trade_recommendation(analysis)
        
        return analysis
    
    def _assess_risk(self, analysis: Dict) -> Dict:
        """Assess current risk level based on signals"""
        risk_factors = []
        risk_score = 50  # Base risk score (0-100, higher = more risk)
        
        signals = analysis.get('signals', {})
        
        # Fear & Greed extremes = high volatility risk
        fg = signals.get('fear_greed', {})
        if fg.get('zone') in ['extreme_fear', 'extreme_greed']:
            risk_factors.append('Extreme sentiment - expect high volatility')
            risk_score += 15
        
        # High funding = liquidation risk
        funding = signals.get('funding', {})
        if funding.get('zone') in ['very_high_positive', 'very_high_negative']:
            risk_factors.append('Extreme funding - liquidation cascades possible')
            risk_score += 20
        
        # BTC regime
        btc = signals.get('btc', {})
        if btc.get('market_regime') in ['crash', 'capitulation']:
            risk_factors.append(f"Market in {btc.get('market_regime')} regime")
            risk_score += 25
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = 'EXTREME'
        elif risk_score >= 60:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }
    
    def _generate_trade_recommendation(self, analysis: Dict) -> Dict:
        """Generate actionable trade recommendation"""
        prediction = analysis.get('prediction', {})
        risk = analysis.get('risk_assessment', {})
        
        direction = prediction.get('direction', 'HOLD')
        confidence = prediction.get('confidence', 0.5)
        conviction = prediction.get('conviction', 'INSUFFICIENT')
        risk_level = risk.get('risk_level', 'MODERATE')
        
        # Determine position size based on conviction and risk
        if conviction == 'HIGH' and risk_level in ['LOW', 'MODERATE']:
            position_size = '15% of portfolio'
            action = 'STRONG_' + ('BUY' if direction == 'UP' else 'SELL' if direction == 'DOWN' else 'HOLD')
        elif conviction == 'MEDIUM' and risk_level != 'EXTREME':
            position_size = '8% of portfolio'
            action = 'BUY' if direction == 'UP' else 'SELL' if direction == 'DOWN' else 'HOLD'
        elif conviction == 'LOW':
            position_size = '3% of portfolio'
            action = 'SMALL_' + ('BUY' if direction == 'UP' else 'SELL' if direction == 'DOWN' else 'HOLD')
        else:
            position_size = '0% - wait for better setup'
            action = 'WAIT'
        
        # Risk warning
        risk_warning = None
        if risk_level in ['HIGH', 'EXTREME']:
            risk_warning = f"⚠️ {risk_level} RISK - Consider smaller position or waiting"
        
        return {
            'action': action,
            'direction': direction,
            'position_size': position_size,
            'confidence': confidence,
            'conviction': conviction,
            'risk_level': risk_level,
            'risk_warning': risk_warning,
            'stop_loss': '5% from entry' if direction != 'HOLD' else None,
            'take_profit': f"+{prediction.get('expected_return_7d', 8):.0f}% (7 day target)" if direction != 'HOLD' else None
        }
    
    def get_top_opportunities(self, symbols: List[str] = None, min_confidence: float = 0.55) -> List[Dict]:
        """
        Scan multiple symbols and return top opportunities.
        """
        if symbols is None:
            symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK']
        
        opportunities = []
        
        for symbol in symbols:
            try:
                analysis = self.get_full_analysis(symbol)
                pred = analysis.get('prediction', {})
                
                if pred.get('confidence', 0) >= min_confidence and pred.get('direction') != 'HOLD':
                    opportunities.append({
                        'symbol': symbol,
                        'direction': pred.get('direction'),
                        'confidence': pred.get('confidence'),
                        'conviction': pred.get('conviction'),
                        'pattern': pred.get('pattern_name'),
                        'expected_return': pred.get('expected_return_7d'),
                        'reasoning': pred.get('reasoning'),
                        'risk_level': analysis.get('risk_assessment', {}).get('risk_level'),
                        'recommendation': analysis.get('trade_recommendation')
                    })
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    def format_daily_report(self, symbols: List[str] = None) -> str:
        """
        Generate daily market report.
        """
        # Market pulse
        pulse = self.get_market_pulse()
        
        # Top opportunities
        opportunities = self.get_top_opportunities(symbols)
        
        # Format report
        report = []
        report.append("🔮 GHOST ORACLE DAILY REPORT")
        report.append(f"📅 {datetime.now().strftime('%B %d, %Y %H:%M UTC')}")
        report.append("")
        report.append("=" * 40)
        report.append("📊 MARKET PULSE")
        report.append("=" * 40)
        report.append(f"State: {pulse['market_state']}")
        report.append(f"Fear & Greed: {pulse['indicators']['fear_greed']['value']}")
        report.append(f"BTC Dominance: {pulse['indicators']['btc_dominance']['value']:.1f}%")
        report.append(f"💡 {pulse['quick_take']}")
        report.append("")
        
        if opportunities:
            report.append("=" * 40)
            report.append("🎯 TOP OPPORTUNITIES")
            report.append("=" * 40)
            
            for i, opp in enumerate(opportunities[:5], 1):
                emoji = "🚀" if opp['direction'] == 'UP' else "📉"
                report.append(f"\n{i}. {opp['symbol']} {emoji}")
                report.append(f"   Direction: {opp['direction']}")
                report.append(f"   Confidence: {opp['confidence']:.0%}")
                report.append(f"   Expected: {opp['expected_return']:+.1f}% (7d)")
                if opp.get('pattern'):
                    report.append(f"   Pattern: {opp['pattern']}")
                report.append(f"   Risk: {opp['risk_level']}")
        else:
            report.append("\n⏳ No high-confidence opportunities found")
            report.append("Wait for better setups")
        
        report.append("")
        report.append("=" * 40)
        report.append("Powered by Pattern Intelligence System")
        report.append("Signal Sources: Fear/Greed + Funding + Social + BTC")
        
        return "\n".join(report)


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    aggregator = SignalAggregator()
    
    print("\n" + "="*60)
    print("🔮 SIGNAL AGGREGATOR TEST")
    print("="*60)
    
    # Quick market pulse
    print("\n📊 MARKET PULSE:")
    pulse = aggregator.get_market_pulse()
    print(f"   State: {pulse['market_state']}")
    print(f"   Quick Take: {pulse['quick_take']}")
    
    # Full analysis for BTC
    print("\n" + "="*60)
    print("FULL BTC ANALYSIS")
    print("="*60)
    
    analysis = aggregator.get_full_analysis('BTC')
    
    pred = analysis['prediction']
    print(f"\n📈 Prediction: {pred['direction']}")
    print(f"   Confidence: {pred['confidence']:.0%}")
    print(f"   Conviction: {pred['conviction']}")
    
    risk = analysis['risk_assessment']
    print(f"\n⚠️ Risk: {risk['risk_level']} (score: {risk['risk_score']})")
    for factor in risk['risk_factors']:
        print(f"   • {factor}")
    
    rec = analysis['trade_recommendation']
    print(f"\n💰 Recommendation: {rec['action']}")
    print(f"   Position: {rec['position_size']}")
    if rec['stop_loss']:
        print(f"   Stop Loss: {rec['stop_loss']}")
    if rec['take_profit']:
        print(f"   Take Profit: {rec['take_profit']}")
    
    # Daily report
    print("\n" + "="*60)
    print("DAILY REPORT")
    print("="*60)
    print(aggregator.format_daily_report(['BTC', 'ETH', 'SOL']))

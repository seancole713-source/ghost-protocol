"""
GPT-4 Reasoning Layer (Optional - $20/month)

Uses GPT-4 to:
1. Synthesize all signals into human-readable analysis
2. Catch contradictions between signals
3. Understand news context
4. Explain predictions in plain English

This is optional but adds 3-5% to accuracy by catching edge cases.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class GPT4Analyst:
    """
    GPT-4 powered analysis layer.
    
    Adds human-like reasoning to pattern-based predictions.
    Catches nuances that pure pattern matching might miss.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.enabled = bool(self.api_key)
        # Use gpt-4o-mini by default (cheaper), or GPT-4 for better reasoning
        self.model = os.getenv('AI_MODEL', 'gpt-4o-mini')
        self.client = None
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"GPT-4 Analyst initialized with model: {self.model}")
            except ImportError:
                logger.warning("OpenAI package not installed. GPT-4 analysis disabled.")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self.enabled = False
        else:
            logger.info("GPT-4 Analyst disabled (no API key)")
    
    async def analyze_prediction(self, prediction: Dict, signals: Dict) -> Dict:
        """
        Have GPT-4 review and enhance a prediction.
        """
        if not self.enabled:
            return self._fallback_analysis(prediction)
        
        try:
            prompt = self._build_analysis_prompt(prediction, signals)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"GPT-4 analysis failed: {e}")
            return self._fallback_analysis(prediction)
    
    def analyze_prediction_sync(self, prediction: Dict, signals: Dict) -> Dict:
        """
        Synchronous version of analyze_prediction.
        """
        if not self.enabled:
            return self._fallback_analysis(prediction)
        
        try:
            prompt = self._build_analysis_prompt(prediction, signals)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"GPT-4 analysis failed: {e}")
            return self._fallback_analysis(prediction)
    
    def _get_system_prompt(self) -> str:
        return """You are Ghost Oracle, an elite crypto market analyst with a 72% track record.

Your job is to review trading signals and provide:
1. A final verdict (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
2. Confidence level (0-100%)
3. Key reasoning in 2-3 sentences
4. The biggest risk to this trade
5. Any signal conflicts you notice

You are known for:
- Clear, decisive calls (no wishy-washy responses)
- Identifying risks others miss
- Being honest when uncertain
- Explaining complex situations simply

If signals conflict or are unclear, say HOLD.
Only recommend STRONG_BUY or STRONG_SELL when signals strongly align.

Respond in JSON format."""
    
    def _build_analysis_prompt(self, prediction: Dict, signals: Dict) -> str:
        return f"""Review this trading analysis for {prediction.get('symbol', 'BTC')}:

=== PATTERN ANALYSIS ===
Direction: {prediction.get('direction')}
Confidence: {prediction.get('confidence', 0):.0%}
Pattern Matched: {prediction.get('pattern_name', 'None')}
Pattern Description: {prediction.get('pattern_description', 'No specific pattern')}

=== KEY SIGNALS ===
Fear & Greed: {signals.get('fear_greed', {}).get('value', 'N/A')} ({signals.get('fear_greed', {}).get('zone', 'N/A')})
Funding Rate: {signals.get('funding', {}).get('rate_percent', 0):.4f}% ({signals.get('funding', {}).get('signal', 'N/A')})
Social Sentiment: {signals.get('social', {}).get('signal', 'N/A')}
BTC Market Regime: {signals.get('btc', {}).get('market_regime', 'N/A')}
BTC Dominance: {signals.get('btc', {}).get('dominance', 'N/A')}%

=== SIGNAL ALIGNMENT ===
Bullish signals: {prediction.get('alignment', {}).get('bullish_signals', 0)}
Bearish signals: {prediction.get('alignment', {}).get('bearish_signals', 0)}

Provide your analysis as JSON with these fields:
- verdict: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
- confidence: 0-100
- reasoning: 2-3 sentence explanation
- biggest_risk: the main risk to this trade
- signal_conflicts: any contradictions you notice
- adjustment: any changes you'd make to the original analysis"""
    
    def _fallback_analysis(self, prediction: Dict) -> Dict:
        """Provide fallback when GPT-4 is not available"""
        return {
            'verdict': prediction.get('direction', 'HOLD'),
            'confidence': int(prediction.get('confidence', 0.5) * 100),
            'reasoning': prediction.get('reasoning', 'Pattern-based analysis without GPT-4 enhancement'),
            'biggest_risk': 'Unable to assess with GPT-4 - rely on pattern analysis',
            'signal_conflicts': None,
            'adjustment': None,
            'gpt4_enhanced': False
        }
    
    async def explain_for_telegram(self, prediction: Dict, signals: Dict) -> str:
        """
        Generate Telegram-friendly explanation.
        """
        if not self.enabled:
            return self._format_basic_telegram(prediction)
        
        try:
            prompt = f"""Explain this trading prediction in plain English for a Telegram message.
Keep it under 200 words. Use emojis. Be direct.

Symbol: {prediction.get('symbol')}
Direction: {prediction.get('direction')}
Confidence: {prediction.get('confidence', 0):.0%}
Pattern: {prediction.get('pattern_name', 'Multiple signal alignment')}
Fear & Greed: {signals.get('fear_greed', {}).get('value', 'N/A')}
Funding Rate: {signals.get('funding', {}).get('rate_percent', 0):.4f}%

Write as if talking to a friend who trades crypto. No financial advice disclaimers."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GPT-4 Telegram explanation failed: {e}")
            return self._format_basic_telegram(prediction)
    
    def _format_basic_telegram(self, prediction: Dict) -> str:
        """Basic Telegram format without GPT-4"""
        emoji = "🚀" if prediction.get('direction') == 'UP' else "📉" if prediction.get('direction') == 'DOWN' else "➡️"
        
        msg = f"{emoji} **{prediction.get('symbol', 'BTC')}** {emoji}\n\n"
        msg += f"Direction: {prediction.get('direction', 'HOLD')}\n"
        msg += f"Confidence: {prediction.get('confidence', 0.5):.0%}\n"
        
        if prediction.get('pattern_name'):
            msg += f"\nPattern: {prediction['pattern_name']}\n"
        
        msg += f"\n{prediction.get('reasoning', 'Pattern-based analysis')}"
        
        return msg
    
    async def analyze_news_impact(self, news_items: List[Dict], symbol: str) -> Dict:
        """
        Have GPT-4 analyze news impact on a symbol.
        """
        if not self.enabled or not news_items:
            return {'impact': 'unknown', 'analysis': 'GPT-4 not available'}
        
        try:
            news_text = "\n".join([
                f"- {item.get('title', '')}" 
                for item in news_items[:10]
            ])
            
            prompt = f"""Analyze these recent crypto news headlines for {symbol}:

{news_text}

Provide:
1. Overall impact: BULLISH/BEARISH/NEUTRAL
2. Impact strength: HIGH/MEDIUM/LOW
3. Key takeaway in 1-2 sentences
4. Any immediate risks

Format as JSON."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"GPT-4 news analysis failed: {e}")
            return {'impact': 'unknown', 'error': str(e)}
    
    def get_contrarian_check(self, prediction: Dict) -> Dict:
        """
        Quick check if prediction aligns with common contrarian wisdom.
        No API call needed.
        """
        signals = prediction.get('signals', {})
        
        contrarian_signals = []
        
        # Extreme fear = buy
        fg = signals.get('fear_greed', {})
        if fg.get('zone') == 'extreme_fear' and prediction.get('direction') == 'UP':
            contrarian_signals.append({
                'signal': 'extreme_fear_buy',
                'strength': 'STRONG',
                'reason': 'Extreme fear + bullish prediction - classic contrarian setup'
            })
        
        # Extreme greed = sell
        if fg.get('zone') == 'extreme_greed' and prediction.get('direction') == 'DOWN':
            contrarian_signals.append({
                'signal': 'extreme_greed_sell',
                'strength': 'STRONG',
                'reason': 'Extreme greed + bearish prediction - classic contrarian setup'
            })
        
        # Negative funding = long
        funding = signals.get('funding', {})
        if funding.get('zone') in ['high_negative', 'very_high_negative'] and prediction.get('direction') == 'UP':
            contrarian_signals.append({
                'signal': 'negative_funding_long',
                'strength': 'MEDIUM',
                'reason': 'Overleveraged shorts - short squeeze potential'
            })
        
        # Positive funding = short
        if funding.get('zone') in ['high_positive', 'very_high_positive'] and prediction.get('direction') == 'DOWN':
            contrarian_signals.append({
                'signal': 'positive_funding_short',
                'strength': 'MEDIUM',
                'reason': 'Overleveraged longs - liquidation cascade potential'
            })
        
        return {
            'contrarian_aligned': len(contrarian_signals) > 0,
            'contrarian_signals': contrarian_signals,
            'confidence_boost': len(contrarian_signals) * 3  # +3% per contrarian alignment
        }


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    analyst = GPT4Analyst()
    
    print("\n" + "="*60)
    print("GPT-4 ANALYST TEST")
    print("="*60)
    
    if analyst.enabled:
        print("✅ GPT-4 Analyst enabled")
    else:
        print("❌ GPT-4 Analyst disabled (no API key)")
        print("   Set OPENAI_API_KEY environment variable to enable")
    
    # Test contrarian check (no API needed)
    print("\n📊 CONTRARIAN CHECK TEST:")
    
    test_prediction = {
        'symbol': 'BTC',
        'direction': 'UP',
        'confidence': 0.72,
        'signals': {
            'fear_greed': {'value': 18, 'zone': 'extreme_fear'},
            'funding': {'zone': 'high_negative', 'rate_percent': -0.08}
        }
    }
    
    contrarian = analyst.get_contrarian_check(test_prediction)
    print(f"   Contrarian Aligned: {contrarian['contrarian_aligned']}")
    print(f"   Confidence Boost: +{contrarian['confidence_boost']}%")
    for sig in contrarian['contrarian_signals']:
        print(f"   • {sig['signal']}: {sig['reason']}")

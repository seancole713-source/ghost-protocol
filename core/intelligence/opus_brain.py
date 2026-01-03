"""
Ghost Protocol - Opus Brain
Uses Claude Opus 4.5 API as Ghost's reasoning engine.

Instead of rule-based analysis, we ask Claude to THINK about:
- What's happening with this asset?
- What does the news mean?
- What did history teach us?
- What are humans likely to do?
- What's the smart play here?

This is the "weatherman" approach - use intelligence, not just data.
"""

import os
import logging
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPUS_MODEL = "claude-sonnet-4-20250514"  # Using Sonnet 4 for cost efficiency, can upgrade to Opus


class OpusBrain:
    """
    Claude as Ghost's reasoning brain.
    
    For each prediction, Claude will:
    1. Analyze current price action
    2. Read and understand recent news
    3. Consider historical patterns
    4. Think about human psychology
    5. Provide reasoned recommendation
    """
    
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.model = OPUS_MODEL
        self.cache = {}
        self.cache_ttl = 900  # 15 minutes
    
    async def analyze(self, symbol: str, context: Dict) -> Dict:
        """
        Ask Claude to analyze a symbol and provide trading insight.
        
        Args:
            symbol: Ticker symbol
            context: Dict containing price data, news, indicators, etc.
        
        Returns:
            Claude's analysis with reasoning and recommendation
        """
        if not self.api_key:
            return {
                "error": "ANTHROPIC_API_KEY not configured",
                "signal": "NEUTRAL",
                "confidence_adjustment": 0,
                "reasoning": "Claude API key not set - using technical analysis only"
            }
        
        # Check cache
        cache_key = f"opus_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"[OPUS] Cache hit for {symbol}")
                return cached_data
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(symbol, context)
        
        try:
            response = await self._call_claude(prompt)
            result = self._parse_response(response, symbol)
            
            # Cache result
            self.cache[cache_key] = (result, datetime.now())
            
            return result
        except Exception as e:
            logger.error(f"Opus brain error for {symbol}: {e}")
            return {
                "error": str(e),
                "signal": "NEUTRAL",
                "confidence_adjustment": 0,
                "reasoning": f"Analysis failed: {e}"
            }
    
    def _build_analysis_prompt(self, symbol: str, context: Dict) -> str:
        """Build the analysis prompt for Claude"""
        
        # Extract context
        current_price = context.get("current_price", "Unknown")
        price_change_24h = context.get("price_change_24h", "Unknown")
        technical_signal = context.get("technical_signal", "Unknown")
        technical_confidence = context.get("technical_confidence", 0)
        news_headlines = context.get("news_headlines", [])
        insider_activity = context.get("insider_activity", "None detected")
        whale_activity = context.get("whale_activity", "None detected")
        social_sentiment = context.get("social_sentiment", "Unknown")
        upcoming_events = context.get("upcoming_events", [])
        historical_pattern = context.get("historical_pattern", "Unknown")
        volume_analysis = context.get("volume_analysis", "Unknown")
        
        # NEW: Ghost Brain intelligence
        ghost_brain_signal = context.get("ghost_brain_signal", "UNKNOWN")
        ghost_brain_confidence_adj = context.get("ghost_brain_confidence_adj", 0)
        dominant_narrative = context.get("dominant_narrative", "Unknown")
        narrative_sentiment = context.get("narrative_sentiment", "Unknown")
        influencer_activity = context.get("influencer_activity", [])
        warnings = context.get("warnings", [])
        positives = context.get("positives", [])
        
        # Format news
        news_text = "\n".join([f"- {h}" for h in news_headlines[:10]]) if news_headlines else "No recent news available"
        
        # Format events
        events_text = "\n".join([f"- {e}" for e in upcoming_events[:5]]) if upcoming_events else "No upcoming events detected"
        
        # Format influencers
        influencer_text = "\n".join([f"  - {i}" for i in influencer_activity]) if influencer_activity else "No notable influencer activity"
        
        # Format warnings/positives
        warnings_text = "\n".join([f"  ⚠️ {w}" for w in warnings]) if warnings else "None"
        positives_text = "\n".join([f"  ✅ {p}" for p in positives]) if positives else "None"
        
        prompt = f"""You are Ghost's brain - an expert market analyst AI. Analyze this trading opportunity.

## ASSET: {symbol}

## CURRENT DATA:
- Price: ${current_price}
- 24h Change: {price_change_24h}%
- Technical Signal: {technical_signal} ({technical_confidence}% confidence)
- Volume: {volume_analysis}

## 🧠 GHOST BRAIN INTELLIGENCE:
- Overall Signal: {ghost_brain_signal}
- Confidence Adjustment: {ghost_brain_confidence_adj:+}%
- Dominant Narrative: {dominant_narrative} ({narrative_sentiment})

### Key Positives:
{positives_text}

### Warnings:
{warnings_text}

### Influencer Activity:
{influencer_text}

## RECENT NEWS:
{news_text}

## MICRO SIGNALS:
- Insider Activity: {insider_activity}
- Whale Activity: {whale_activity}
- Social Sentiment: {social_sentiment}

## UPCOMING EVENTS:
{events_text}

## HISTORICAL CONTEXT:
{historical_pattern}

## YOUR TASK:
Analyze this opportunity like a professional trader. Consider:

1. **WHAT'S THE STORY?** What narrative or theme is driving this asset?

2. **WHAT ARE THE RISKS?** Any red flags in the news, insider activity, or upcoming events?

3. **WHAT DOES HISTORY SAY?** Similar situations in the past - what happened?

4. **WHAT WILL HUMANS DO?** Given fear/greed levels and news, how will retail and institutions react?

5. **WHAT'S YOUR CALL?** Should Ghost be bullish, bearish, or neutral?

## RESPONSE FORMAT (JSON):
{{
    "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence_adjustment": -30 to +30 (how much to adjust technical confidence),
    "reasoning": "2-3 sentence explanation of your analysis",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks": ["risk1", "risk2"],
    "recommendation": "One sentence actionable advice for the next 48 hours"
}}

IMPORTANT: Respond ONLY with the JSON object, no other text or markdown."""

        return prompt
    
    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API"""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        logger.info(f"[OPUS] Calling Claude API with model {self.model}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["content"][0]["text"]
                else:
                    error_text = await resp.text()
                    logger.error(f"[OPUS] Claude API error {resp.status}: {error_text}")
                    raise Exception(f"Claude API error {resp.status}: {error_text[:200]}")
    
    def _parse_response(self, response: str, symbol: str) -> Dict:
        """Parse Claude's JSON response"""
        try:
            # Clean up response (remove markdown if present)
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first and last lines (```)
                response = "\n".join(lines[1:-1])
                if response.startswith("json"):
                    response = response[4:].strip()
            
            analysis = json.loads(response)
            
            return {
                "symbol": symbol,
                "ok": True,
                "signal": analysis.get("signal", "NEUTRAL"),
                "confidence_adjustment": max(-30, min(30, analysis.get("confidence_adjustment", 0))),
                "reasoning": analysis.get("reasoning", ""),
                "key_factors": analysis.get("key_factors", []),
                "risks": analysis.get("risks", []),
                "recommendation": analysis.get("recommendation", ""),
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        except json.JSONDecodeError as e:
            logger.error(f"[OPUS] Failed to parse Claude response: {e}")
            logger.debug(f"[OPUS] Raw response: {response[:500]}")
            
            # Try to extract useful info even if JSON parsing fails
            return {
                "symbol": symbol,
                "ok": False,
                "signal": "NEUTRAL",
                "confidence_adjustment": 0,
                "reasoning": response[:500] if response else "Failed to parse response",
                "key_factors": [],
                "risks": [],
                "recommendation": "",
                "error": "Failed to parse JSON response",
                "raw_response": response[:1000]
            }
    
    async def deep_research(self, symbol: str, question: str = None) -> Dict:
        """
        Ask Claude to do deep research on a symbol.
        
        This is for more open-ended analysis, like:
        - "What's the bull case for ETH right now?"
        - "Why did BTC crash in March 2020?"
        - "What should I watch for with AAPL earnings?"
        """
        if not self.api_key:
            return {"ok": False, "error": "ANTHROPIC_API_KEY not configured"}
        
        if question:
            prompt = f"""You are Ghost's research brain - an expert market analyst AI. Answer this question about {symbol}:

QUESTION: {question}

Provide a thorough but concise answer (3-5 paragraphs max).
Include specific data points, dates, and numbers where relevant.
End with actionable insights for a trader looking at the next 48 hours to 2 weeks.

Be direct and specific. No fluff."""
        else:
            prompt = f"""You are Ghost's research brain - an expert market analyst AI. Provide a comprehensive analysis of {symbol}.

Cover these points:
1. **Current Situation**: What's happening right now with this asset?
2. **Bull Case**: What are the strongest arguments for going long?
3. **Bear Case**: What are the key risks and bearish arguments?
4. **Key Levels**: Important support and resistance levels to watch
5. **Catalysts**: Upcoming events that could move the price
6. **Smart Money**: What are institutions likely doing?
7. **Bottom Line**: Your overall assessment and recommended action

Be specific with numbers, percentages, and price levels.
Keep it actionable for a trader."""

        try:
            response = await self._call_claude(prompt)
            return {
                "ok": True,
                "symbol": symbol,
                "question": question,
                "research": response,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[OPUS] Research error for {symbol}: {e}")
            return {"ok": False, "error": str(e)}
    
    async def explain_move(self, symbol: str, move_description: str) -> Dict:
        """
        Ask Claude to explain a price move.
        
        Example: "BTC dropped 5% in the last hour"
        """
        if not self.api_key:
            return {"ok": False, "error": "ANTHROPIC_API_KEY not configured"}
        
        prompt = f"""You are Ghost's market explainer - an expert at understanding why markets move.

Explain this move:
ASSET: {symbol}
MOVE: {move_description}

In 2-3 concise paragraphs, answer:
1. **What caused this?** Most likely catalyst or reason for the move
2. **Is this significant?** Real trend change or just noise/volatility?
3. **What now?** What should traders do in response?

Be specific and actionable. Reference recent news, technical levels, or market context if relevant."""

        try:
            response = await self._call_claude(prompt)
            return {
                "ok": True,
                "symbol": symbol,
                "move": move_description,
                "explanation": response,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[OPUS] Explain error for {symbol}: {e}")
            return {"ok": False, "error": str(e)}
    
    async def compare_assets(self, symbols: List[str], question: str = None) -> Dict:
        """
        Ask Claude to compare multiple assets.
        
        Example: Compare BTC vs ETH for the next month
        """
        if not self.api_key:
            return {"ok": False, "error": "ANTHROPIC_API_KEY not configured"}
        
        symbols_str = ", ".join(symbols)
        
        if question:
            prompt = f"""You are Ghost's comparison analyst. Compare these assets: {symbols_str}

QUESTION: {question}

Provide a clear comparison with specific recommendations."""
        else:
            prompt = f"""You are Ghost's comparison analyst. Compare these assets for trading: {symbols_str}

For each asset, briefly cover:
1. Current momentum (bullish/bearish/neutral)
2. Key levels
3. Upcoming catalysts

Then provide:
- **Best opportunity**: Which one has the best risk/reward right now?
- **Avoid**: Which one(s) should traders be cautious about?
- **Recommendation**: Specific action to take

Be concise and actionable."""

        try:
            response = await self._call_claude(prompt)
            return {
                "ok": True,
                "symbols": symbols,
                "question": question,
                "comparison": response,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[OPUS] Compare error: {e}")
            return {"ok": False, "error": str(e)}


# Singleton
_brain = None

def get_opus_brain() -> OpusBrain:
    global _brain
    if _brain is None:
        _brain = OpusBrain()
    return _brain


async def opus_analyze(symbol: str, context: Dict) -> Dict:
    """Quick access to Opus analysis"""
    return await get_opus_brain().analyze(symbol, context)


async def opus_research(symbol: str, question: str = None) -> Dict:
    """Quick access to Opus research"""
    return await get_opus_brain().deep_research(symbol, question)


async def opus_explain(symbol: str, move: str) -> Dict:
    """Quick access to move explanation"""
    return await get_opus_brain().explain_move(symbol, move)


async def opus_compare(symbols: List[str], question: str = None) -> Dict:
    """Quick access to asset comparison"""
    return await get_opus_brain().compare_assets(symbols, question)

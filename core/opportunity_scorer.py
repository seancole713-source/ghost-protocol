#!/usr/bin/env python3
"""
🎯 GHOST OPPORTUNITY SCORING SYSTEM

Ranks every investment opportunity from 0-100 based on multiple factors.
This is how Ghost determines the BEST opportunities to alert you about.

Scoring Formula:
- AI Prediction Confidence: 0-40 points
- Volume Anomaly Strength: 0-20 points
- Sentiment Score: 0-20 points
- Technical Pattern Strength: 0-10 points
- Timeframe Urgency: 0-10 points

Total: 100 points maximum
"""

import logging
from typing import Any

LOGGER = logging.getLogger("ghost.opportunity_scorer")

# Scoring weights
WEIGHT_CONFIDENCE = 40
WEIGHT_VOLUME = 20
WEIGHT_SENTIMENT = 20
WEIGHT_TECHNICAL = 10
WEIGHT_URGENCY = 10


def score_confidence(confidence: float) -> float:
    """
    Score AI prediction confidence (0-40 points).
    
    Args:
        confidence: AI confidence (0-1)
        
    Returns:
        Score 0-40
    """
    # Linear scaling: 0.5 confidence = 20 points, 1.0 = 40 points
    return max(0, min(WEIGHT_CONFIDENCE, confidence * WEIGHT_CONFIDENCE))


def score_volume_anomaly(volume_ratio: float | None) -> float:
    """
    Score volume anomaly strength (0-20 points).
    
    Args:
        volume_ratio: Current volume / average volume (e.g., 3.5 = 350% volume)
        
    Returns:
        Score 0-20
    """
    if volume_ratio is None:
        return 0

    # Scoring:
    # 1x (normal) = 0 points
    # 3x = 10 points
    # 5x = 15 points
    # 10x+ = 20 points

    if volume_ratio <= 1.0:
        return 0
    elif volume_ratio <= 3.0:
        return 10 * ((volume_ratio - 1.0) / 2.0)  # 0-10 points
    elif volume_ratio <= 5.0:
        return 10 + (5 * ((volume_ratio - 3.0) / 2.0))  # 10-15 points
    else:
        return min(WEIGHT_VOLUME, 15 + (5 * ((volume_ratio - 5.0) / 5.0)))  # 15-20 points


def score_sentiment(sentiment: float | None) -> float:
    """
    Score sentiment (0-20 points).
    
    Args:
        sentiment: Sentiment score (-1 to +1, where +1 is most bullish)
        
    Returns:
        Score 0-20
    """
    if sentiment is None:
        return 10  # Neutral default

    # Scoring:
    # -1.0 (very bearish) = 0 points
    #  0.0 (neutral) = 10 points
    # +1.0 (very bullish) = 20 points

    return max(0, min(WEIGHT_SENTIMENT, 10 + (sentiment * 10)))


def score_technical_pattern(momentum_pct: float | None) -> float:
    """
    Score technical pattern strength (0-10 points).
    
    Args:
        momentum_pct: Price momentum % (e.g., +5.5 = up 5.5%)
        
    Returns:
        Score 0-10
    """
    if momentum_pct is None:
        return 0

    # Scoring based on absolute momentum:
    # 0-3% = 3 points
    # 3-5% = 5 points
    # 5-10% = 7 points
    # 10%+ = 10 points

    abs_momentum = abs(momentum_pct)

    if abs_momentum < 3.0:
        return 3
    elif abs_momentum < 5.0:
        return 5
    elif abs_momentum < 10.0:
        return 7
    else:
        return WEIGHT_URGENCY


def score_timeframe_urgency(timeframe_hours: int) -> float:
    """
    Score timeframe urgency (0-10 points).
    Shorter timeframes = higher urgency = higher score.
    
    Args:
        timeframe_hours: Prediction timeframe (2-48 hours)
        
    Returns:
        Score 0-10
    """
    # Scoring:
    # 2h = 10 points (immediate)
    # 6h = 8 points
    # 12h = 6 points
    # 24h = 4 points
    # 48h = 2 points

    if timeframe_hours <= 2:
        return 10
    elif timeframe_hours <= 6:
        return 8
    elif timeframe_hours <= 12:
        return 6
    elif timeframe_hours <= 24:
        return 4
    else:
        return 2


def calculate_opportunity_score(opportunity: dict[str, Any]) -> int:
    """
    Calculate total opportunity score (0-100).
    
    Args:
        opportunity: Opportunity dict with fields:
            - confidence: AI confidence (required)
            - signals: {volume_anomaly, momentum}
            - timeframe_hours: prediction window
            - sentiment: optional sentiment score
            
    Returns:
        Total score (0-100)
    """
    try:
        # Extract fields
        confidence = opportunity.get("confidence", 0.5)
        signals = opportunity.get("signals", {})
        timeframe_hours = opportunity.get("timeframe_hours", 24)
        sentiment = opportunity.get("sentiment")

        # Get signal data
        volume_data = signals.get("volume_anomaly") if isinstance(signals, dict) else None
        momentum_data = signals.get("momentum") if isinstance(signals, dict) else None

        # Calculate component scores
        confidence_score = score_confidence(confidence)

        volume_ratio = None
        if volume_data and isinstance(volume_data, dict):
            volume_ratio = volume_data.get("volume_ratio")
        elif isinstance(volume_data, (int, float)):
            volume_ratio = volume_data
        volume_score = score_volume_anomaly(volume_ratio)

        sentiment_score = score_sentiment(sentiment)

        momentum_pct = None
        if momentum_data and isinstance(momentum_data, dict):
            momentum_pct = momentum_data.get("change_pct")
        elif opportunity.get("predicted_pct"):
            momentum_pct = opportunity.get("predicted_pct")
        technical_score = score_technical_pattern(momentum_pct)

        urgency_score = score_timeframe_urgency(timeframe_hours)

        # Total score
        total = (
            confidence_score
            + volume_score
            + sentiment_score
            + technical_score
            + urgency_score
        )

        # Round to integer
        final_score = int(round(total))

        # Log breakdown
        LOGGER.debug(
            f"Score breakdown for {opportunity.get('symbol', '???')}: "
            f"confidence={confidence_score:.1f}, volume={volume_score:.1f}, "
            f"sentiment={sentiment_score:.1f}, technical={technical_score:.1f}, "
            f"urgency={urgency_score:.1f} → TOTAL={final_score}"
        )

        return final_score

    except Exception as e:
        LOGGER.error(f"Failed to calculate opportunity score: {e}")
        return 0


def rank_opportunities(opportunities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rank opportunities by score and add rank field.
    
    Args:
        opportunities: List of opportunity dicts
        
    Returns:
        Sorted list with 'score' and 'rank' fields added
    """
    try:
        # Calculate scores
        for opp in opportunities:
            opp["score"] = calculate_opportunity_score(opp)

        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Add rank field
        for i, opp in enumerate(opportunities, start=1):
            opp["rank"] = i

        LOGGER.info(
            f"📊 Ranked {len(opportunities)} opportunities "
            f"(top score: {opportunities[0].get('score', 0)} if opportunities else 0)"
        )

        return opportunities

    except Exception as e:
        LOGGER.error(f"Failed to rank opportunities: {e}")
        return opportunities


def get_score_grade(score: int) -> str:
    """
    Convert numeric score to letter grade.
    
    Args:
        score: 0-100
        
    Returns:
        Grade: S, A, B, C, D, F
    """
    if score >= 90:
        return "S"  # S-tier (exceptional)
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def get_score_emoji(score: int) -> str:
    """
    Get emoji for score.
    
    Args:
        score: 0-100
        
    Returns:
        Emoji representing score quality
    """
    if score >= 90:
        return "🔥"  # Fire - exceptional opportunity
    elif score >= 80:
        return "⭐"  # Star - great opportunity
    elif score >= 70:
        return "✨"  # Sparkles - good opportunity
    elif score >= 60:
        return "👍"  # Thumbs up - decent
    elif score >= 50:
        return "😐"  # Neutral - mediocre
    else:
        return "⚠️"  # Warning - low confidence


def format_opportunity_summary(opportunity: dict[str, Any]) -> str:
    """
    Format opportunity as human-readable summary.
    
    Args:
        opportunity: Opportunity dict with score
        
    Returns:
        Formatted string for Telegram/UI
    """
    symbol = opportunity.get("symbol", "???")
    score = opportunity.get("score", 0)
    confidence = opportunity.get("confidence", 0)
    predicted_pct = opportunity.get("predicted_pct", 0)
    timeframe_hours = opportunity.get("timeframe_hours", 24)
    action = opportunity.get("action", "HOLD")

    emoji = get_score_emoji(score)
    grade = get_score_grade(score)

    summary = (
        f"{emoji} **{symbol}** (Score: {score}/100 - Grade {grade})\n"
        f"Action: {action} | Predicted: {predicted_pct:+.1f}% in {timeframe_hours}h\n"
        f"Confidence: {confidence:.0%}"
    )

    return summary


if __name__ == "__main__":
    # Test scoring system
    logging.basicConfig(level=logging.DEBUG)

    print("🎯 Testing Ghost Opportunity Scoring System")
    print()

    # Test opportunity 1: High-confidence with volume surge
    test_opp1 = {
        "symbol": "AAPL",
        "confidence": 0.89,
        "predicted_pct": 7.5,
        "timeframe_hours": 6,
        "signals": {"volume_anomaly": {"volume_ratio": 4.2}, "momentum": {"change_pct": 5.2}},
        "sentiment": 0.7,
    }

    score1 = calculate_opportunity_score(test_opp1)
    print(f"Test 1 - High confidence + volume: {score1}/100")
    print(format_opportunity_summary({**test_opp1, "score": score1, "action": "BUY"}))
    print()

    # Test opportunity 2: Medium confidence, longer timeframe
    test_opp2 = {
        "symbol": "TSLA",
        "confidence": 0.72,
        "predicted_pct": 4.2,
        "timeframe_hours": 24,
        "signals": {},
        "sentiment": 0.3,
    }

    score2 = calculate_opportunity_score(test_opp2)
    print(f"Test 2 - Medium confidence: {score2}/100")
    print(format_opportunity_summary({**test_opp2, "score": score2, "action": "BUY"}))
    print()

    # Test ranking
    opportunities = [test_opp1, test_opp2]
    ranked = rank_opportunities(opportunities)

    print("Ranked Opportunities:")
    for opp in ranked:
        print(f"  #{opp['rank']}: {opp['symbol']} - {opp['score']}/100")

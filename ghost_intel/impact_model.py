"""
GHOST INTEL - IMPACT SCORING MODEL
====================================
The core intelligence engine that scores events on market impact.

Weighted scoring (0-100):
- A) Policy/Rate Sensitivity (0-25)
- B) Liquidity/Conditions (0-20)
- C) Earnings/Guidance Relevance (0-15)
- D) Geopolitical Shock Potential (0-15)
- E) Virality/Attention Velocity (0-10)
- F) Positioning Fragility (0-15)

Credibility multiplier: 0.55 - 1.0

Author: Ghost AI
Date: 2026-01-26
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ghost_intel.normalize import (
    IntelEvent, EventLayer, EventDirection, EventScope, 
    EventHorizon, SourceTier
)

logger = logging.getLogger("ghost.intel")


@dataclass
class ImpactScore:
    """Result of impact scoring"""
    score: float                           # 0-100
    raw_score: float                       # Before multiplier
    multiplier: float                      # Credibility multiplier
    
    # Impact assessment
    direction: EventDirection
    scope: EventScope
    horizon: EventHorizon
    confidence: float                      # 0-1
    
    # Component scores
    rate_sensitivity: float
    liquidity_score: float
    earnings_relevance: float
    geopolitical_score: float
    virality_score: float
    positioning_fragility: float
    
    # Analysis
    rationale: str
    action_signal: str                     # "WATCH", "PREPARE", "ACT", "IGNORE"
    affected_tickers: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "score": round(self.score, 1),
            "raw_score": round(self.raw_score, 1),
            "multiplier": round(self.multiplier, 2),
            "direction": self.direction.value,
            "scope": self.scope.value,
            "horizon": self.horizon.value,
            "confidence": round(self.confidence, 2),
            "components": {
                "rate_sensitivity": round(self.rate_sensitivity, 1),
                "liquidity": round(self.liquidity_score, 1),
                "earnings": round(self.earnings_relevance, 1),
                "geopolitical": round(self.geopolitical_score, 1),
                "virality": round(self.virality_score, 1),
                "positioning": round(self.positioning_fragility, 1),
            },
            "rationale": self.rationale,
            "action_signal": self.action_signal,
            "affected_tickers": self.affected_tickers,
            "affected_sectors": self.affected_sectors,
        }


class ImpactScorer:
    """
    The core impact scoring engine.
    
    Takes an IntelEvent and market context, returns ImpactScore.
    """
    
    def __init__(self):
        # Current market context (updated externally)
        self.market_context = {
            "vix": 15.0,
            "put_call_ratio": 0.9,
            "spread_2s10s": 0.5,
            "dxy": 104.0,
        }
    
    def update_context(self, context: Dict[str, Any]):
        """Update market context for scoring"""
        self.market_context.update(context)
    
    def score(self, event: IntelEvent) -> ImpactScore:
        """
        Calculate impact score for an event.
        
        Args:
            event: Normalized IntelEvent
        
        Returns:
            ImpactScore with full breakdown
        """
        logger.debug(f"[INTEL] Scoring event: {event.headline[:50]}...")
        
        # Calculate component scores
        rate_sens = self._rate_sensitivity_score(event)
        liquidity = self._liquidity_score(event)
        earnings = self._earnings_relevance_score(event)
        geopolitical = self._geopolitical_score(event)
        virality = self._virality_score(event)
        positioning = self._positioning_fragility_score(event)
        
        # Sum raw score
        raw_score = rate_sens + liquidity + earnings + geopolitical + virality + positioning
        
        # Apply credibility multiplier
        multiplier = self._get_credibility_multiplier(event)
        final_score = raw_score * multiplier
        
        # Determine direction
        direction = self._determine_direction(event)
        
        # Determine horizon
        horizon = self._determine_horizon(event)
        
        # Update event with assessments
        event.direction = direction
        event.horizon = horizon
        
        # Build rationale
        rationale = self._build_rationale(
            event, rate_sens, liquidity, earnings, 
            geopolitical, virality, positioning
        )
        
        # Determine action signal
        action_signal = self._determine_action(final_score, event)
        
        # Tag event if unverified
        if multiplier < 0.7:
            if "UNVERIFIED" not in event.tags:
                event.tags.append("UNVERIFIED")
        
        return ImpactScore(
            score=min(100, final_score),
            raw_score=raw_score,
            multiplier=multiplier,
            direction=direction,
            scope=event.scope,
            horizon=horizon,
            confidence=multiplier,
            rate_sensitivity=rate_sens,
            liquidity_score=liquidity,
            earnings_relevance=earnings,
            geopolitical_score=geopolitical,
            virality_score=virality,
            positioning_fragility=positioning,
            rationale=rationale,
            action_signal=action_signal,
            affected_tickers=event.tickers,
            affected_sectors=event.sectors,
        )
    
    # =========================================================================
    # COMPONENT SCORERS (0-25, 0-20, 0-15, 0-15, 0-10, 0-15)
    # =========================================================================
    
    def _rate_sensitivity_score(self, event: IntelEvent) -> float:
        """
        A) Policy/Rate Sensitivity (0-25)
        Does this change rate expectations?
        """
        max_score = 25.0
        
        # High-impact rate events
        rate_keywords = {
            "fomc": 25,
            "fed": 20,
            "interest rate": 25,
            "rate hike": 25,
            "rate cut": 25,
            "hawkish": 20,
            "dovish": 20,
            "inflation": 18,
            "cpi": 22,
            "pce": 20,
            "employment": 15,
            "jobs": 15,
            "nfp": 20,
            "powell": 22,
            "yellen": 15,
            "treasury": 12,
            "yield": 15,
            "taper": 20,
            "qe": 18,
            "quantitative": 18,
        }
        
        # Check layer - macro events are rate-sensitive
        if event.layer == EventLayer.MACRO:
            base_score = 15.0
        elif event.layer == EventLayer.RATES:
            base_score = 12.0
        else:
            base_score = 0.0
        
        # Keyword matching
        text = f"{event.headline} {event.description or ''}".lower()
        keyword_score = 0.0
        for keyword, weight in rate_keywords.items():
            if keyword in text:
                keyword_score = max(keyword_score, weight)
        
        return min(max_score, base_score + keyword_score * 0.4)
    
    def _liquidity_score(self, event: IntelEvent) -> float:
        """
        B) Liquidity/Conditions (0-20)
        Is market fragile right now?
        """
        max_score = 20.0
        score = 0.0
        
        # VIX-based fragility
        vix = self.market_context.get("vix", 15)
        if vix > 30:
            score += 15  # Extremely fragile
        elif vix > 25:
            score += 10
        elif vix > 20:
            score += 5
        
        # P/C ratio fragility
        pcr = self.market_context.get("put_call_ratio", 0.9)
        if pcr > 1.2:
            score += 5  # High put buying = fragile
        
        # Yield curve - inverted = fragile
        spread = self.market_context.get("spread_2s10s", 0.5)
        if spread < 0:
            score += 5  # Inverted curve
        
        return min(max_score, score)
    
    def _earnings_relevance_score(self, event: IntelEvent) -> float:
        """
        C) Earnings/Guidance Relevance (0-15)
        Does this change forward earnings?
        """
        max_score = 15.0
        
        if event.layer != EventLayer.CORPORATE:
            return 0.0
        
        earnings_keywords = {
            "earnings": 15,
            "guidance": 15,
            "outlook": 12,
            "revenue": 12,
            "profit": 10,
            "beat": 10,
            "miss": 12,  # Miss is more impactful
            "raised": 10,
            "lowered": 12,
            "cut": 12,
            "warning": 15,
            "buyback": 8,
            "dividend": 6,
        }
        
        text = f"{event.headline} {event.description or ''}".lower()
        score = 0.0
        for keyword, weight in earnings_keywords.items():
            if keyword in text:
                score = max(score, weight)
        
        return min(max_score, score)
    
    def _geopolitical_score(self, event: IntelEvent) -> float:
        """
        D) Geopolitical Shock Potential (0-15)
        Energy, trade, conflict risk?
        """
        max_score = 15.0
        
        if event.layer not in [EventLayer.GEOPOLITICS, EventLayer.POLITICS]:
            return 0.0
        
        geo_keywords = {
            "war": 15,
            "conflict": 12,
            "invasion": 15,
            "sanction": 12,
            "tariff": 10,
            "oil": 10,
            "energy": 8,
            "opec": 10,
            "china": 8,
            "russia": 10,
            "taiwan": 12,
            "nuclear": 15,
            "missile": 12,
            "attack": 12,
            "embargo": 12,
            "blockade": 12,
            "shipping": 8,
            "suez": 10,
            "strait": 10,
        }
        
        text = f"{event.headline} {event.description or ''}".lower()
        score = 0.0
        for keyword, weight in geo_keywords.items():
            if keyword in text:
                score = max(score, weight)
        
        return min(max_score, score)
    
    def _virality_score(self, event: IntelEvent) -> float:
        """
        E) Virality/Attention Velocity (0-10)
        How fast is this spreading?
        """
        max_score = 10.0
        
        # Social layer events get base virality
        if event.layer == EventLayer.SOCIAL:
            base = 5.0
        elif event.layer == EventLayer.INDIVIDUALS:
            base = 6.0
        else:
            base = 0.0
        
        # Corroborated events are more viral
        if event.corroborated:
            base += 3.0
        
        # Multiple sources
        if event.source_count >= 3:
            base += 2.0
        
        # Breaking news flag
        if event.is_breaking:
            base += 3.0
        
        return min(max_score, base)
    
    def _positioning_fragility_score(self, event: IntelEvent) -> float:
        """
        F) Positioning Fragility (0-15)
        Is market positioned against this?
        """
        max_score = 15.0
        score = 0.0
        
        # P/C ratio indicates positioning
        pcr = self.market_context.get("put_call_ratio", 0.9)
        
        # High puts = positioned for downside
        # Bullish event against bearish positioning = high impact
        if pcr > 1.0:
            if self._determine_direction(event) == EventDirection.BULLISH:
                score += 10  # Market positioned wrong
            else:
                score += 3   # Market positioned right
        else:
            if self._determine_direction(event) == EventDirection.BEARISH:
                score += 10  # Market positioned wrong (too bullish)
            else:
                score += 3
        
        # VIX spike potential
        vix = self.market_context.get("vix", 15)
        if vix < 15 and event.layer in [EventLayer.GEOPOLITICS, EventLayer.MACRO]:
            score += 5  # Low vol + shock = big impact
        
        return min(max_score, score)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _get_credibility_multiplier(self, event: IntelEvent) -> float:
        """Get credibility multiplier based on source tier"""
        tier_multipliers = {
            SourceTier.TIER1: 1.0,
            SourceTier.TIER2: 0.95,
            SourceTier.TIER3: 0.85,
            SourceTier.TIER4: 0.70,
            SourceTier.TIER5: 0.55,
        }
        
        base = tier_multipliers.get(event.source_tier, 0.55)
        
        # Boost if corroborated
        if event.corroborated:
            base = min(1.0, base + 0.15)
        
        return base
    
    def _determine_direction(self, event: IntelEvent) -> EventDirection:
        """Determine market direction from event"""
        bullish_keywords = [
            "beat", "strong", "surge", "rally", "bullish", "growth",
            "dovish", "cut", "stimulus", "easing", "raised guidance",
            "buyback", "upgrade", "outperform"
        ]
        
        bearish_keywords = [
            "miss", "weak", "drop", "crash", "bearish", "recession",
            "hawkish", "hike", "tightening", "lowered guidance",
            "warning", "downgrade", "underperform", "tariff",
            "sanction", "conflict", "war"
        ]
        
        text = f"{event.headline} {event.description or ''}".lower()
        
        bullish_count = sum(1 for k in bullish_keywords if k in text)
        bearish_count = sum(1 for k in bearish_keywords if k in text)
        
        if bullish_count > bearish_count + 1:
            return EventDirection.BULLISH
        elif bearish_count > bullish_count + 1:
            return EventDirection.BEARISH
        elif bullish_count > 0 and bearish_count > 0:
            return EventDirection.MIXED
        else:
            return EventDirection.NEUTRAL
    
    def _determine_horizon(self, event: IntelEvent) -> EventHorizon:
        """Determine time horizon of impact"""
        # Scheduled events have known horizons
        if event.is_scheduled:
            if event.category in ["fomc", "cpi", "nfp"]:
                return EventHorizon.SAME_DAY
            elif event.category in ["earnings"]:
                return EventHorizon.MULTI_DAY
        
        # Layer-based defaults
        if event.layer == EventLayer.MACRO:
            return EventHorizon.MULTI_DAY
        elif event.layer == EventLayer.RATES:
            return EventHorizon.SAME_DAY
        elif event.layer == EventLayer.GEOPOLITICS:
            return EventHorizon.WEEKS
        elif event.layer == EventLayer.SOCIAL:
            return EventHorizon.IMMEDIATE
        elif event.layer == EventLayer.INDIVIDUALS:
            return EventHorizon.IMMEDIATE
        elif event.layer == EventLayer.POSITIONING:
            return EventHorizon.IMMEDIATE
        else:
            return EventHorizon.SAME_DAY
    
    def _build_rationale(
        self, event: IntelEvent,
        rate_sens: float, liquidity: float, earnings: float,
        geopolitical: float, virality: float, positioning: float
    ) -> str:
        """Build human-readable rationale"""
        parts = []
        
        # Top contributing factors
        factors = [
            ("Rate sensitivity", rate_sens, 25),
            ("Liquidity conditions", liquidity, 20),
            ("Earnings relevance", earnings, 15),
            ("Geopolitical risk", geopolitical, 15),
            ("Virality", virality, 10),
            ("Positioning fragility", positioning, 15),
        ]
        
        # Sort by contribution
        factors.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 factors
        top_factors = [f for f in factors if f[1] > f[2] * 0.3][:3]
        
        if top_factors:
            factor_text = ", ".join([f[0] for f in top_factors])
            parts.append(f"Key drivers: {factor_text}")
        
        # Source credibility
        if event.source_tier == SourceTier.TIER1:
            parts.append("Official source (Tier 1)")
        elif event.source_tier == SourceTier.TIER5:
            parts.append("Unverified source (Tier 5) - treat with caution")
        
        # Corroboration
        if event.corroborated:
            parts.append(f"Corroborated by {event.source_count} sources")
        
        return ". ".join(parts)
    
    def _determine_action(self, score: float, event: IntelEvent) -> str:
        """Determine action signal based on score"""
        if score >= 70:
            return "ACT"       # High impact - immediate attention
        elif score >= 50:
            return "PREPARE"   # Significant - prepare for impact
        elif score >= 30:
            return "WATCH"     # Monitor situation
        else:
            return "IGNORE"    # Low impact - noise


def is_signal_not_noise(event: IntelEvent, scorer: ImpactScorer) -> bool:
    """
    Four questions that matter for filtering signal from noise.
    
    1. Does it change rates or liquidity expectations?
    2. Does it change forward earnings meaningfully?
    3. Is positioning fragile?
    4. Is it corroborated?
    """
    # 1. Does it change rates or liquidity?
    changes_rates = event.layer in [EventLayer.MACRO, EventLayer.RATES]
    
    # 2. Does it change earnings?
    changes_earnings = event.layer == EventLayer.CORPORATE
    
    # 3. Is positioning fragile?
    vix = scorer.market_context.get("vix", 15)
    pcr = scorer.market_context.get("put_call_ratio", 0.9)
    positioning_fragile = vix > 20 or pcr > 1.2
    
    # If no to 1-3, it's likely noise
    if not (changes_rates or changes_earnings or positioning_fragile):
        # Unless it's high-impact geopolitical
        if event.layer == EventLayer.GEOPOLITICS:
            pass  # Still consider it
        else:
            return False  # NOISE
    
    return True  # SIGNAL


def detect_narrative_lag(event: IntelEvent, symbol: str, price_func) -> bool:
    """
    Did price move before the news?
    If so, the news didn't cause the move - it's narrative lag.
    
    Args:
        event: The event to check
        symbol: Symbol to check price for
        price_func: Function(symbol, timestamp) -> price
    
    Returns:
        True if price led the news (narrative lag detected)
    """
    if not event.event_time:
        return False
    
    event_time = event.event_time
    
    try:
        # Get prices around the event
        price_before = price_func(symbol, event_time - 1800)  # 30 min before
        price_at = price_func(symbol, event_time)
        price_after = price_func(symbol, event_time + 1800)   # 30 min after
        
        if not all([price_before, price_at, price_after]):
            return False
        
        # Calculate moves
        move_before = abs(price_at - price_before) / price_before
        move_after = abs(price_after - price_at) / price_at
        
        # If move before > move after and significant, price led
        if move_before > move_after and move_before > 0.005:  # 0.5% threshold
            logger.info(f"[INTEL] Narrative lag detected for {event.headline[:50]}... - price led by {move_before*100:.1f}%")
            return True
        
        return False
    except Exception as e:
        logger.error(f"[INTEL] Narrative lag detection failed: {e}")
        return False


# Singleton scorer
_scorer: Optional[ImpactScorer] = None


def get_impact_scorer() -> ImpactScorer:
    """Get singleton scorer instance"""
    global _scorer
    if _scorer is None:
        _scorer = ImpactScorer()
    return _scorer

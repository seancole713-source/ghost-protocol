"""
EVENT MEMORY - Ghost learns from WHAT HAPPENED, not just outcomes

Human traders learn patterns:
- "When Elon tweets, DOGE pumps then dumps"
- "When Fed raises rates, crypto drops 5-10%"
- "When exchange gets hacked, flash crash then recovery"
- "When whale dumps, cascade selling follows"

Ghost should learn these patterns too.

This module:
1. Detects events (news, tweets, whale moves, fed announcements)
2. Tracks what happened to prices AFTER the event
3. Stores the pattern (event → reaction)
4. Uses patterns to adjust future predictions
"""

import os
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

LOGGER = logging.getLogger(__name__)

# ============================================================================
# EVENT TYPES - Categories of market-moving events
# ============================================================================

class EventType(Enum):
    # Social/Influencer
    ELON_TWEET = "elon_tweet"
    CELEBRITY_MENTION = "celebrity_mention"
    VIRAL_SOCIAL = "viral_social"
    
    # Regulatory/Government
    FED_RATE_DECISION = "fed_rate_decision"
    SEC_ACTION = "sec_action"
    COUNTRY_BAN = "country_ban"
    REGULATION_NEWS = "regulation_news"
    
    # Exchange/Security
    EXCHANGE_HACK = "exchange_hack"
    EXCHANGE_LISTING = "exchange_listing"
    EXCHANGE_DELISTING = "exchange_delisting"
    EXCHANGE_OUTAGE = "exchange_outage"
    
    # Whale/Large Movements
    WHALE_BUY = "whale_buy"
    WHALE_SELL = "whale_sell"
    LARGE_TRANSFER = "large_transfer"
    
    # Project/Company
    CEO_NEWS = "ceo_news"
    PARTNERSHIP = "partnership"
    PRODUCT_LAUNCH = "product_launch"
    EARNINGS_REPORT = "earnings_report"
    
    # Macro/Geopolitical
    WAR_CONFLICT = "war_conflict"
    ECONOMIC_DATA = "economic_data"
    MARKET_CRASH = "market_crash"
    
    # Technical
    HALVING = "halving"
    FORK = "fork"
    UPGRADE = "upgrade"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class EventPattern:
    """A learned pattern: when X happens, expect Y"""
    event_type: str
    keywords: List[str]  # What triggered detection
    affected_symbols: List[str]  # Which symbols were affected
    
    # Price reaction patterns (learned from history)
    immediate_reaction: float  # % change in first 1-4 hours
    peak_reaction: float  # Max % change within 24-48 hours
    recovery_time_hours: int  # How long until price stabilizes
    typical_direction: str  # "pump", "dump", "volatile", "neutral"
    
    # Confidence in this pattern
    times_observed: int
    last_observed: str
    accuracy: float  # How often the pattern holds
    
    # Additional context
    notes: str


@dataclass 
class EventMemoryEntry:
    """A single event that Ghost observed and learned from"""
    event_id: str
    event_type: str
    timestamp: str
    
    # What triggered the event
    trigger: str  # "Elon tweeted about DOGE", "Fed raised rates 0.25%"
    source: str  # "twitter", "news", "on-chain", "fed"
    
    # Affected assets
    primary_symbol: str  # Main affected symbol
    related_symbols: List[str]  # Other affected symbols
    
    # Price data at event time
    price_at_event: float
    price_1h_later: float
    price_4h_later: float
    price_24h_later: float
    price_48h_later: float
    
    # Calculated reactions
    reaction_1h: float  # % change
    reaction_4h: float
    reaction_24h: float
    reaction_48h: float
    peak_reaction: float
    peak_time_hours: float
    
    # What Ghost predicted vs what happened
    ghost_prediction: Optional[str]  # "LONG" or "SHORT"
    ghost_was_right: Optional[bool]
    
    # Lessons learned
    lesson: str  # "Elon tweets cause 20% pump then 15% dump within 24h"


class EventMemory:
    """
    Ghost's memory of market events and their outcomes.
    
    This is how Ghost learns from experience:
    1. Event happens (Elon tweets, Fed announces, hack occurs)
    2. Ghost tracks what prices did after
    3. Ghost stores the pattern
    4. Next time similar event happens, Ghost knows what to expect
    """
    
    def __init__(self):
        self.db_url = os.environ.get('DATABASE_URL')
        self.events: List[EventMemoryEntry] = []
        self.patterns: Dict[str, EventPattern] = {}
        self._init_db()
        self._load_historical_patterns()
    
    def _init_db(self):
        """Create event memory tables if they don't exist"""
        if not self.db_url or 'sqlite' in self.db_url:
            LOGGER.warning("[EVENT_MEMORY] No PostgreSQL - using in-memory storage")
            return
            
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            # Events table - individual events Ghost observed
            cur.execute('''
                CREATE TABLE IF NOT EXISTS ghost_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    trigger TEXT,
                    source TEXT,
                    primary_symbol TEXT,
                    related_symbols JSONB,
                    price_at_event FLOAT,
                    price_1h_later FLOAT,
                    price_4h_later FLOAT,
                    price_24h_later FLOAT,
                    price_48h_later FLOAT,
                    reaction_1h FLOAT,
                    reaction_4h FLOAT,
                    reaction_24h FLOAT,
                    reaction_48h FLOAT,
                    peak_reaction FLOAT,
                    peak_time_hours FLOAT,
                    ghost_prediction TEXT,
                    ghost_was_right BOOLEAN,
                    lesson TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Patterns table - aggregated learnings
            cur.execute('''
                CREATE TABLE IF NOT EXISTS ghost_event_patterns (
                    event_type TEXT PRIMARY KEY,
                    keywords JSONB,
                    affected_symbols JSONB,
                    immediate_reaction FLOAT,
                    peak_reaction FLOAT,
                    recovery_time_hours INT,
                    typical_direction TEXT,
                    times_observed INT DEFAULT 0,
                    last_observed TIMESTAMP,
                    accuracy FLOAT DEFAULT 0,
                    notes TEXT,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            conn.commit()
            cur.close()
            conn.close()
            LOGGER.info("[EVENT_MEMORY] ✅ Database tables initialized")
            
        except Exception as e:
            LOGGER.error(f"[EVENT_MEMORY] DB init failed: {e}")
    
    def _load_historical_patterns(self):
        """Load known patterns from historical market data"""
        # These are KNOWN patterns from market history
        # Ghost will refine these as it observes more events
        
        self.patterns = {
            # Elon Musk tweets - well documented pattern
            EventType.ELON_TWEET.value: EventPattern(
                event_type=EventType.ELON_TWEET.value,
                keywords=["elon", "musk", "tesla", "doge", "dogecoin"],
                affected_symbols=["DOGE", "SHIB", "TSLA", "BTC"],
                immediate_reaction=15.0,  # +15% in first hours
                peak_reaction=30.0,  # Can go +30%
                recovery_time_hours=24,  # Usually dumps back within 24h
                typical_direction="pump_then_dump",
                times_observed=50,  # Many documented cases
                last_observed="2025-01-01",
                accuracy=0.85,  # Very reliable pattern
                notes="Elon tweets cause immediate pump, followed by dump within 24h. Don't chase the pump."
            ),
            
            # Fed rate decisions
            EventType.FED_RATE_DECISION.value: EventPattern(
                event_type=EventType.FED_RATE_DECISION.value,
                keywords=["fed", "federal reserve", "rate", "fomc", "powell", "interest rate"],
                affected_symbols=["BTC", "ETH", "SPY", "QQQ"],
                immediate_reaction=-5.0,  # Usually drops on rate hikes
                peak_reaction=-10.0,
                recovery_time_hours=72,
                typical_direction="dump_on_hike",
                times_observed=30,
                last_observed="2025-01-01",
                accuracy=0.75,
                notes="Rate hikes = risk-off = crypto dumps. Rate cuts = pump."
            ),
            
            # Exchange hacks
            EventType.EXCHANGE_HACK.value: EventPattern(
                event_type=EventType.EXCHANGE_HACK.value,
                keywords=["hack", "exploit", "stolen", "breach", "drained"],
                affected_symbols=["BTC", "ETH"],  # Whole market affected
                immediate_reaction=-15.0,  # Flash crash
                peak_reaction=-25.0,
                recovery_time_hours=48,  # Usually recovers
                typical_direction="flash_crash_recovery",
                times_observed=20,
                last_observed="2025-01-01",
                accuracy=0.80,
                notes="Hacks cause panic selling, but market usually recovers. Buy the dip opportunity."
            ),
            
            # Whale dumps
            EventType.WHALE_SELL.value: EventPattern(
                event_type=EventType.WHALE_SELL.value,
                keywords=["whale", "large transfer", "moved to exchange", "dump"],
                affected_symbols=[],  # Depends on which whale
                immediate_reaction=-8.0,
                peak_reaction=-15.0,
                recovery_time_hours=12,
                typical_direction="cascade_then_recovery",
                times_observed=100,
                last_observed="2025-01-01",
                accuracy=0.70,
                notes="Whale sells trigger stop losses and cascade selling. Usually oversold."
            ),
            
            # Exchange listings
            EventType.EXCHANGE_LISTING.value: EventPattern(
                event_type=EventType.EXCHANGE_LISTING.value,
                keywords=["listed on", "listing", "binance listing", "coinbase listing"],
                affected_symbols=[],
                immediate_reaction=25.0,  # Big pump on listing
                peak_reaction=50.0,
                recovery_time_hours=48,
                typical_direction="pump_then_dump",
                times_observed=200,
                last_observed="2025-01-01",
                accuracy=0.90,
                notes="Buy the rumor, sell the news. Listings pump hard then dump."
            ),
            
            # War/Conflict
            EventType.WAR_CONFLICT.value: EventPattern(
                event_type=EventType.WAR_CONFLICT.value,
                keywords=["war", "invasion", "attack", "military", "conflict", "missile"],
                affected_symbols=["BTC", "GOLD", "OIL"],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=168,  # Week to stabilize
                typical_direction="risk_off",
                times_observed=10,
                last_observed="2024-01-01",
                accuracy=0.75,
                notes="War = uncertainty = risk-off. But crypto can be safe haven long term."
            ),
            
            # Bitcoin halving
            EventType.HALVING.value: EventPattern(
                event_type=EventType.HALVING.value,
                keywords=["halving", "halvening", "block reward"],
                affected_symbols=["BTC", "ETH", "altcoins"],
                immediate_reaction=5.0,
                peak_reaction=100.0,  # Historically 2-10x over months
                recovery_time_hours=0,  # Not a dump pattern
                typical_direction="long_term_pump",
                times_observed=4,
                last_observed="2024-04-01",
                accuracy=1.0,  # 4/4 halvings led to bull runs
                notes="Every halving has led to new ATH within 12-18 months. HODL."
            ),
        }
        
        LOGGER.info(f"[EVENT_MEMORY] Loaded {len(self.patterns)} historical patterns")
    
    def detect_event_type(self, text: str) -> Tuple[EventType, float]:
        """
        Analyze text (news, tweet, etc) to detect what type of event it is
        Returns (event_type, confidence)
        """
        text_lower = text.lower()
        
        for event_type, pattern in self.patterns.items():
            matches = sum(1 for kw in pattern.keywords if kw in text_lower)
            if matches >= 2:
                confidence = min(matches / len(pattern.keywords), 1.0)
                return EventType(event_type), confidence
        
        return EventType.UNKNOWN, 0.0
    
    def get_expected_reaction(self, event_type: EventType, symbol: str) -> Dict:
        """
        Given an event type, what should Ghost expect?
        
        Returns prediction adjustments based on learned patterns.
        """
        pattern = self.patterns.get(event_type.value)
        
        if not pattern:
            return {
                "adjustment": 0,
                "confidence_modifier": 1.0,
                "expected_direction": "unknown",
                "warning": None
            }
        
        # Determine if this symbol is typically affected
        is_affected = (
            symbol in pattern.affected_symbols or 
            not pattern.affected_symbols  # Empty = affects all
        )
        
        if not is_affected:
            return {
                "adjustment": 0,
                "confidence_modifier": 1.0,
                "expected_direction": "neutral",
                "warning": None
            }
        
        # Calculate adjustment based on pattern
        direction = pattern.typical_direction
        
        if direction == "pump_then_dump":
            return {
                "adjustment": "wait",  # Don't chase
                "confidence_modifier": 0.5,  # Reduce confidence
                "expected_direction": "volatile",
                "warning": f"⚠️ {event_type.value}: Expect pump then dump. Don't chase.",
                "pattern": pattern.notes
            }
        
        elif direction == "dump_on_hike":
            return {
                "adjustment": "bearish",
                "confidence_modifier": 1.2 if pattern.accuracy > 0.7 else 1.0,
                "expected_direction": "down",
                "warning": f"⚠️ {event_type.value}: Expect downward pressure.",
                "pattern": pattern.notes
            }
        
        elif direction == "flash_crash_recovery":
            return {
                "adjustment": "buy_dip",
                "confidence_modifier": 1.3,
                "expected_direction": "recovery",
                "warning": f"🎯 {event_type.value}: Flash crash = buy opportunity.",
                "pattern": pattern.notes
            }
        
        elif direction == "long_term_pump":
            return {
                "adjustment": "bullish",
                "confidence_modifier": 1.5,
                "expected_direction": "up",
                "warning": f"🚀 {event_type.value}: Historically very bullish.",
                "pattern": pattern.notes
            }
        
        return {
            "adjustment": 0,
            "confidence_modifier": 1.0,
            "expected_direction": pattern.typical_direction,
            "warning": pattern.notes
        }
    
    def record_event(self, 
                     event_type: EventType,
                     trigger: str,
                     symbol: str,
                     price_at_event: float,
                     source: str = "manual") -> str:
        """
        Record a new event. Ghost will track what happens after.
        """
        import uuid
        event_id = str(uuid.uuid4())[:8]
        
        entry = EventMemoryEntry(
            event_id=event_id,
            event_type=event_type.value,
            timestamp=datetime.utcnow().isoformat(),
            trigger=trigger,
            source=source,
            primary_symbol=symbol,
            related_symbols=[],
            price_at_event=price_at_event,
            price_1h_later=0,
            price_4h_later=0,
            price_24h_later=0,
            price_48h_later=0,
            reaction_1h=0,
            reaction_4h=0,
            reaction_24h=0,
            reaction_48h=0,
            peak_reaction=0,
            peak_time_hours=0,
            ghost_prediction=None,
            ghost_was_right=None,
            lesson=""
        )
        
        self.events.append(entry)
        self._save_event(entry)
        
        LOGGER.info(f"[EVENT_MEMORY] 📝 Recorded event: {event_type.value} for {symbol}")
        return event_id
    
    def update_event_outcome(self, event_id: str, 
                             price_1h: float = None,
                             price_4h: float = None,
                             price_24h: float = None,
                             price_48h: float = None):
        """
        Update an event with price data as time passes.
        This is how Ghost learns - by tracking what actually happened.
        """
        for event in self.events:
            if event.event_id == event_id:
                if price_1h:
                    event.price_1h_later = price_1h
                    event.reaction_1h = ((price_1h - event.price_at_event) / event.price_at_event) * 100
                if price_4h:
                    event.price_4h_later = price_4h
                    event.reaction_4h = ((price_4h - event.price_at_event) / event.price_at_event) * 100
                if price_24h:
                    event.price_24h_later = price_24h
                    event.reaction_24h = ((price_24h - event.price_at_event) / event.price_at_event) * 100
                if price_48h:
                    event.price_48h_later = price_48h
                    event.reaction_48h = ((price_48h - event.price_at_event) / event.price_at_event) * 100
                    
                    # Calculate peak and lesson
                    reactions = [abs(event.reaction_1h), abs(event.reaction_4h), 
                                abs(event.reaction_24h), abs(event.reaction_48h)]
                    event.peak_reaction = max(reactions)
                    
                    # Learn the lesson
                    self._learn_from_event(event)
                
                self._save_event(event)
                return
    
    def _learn_from_event(self, event: EventMemoryEntry):
        """
        Extract lessons from a completed event observation.
        Update the pattern with new data.
        """
        pattern = self.patterns.get(event.event_type)
        
        if pattern:
            # Update pattern with new observation
            old_count = pattern.times_observed
            new_count = old_count + 1
            
            # Rolling average of reactions
            pattern.immediate_reaction = (
                (pattern.immediate_reaction * old_count + event.reaction_1h) / new_count
            )
            pattern.peak_reaction = (
                (pattern.peak_reaction * old_count + event.peak_reaction) / new_count
            )
            
            pattern.times_observed = new_count
            pattern.last_observed = event.timestamp
            
            # Determine if pattern held
            expected_dir = pattern.typical_direction
            actual_positive = event.reaction_48h > 0
            
            if expected_dir in ["pump", "pump_then_dump", "long_term_pump"]:
                pattern_held = event.reaction_4h > 5  # Initial pump happened
            elif expected_dir in ["dump", "dump_on_hike", "flash_crash_recovery"]:
                pattern_held = event.reaction_4h < -5  # Initial dump happened
            else:
                pattern_held = True  # Volatile = always "correct"
            
            # Update accuracy
            pattern.accuracy = (
                (pattern.accuracy * old_count + (1 if pattern_held else 0)) / new_count
            )
            
            LOGGER.info(f"[EVENT_MEMORY] 🧠 Updated {event.event_type} pattern: "
                       f"accuracy={pattern.accuracy:.1%}, observed={new_count}x")
        
        # Generate lesson
        direction = "pumped" if event.reaction_48h > 0 else "dumped"
        event.lesson = (
            f"{event.event_type}: {event.trigger} → {event.primary_symbol} {direction} "
            f"{abs(event.reaction_48h):.1f}% over 48h (peak: {event.peak_reaction:.1f}%)"
        )
    
    def _save_event(self, event: EventMemoryEntry):
        """Save event to database"""
        if not self.db_url or 'sqlite' in self.db_url:
            return
            
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute('''
                INSERT INTO ghost_events 
                (event_id, event_type, timestamp, trigger, source, primary_symbol,
                 related_symbols, price_at_event, price_1h_later, price_4h_later,
                 price_24h_later, price_48h_later, reaction_1h, reaction_4h,
                 reaction_24h, reaction_48h, peak_reaction, peak_time_hours,
                 ghost_prediction, ghost_was_right, lesson)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET
                    price_1h_later = EXCLUDED.price_1h_later,
                    price_4h_later = EXCLUDED.price_4h_later,
                    price_24h_later = EXCLUDED.price_24h_later,
                    price_48h_later = EXCLUDED.price_48h_later,
                    reaction_1h = EXCLUDED.reaction_1h,
                    reaction_4h = EXCLUDED.reaction_4h,
                    reaction_24h = EXCLUDED.reaction_24h,
                    reaction_48h = EXCLUDED.reaction_48h,
                    peak_reaction = EXCLUDED.peak_reaction,
                    lesson = EXCLUDED.lesson
            ''', (
                event.event_id, event.event_type, event.timestamp, event.trigger,
                event.source, event.primary_symbol, json.dumps(event.related_symbols),
                event.price_at_event, event.price_1h_later, event.price_4h_later,
                event.price_24h_later, event.price_48h_later, event.reaction_1h,
                event.reaction_4h, event.reaction_24h, event.reaction_48h,
                event.peak_reaction, event.peak_time_hours, event.ghost_prediction,
                event.ghost_was_right, event.lesson
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            LOGGER.error(f"[EVENT_MEMORY] Failed to save event: {e}")
    
    def get_lessons_for_symbol(self, symbol: str) -> List[str]:
        """Get all lessons Ghost has learned about a specific symbol"""
        lessons = []
        for event in self.events:
            if event.primary_symbol == symbol and event.lesson:
                lessons.append(event.lesson)
        return lessons
    
    def get_active_events(self) -> List[EventMemoryEntry]:
        """Get events from the last 48 hours that might still be affecting prices"""
        cutoff = datetime.utcnow() - timedelta(hours=48)
        return [
            e for e in self.events 
            if datetime.fromisoformat(e.timestamp) > cutoff
        ]
    
    def should_adjust_prediction(self, symbol: str, direction: str) -> Dict:
        """
        Check if any recent events should cause Ghost to adjust its prediction.
        
        This is where EVENT MEMORY meets PREDICTION LOGIC.
        """
        active_events = self.get_active_events()
        
        for event in active_events:
            if event.primary_symbol == symbol or symbol in event.related_symbols:
                pattern = self.patterns.get(event.event_type)
                if pattern:
                    expected = self.get_expected_reaction(EventType(event.event_type), symbol)
                    
                    # If Ghost's prediction conflicts with learned pattern
                    if expected["expected_direction"] == "down" and direction == "LONG":
                        return {
                            "should_adjust": True,
                            "reason": f"Recent {event.event_type} event suggests downward pressure",
                            "recommendation": "Consider SHORT or SKIP",
                            "event": event.trigger
                        }
                    
                    if expected["expected_direction"] == "volatile":
                        return {
                            "should_adjust": True,
                            "reason": f"Recent {event.event_type} event causing high volatility",
                            "recommendation": "Reduce position size or SKIP",
                            "event": event.trigger
                        }
        
        return {"should_adjust": False}


# Global instance
_event_memory = None

def get_event_memory() -> EventMemory:
    global _event_memory
    if _event_memory is None:
        _event_memory = EventMemory()
    return _event_memory


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_for_event_impact(symbol: str, prediction_direction: str) -> Dict:
    """
    Quick check: Should Ghost adjust its prediction based on recent events?
    """
    memory = get_event_memory()
    return memory.should_adjust_prediction(symbol, prediction_direction)


def record_market_event(event_type: str, trigger: str, symbol: str, price: float) -> str:
    """
    Record a market event that Ghost should learn from.
    
    Example:
        record_market_event("elon_tweet", "Elon posted DOGE meme", "DOGE", 0.08)
    """
    memory = get_event_memory()
    try:
        et = EventType(event_type)
    except:
        et = EventType.UNKNOWN
    return memory.record_event(et, trigger, symbol, price)


def get_pattern_for_event(event_type: str) -> Optional[Dict]:
    """
    Get the learned pattern for an event type.
    
    Example:
        pattern = get_pattern_for_event("elon_tweet")
        print(f"Expected reaction: {pattern['immediate_reaction']}%")
    """
    memory = get_event_memory()
    pattern = memory.patterns.get(event_type)
    if pattern:
        return asdict(pattern)
    return None

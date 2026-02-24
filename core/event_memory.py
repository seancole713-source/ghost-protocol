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
    # =========================================================================
    # A. COMPANY-SPECIFIC (STOCKS)
    # =========================================================================
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    REVENUE_GROWTH = "revenue_growth"
    REVENUE_DECLINE = "revenue_decline"
    GUIDANCE_RAISED = "guidance_raised"
    GUIDANCE_LOWERED = "guidance_lowered"
    PROFIT_MARGIN_UP = "profit_margin_up"
    PROFIT_MARGIN_DOWN = "profit_margin_down"
    MAJOR_CONTRACT_WON = "major_contract_won"
    MAJOR_CONTRACT_LOST = "major_contract_lost"
    MERGER_ANNOUNCED = "merger_announced"
    ACQUISITION_ANNOUNCED = "acquisition_announced"
    STOCK_BUYBACK = "stock_buyback"
    DIVIDEND_INCREASE = "dividend_increase"
    DIVIDEND_CUT = "dividend_cut"
    INSIDER_BUYING = "insider_buying"
    INSIDER_SELLING = "insider_selling"
    CEO_CHANGE = "ceo_change"
    CFO_CHANGE = "cfo_change"
    EXECUTIVE_SCANDAL = "executive_scandal"
    CORPORATE_FRAUD = "corporate_fraud"
    ACCOUNTING_RESTATEMENT = "accounting_restatement"
    LAWSUIT_FILED = "lawsuit_filed"
    LAWSUIT_SETTLED = "lawsuit_settled"
    REGULATORY_FINE = "regulatory_fine"
    PRODUCT_LAUNCH = "product_launch"
    PRODUCT_FAILURE = "product_failure"
    PRODUCT_RECALL = "product_recall"
    PATENT_APPROVED = "patent_approved"
    PATENT_EXPIRED = "patent_expired"
    RD_BREAKTHROUGH = "rd_breakthrough"
    SUPPLY_CHAIN_ISSUE = "supply_chain_issue"
    FACTORY_SHUTDOWN = "factory_shutdown"
    FACTORY_EXPANSION = "factory_expansion"
    CREDIT_UPGRADE = "credit_upgrade"
    CREDIT_DOWNGRADE = "credit_downgrade"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    SPINOFF = "spinoff"
    BANKRUPTCY = "bankruptcy"
    
    # =========================================================================
    # B. SECTOR & INDUSTRY
    # =========================================================================
    SECTOR_ROTATION = "sector_rotation"
    COMPETITOR_EARNINGS_SPILLOVER = "competitor_earnings_spillover"
    INDUSTRY_REGULATION = "industry_regulation"
    TARIFF_CHANGE = "tariff_change"
    TECH_DISRUPTION = "tech_disruption"
    LABOR_STRIKE = "labor_strike"
    LABOR_SHORTAGE = "labor_shortage"
    INDUSTRY_CONSOLIDATION = "industry_consolidation"
    MARKET_SHARE_SHIFT = "market_share_shift"
    
    # =========================================================================
    # C. MACRO & ECONOMIC
    # =========================================================================
    FED_RATE_HIKE = "fed_rate_hike"
    FED_RATE_CUT = "fed_rate_cut"
    FED_RATE_DECISION = "fed_rate_decision"
    FED_QE_ANNOUNCED = "fed_qe_announced"
    FED_QT_ANNOUNCED = "fed_qt_announced"
    INFLATION_HIGH = "inflation_high"
    INFLATION_LOW = "inflation_low"
    CPI_RELEASE = "cpi_release"
    JOBS_REPORT_STRONG = "jobs_report_strong"
    JOBS_REPORT_WEAK = "jobs_report_weak"
    UNEMPLOYMENT_UP = "unemployment_up"
    UNEMPLOYMENT_DOWN = "unemployment_down"
    GDP_GROWTH = "gdp_growth"
    GDP_CONTRACTION = "gdp_contraction"
    YIELD_CURVE_INVERSION = "yield_curve_inversion"
    BOND_VOLATILITY = "bond_volatility"
    DOLLAR_STRENGTH = "dollar_strength"
    DOLLAR_WEAKNESS = "dollar_weakness"
    CONSUMER_SPENDING_UP = "consumer_spending_up"
    CONSUMER_SPENDING_DOWN = "consumer_spending_down"
    CONSUMER_CONFIDENCE_UP = "consumer_confidence_up"
    CONSUMER_CONFIDENCE_DOWN = "consumer_confidence_down"
    HOUSING_DATA_STRONG = "housing_data_strong"
    HOUSING_DATA_WEAK = "housing_data_weak"
    RETAIL_SALES_UP = "retail_sales_up"
    RETAIL_SALES_DOWN = "retail_sales_down"
    PMI_EXPANSION = "pmi_expansion"
    PMI_CONTRACTION = "pmi_contraction"
    RECESSION_FEAR = "recession_fear"
    RECESSION_CONFIRMED = "recession_confirmed"
    BANKING_CRISIS = "banking_crisis"
    DEBT_CEILING_CRISIS = "debt_ceiling_crisis"
    
    # =========================================================================
    # D. GEOPOLITICAL & SENTIMENT
    # =========================================================================
    WAR_CONFLICT = "war_conflict"
    WAR_ESCALATION = "war_escalation"
    WAR_DEESCALATION = "war_deescalation"
    SANCTIONS_IMPOSED = "sanctions_imposed"
    TRADE_WAR = "trade_war"
    TRADE_DEAL = "trade_deal"
    OIL_PRICE_SPIKE = "oil_price_spike"
    OIL_PRICE_CRASH = "oil_price_crash"
    ELECTION_RESULT = "election_result"
    POLITICAL_CHANGE = "political_change"
    GOVERNMENT_STIMULUS = "government_stimulus"
    GOVERNMENT_AUSTERITY = "government_austerity"
    PANDEMIC_OUTBREAK = "pandemic_outbreak"
    PANDEMIC_RECOVERY = "pandemic_recovery"
    NATURAL_DISASTER = "natural_disaster"
    HEDGE_FUND_REBALANCE = "hedge_fund_rebalance"
    INSTITUTIONAL_FLOWS = "institutional_flows"
    ALGO_TRADING_GLITCH = "algo_trading_glitch"
    FLASH_CRASH = "flash_crash"
    SHORT_SQUEEZE = "short_squeeze"
    
    # =========================================================================
    # E. COMMODITY & CURRENCY
    # =========================================================================
    GOLD_PRICE_UP = "gold_price_up"
    GOLD_PRICE_DOWN = "gold_price_down"
    COPPER_PRICE_UP = "copper_price_up"
    COPPER_PRICE_DOWN = "copper_price_down"
    COMMODITY_SURGE = "commodity_surge"
    COMMODITY_CRASH = "commodity_crash"
    CURRENCY_CRISIS = "currency_crisis"
    FOREX_VOLATILITY = "forex_volatility"
    
    # =========================================================================
    # F. CRYPTO-SPECIFIC - BITCOIN & MARKET STRUCTURE
    # =========================================================================
    BTC_PRICE_SURGE = "btc_price_surge"
    BTC_PRICE_CRASH = "btc_price_crash"
    HALVING = "halving"
    BTC_DOMINANCE_UP = "btc_dominance_up"
    BTC_DOMINANCE_DOWN = "btc_dominance_down"
    WHALE_BUY = "whale_buy"
    WHALE_SELL = "whale_sell"
    MINER_CAPITULATION = "miner_capitulation"
    MINER_ACCUMULATION = "miner_accumulation"
    HASH_RATE_UP = "hash_rate_up"
    HASH_RATE_DOWN = "hash_rate_down"
    NETWORK_CONGESTION = "network_congestion"
    TX_FEES_SPIKE = "tx_fees_spike"
    ETF_INFLOW = "etf_inflow"
    ETF_OUTFLOW = "etf_outflow"
    EXCHANGE_RESERVE_UP = "exchange_reserve_up"
    EXCHANGE_RESERVE_DOWN = "exchange_reserve_down"
    STABLECOIN_SUPPLY_UP = "stablecoin_supply_up"
    STABLECOIN_SUPPLY_DOWN = "stablecoin_supply_down"
    
    # =========================================================================
    # G. CRYPTO-SPECIFIC - REGULATION & NEWS
    # =========================================================================
    SEC_ACTION = "sec_action"
    SEC_APPROVAL = "sec_approval"
    ETF_APPROVED = "etf_approved"
    ETF_REJECTED = "etf_rejected"
    COUNTRY_BAN = "country_ban"
    COUNTRY_ADOPTION = "country_adoption"
    CRYPTO_TAX_CHANGE = "crypto_tax_change"
    AML_KYC_CRACKDOWN = "aml_kyc_crackdown"
    COURT_RULING_POSITIVE = "court_ruling_positive"
    COURT_RULING_NEGATIVE = "court_ruling_negative"
    STABLECOIN_REGULATION = "stablecoin_regulation"
    CBDC_ANNOUNCEMENT = "cbdc_announcement"
    INSTITUTIONAL_ADOPTION = "institutional_adoption"
    CORPORATE_ADOPTION = "corporate_adoption"
    
    # =========================================================================
    # H. CRYPTO-SPECIFIC - PROJECT/ALTCOIN
    # =========================================================================
    MAINNET_LAUNCH = "mainnet_launch"
    NETWORK_UPGRADE = "network_upgrade"
    HARD_FORK = "hard_fork"
    TOKEN_BURN = "token_burn"
    TOKEN_UNLOCK = "token_unlock"
    VESTING_RELEASE = "vesting_release"
    PARTNERSHIP_ANNOUNCED = "partnership_announced"
    TVL_SURGE = "tvl_surge"
    TVL_DECLINE = "tvl_decline"
    EXCHANGE_HACK = "exchange_hack"
    PROTOCOL_EXPLOIT = "protocol_exploit"
    SMART_CONTRACT_BUG = "smart_contract_bug"
    BRIDGE_HACK = "bridge_hack"
    FLASH_LOAN_ATTACK = "flash_loan_attack"
    ORACLE_FAILURE = "oracle_failure"
    DAO_VOTE = "dao_vote"
    DEV_ACTIVITY_UP = "dev_activity_up"
    DEV_ACTIVITY_DOWN = "dev_activity_down"
    ROADMAP_DELAY = "roadmap_delay"
    ROADMAP_ACCELERATION = "roadmap_acceleration"
    EXCHANGE_LISTING = "exchange_listing"
    EXCHANGE_DELISTING = "exchange_delisting"
    AIRDROP_ANNOUNCED = "airdrop_announced"
    RUG_PULL = "rug_pull"
    
    # =========================================================================
    # I. SOCIAL & SENTIMENT
    # =========================================================================
    ELON_TWEET = "elon_tweet"
    CELEBRITY_MENTION = "celebrity_mention"
    INFLUENCER_HYPE = "influencer_hype"
    INFLUENCER_FUD = "influencer_fud"
    VIRAL_SOCIAL = "viral_social"
    MEME_TREND = "meme_trend"
    PUMP_AND_DUMP = "pump_and_dump"
    FUD_WAVE = "fud_wave"
    NARRATIVE_SHIFT = "narrative_shift"
    FEAR_GREED_EXTREME = "fear_greed_extreme"
    
    # =========================================================================
    # J. SEASONAL & CYCLICAL
    # =========================================================================
    HOLIDAY_SHOPPING = "holiday_shopping"
    EARNINGS_SEASON = "earnings_season"
    TAX_SEASON = "tax_season"
    QUARTER_END_REBALANCE = "quarter_end_rebalance"
    YEAR_END_POSITIONING = "year_end_positioning"
    
    # Unknown fallback
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
        
        # LEARNING FIX: Load refined patterns from PostgreSQL (survive restarts)
        self._load_persisted_patterns()
        
        # LEARNING FIX: Overlay real accuracy from pattern_tracker
        self._overlay_pattern_tracker_accuracy()
    
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
            # =================================================================
            # A. COMPANY-SPECIFIC (STOCKS)
            # =================================================================
            
            EventType.EARNINGS_BEAT.value: EventPattern(
                event_type=EventType.EARNINGS_BEAT.value,
                keywords=["earnings beat", "eps beat", "beat estimates", "exceeded expectations"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=500,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Earnings beats typically pump 5-10%. Watch for 'sell the news' if already priced in."
            ),
            
            EventType.EARNINGS_MISS.value: EventPattern(
                event_type=EventType.EARNINGS_MISS.value,
                keywords=["earnings miss", "eps miss", "missed estimates", "below expectations"],
                affected_symbols=[],
                immediate_reaction=-8.0,
                peak_reaction=-15.0,
                recovery_time_hours=48,
                typical_direction="dump",
                times_observed=500,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Earnings misses dump hard. Gap down at open. Wait for dust to settle."
            ),
            
            EventType.GUIDANCE_RAISED.value: EventPattern(
                event_type=EventType.GUIDANCE_RAISED.value,
                keywords=["raised guidance", "increased outlook", "raised forecast", "upward revision"],
                affected_symbols=[],
                immediate_reaction=7.0,
                peak_reaction=12.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.78,
                notes="Raised guidance is very bullish. Shows management confidence."
            ),
            
            EventType.GUIDANCE_LOWERED.value: EventPattern(
                event_type=EventType.GUIDANCE_LOWERED.value,
                keywords=["lowered guidance", "cut outlook", "reduced forecast", "downward revision"],
                affected_symbols=[],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=72,
                typical_direction="dump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="Lowered guidance is a red flag. Usually indicates deeper problems."
            ),
            
            EventType.MERGER_ANNOUNCED.value: EventPattern(
                event_type=EventType.MERGER_ANNOUNCED.value,
                keywords=["merger", "merge with", "combination", "merger agreement"],
                affected_symbols=[],
                immediate_reaction=15.0,
                peak_reaction=25.0,
                recovery_time_hours=0,
                typical_direction="target_pumps",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="Target company pumps to acquisition price. Acquirer may dip short term."
            ),
            
            EventType.ACQUISITION_ANNOUNCED.value: EventPattern(
                event_type=EventType.ACQUISITION_ANNOUNCED.value,
                keywords=["acquired", "acquisition", "buyout", "takeover", "tender offer"],
                affected_symbols=[],
                immediate_reaction=20.0,
                peak_reaction=30.0,
                recovery_time_hours=0,
                typical_direction="target_pumps",
                times_observed=150,
                last_observed="2026-01-01",
                accuracy=0.90,
                notes="Acquisition targets gap up to offer price. Usually 20-40% premium."
            ),
            
            EventType.STOCK_BUYBACK.value: EventPattern(
                event_type=EventType.STOCK_BUYBACK.value,
                keywords=["buyback", "share repurchase", "stock repurchase"],
                affected_symbols=[],
                immediate_reaction=3.0,
                peak_reaction=8.0,
                recovery_time_hours=0,
                typical_direction="gradual_pump",
                times_observed=300,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Buybacks are long-term bullish. Shows company believes stock is undervalued."
            ),
            
            EventType.DIVIDEND_INCREASE.value: EventPattern(
                event_type=EventType.DIVIDEND_INCREASE.value,
                keywords=["dividend increase", "raised dividend", "dividend hike"],
                affected_symbols=[],
                immediate_reaction=2.0,
                peak_reaction=5.0,
                recovery_time_hours=0,
                typical_direction="mild_pump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.65,
                notes="Dividend increases signal financial health. Attracts income investors."
            ),
            
            EventType.DIVIDEND_CUT.value: EventPattern(
                event_type=EventType.DIVIDEND_CUT.value,
                keywords=["dividend cut", "suspended dividend", "eliminated dividend"],
                affected_symbols=[],
                immediate_reaction=-12.0,
                peak_reaction=-20.0,
                recovery_time_hours=168,
                typical_direction="dump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="Dividend cuts signal financial distress. Major red flag."
            ),
            
            EventType.INSIDER_BUYING.value: EventPattern(
                event_type=EventType.INSIDER_BUYING.value,
                keywords=["insider buying", "ceo bought", "director purchased", "insider purchase"],
                affected_symbols=[],
                immediate_reaction=3.0,
                peak_reaction=8.0,
                recovery_time_hours=0,
                typical_direction="gradual_pump",
                times_observed=150,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Insiders know the company best. Buying is bullish signal."
            ),
            
            EventType.INSIDER_SELLING.value: EventPattern(
                event_type=EventType.INSIDER_SELLING.value,
                keywords=["insider selling", "ceo sold", "director sold", "insider sale"],
                affected_symbols=[],
                immediate_reaction=-2.0,
                peak_reaction=-5.0,
                recovery_time_hours=24,
                typical_direction="mild_dump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.55,
                notes="Insider selling is often planned/diversification. Not always bearish."
            ),
            
            EventType.CEO_CHANGE.value: EventPattern(
                event_type=EventType.CEO_CHANGE.value,
                keywords=["ceo resigned", "ceo appointed", "new ceo", "ceo steps down"],
                affected_symbols=[],
                immediate_reaction=-5.0,
                peak_reaction=-10.0,
                recovery_time_hours=72,
                typical_direction="volatile",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.60,
                notes="CEO changes create uncertainty. Market usually dips initially."
            ),
            
            EventType.EXECUTIVE_SCANDAL.value: EventPattern(
                event_type=EventType.EXECUTIVE_SCANDAL.value,
                keywords=["scandal", "misconduct", "investigation", "fraud allegation"],
                affected_symbols=[],
                immediate_reaction=-15.0,
                peak_reaction=-30.0,
                recovery_time_hours=168,
                typical_direction="dump",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="Scandals destroy trust. Stay away until dust settles."
            ),
            
            EventType.CORPORATE_FRAUD.value: EventPattern(
                event_type=EventType.CORPORATE_FRAUD.value,
                keywords=["fraud", "accounting fraud", "sec investigation", "financial fraud"],
                affected_symbols=[],
                immediate_reaction=-25.0,
                peak_reaction=-50.0,
                recovery_time_hours=720,
                typical_direction="dump_hard",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.95,
                notes="Fraud = potential delisting. Avoid completely. Think Enron, Wirecard."
            ),
            
            EventType.LAWSUIT_FILED.value: EventPattern(
                event_type=EventType.LAWSUIT_FILED.value,
                keywords=["lawsuit filed", "sued", "legal action", "class action"],
                affected_symbols=[],
                immediate_reaction=-5.0,
                peak_reaction=-10.0,
                recovery_time_hours=48,
                typical_direction="mild_dump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.65,
                notes="Lawsuits create uncertainty. Impact depends on merit and size."
            ),
            
            EventType.LAWSUIT_SETTLED.value: EventPattern(
                event_type=EventType.LAWSUIT_SETTLED.value,
                keywords=["settlement", "lawsuit settled", "legal settlement", "agreed to pay"],
                affected_symbols=[],
                immediate_reaction=3.0,
                peak_reaction=5.0,
                recovery_time_hours=0,
                typical_direction="relief_pump",
                times_observed=150,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Settlements remove uncertainty. Usually positive once amount known."
            ),
            
            EventType.PRODUCT_LAUNCH.value: EventPattern(
                event_type=EventType.PRODUCT_LAUNCH.value,
                keywords=["product launch", "new product", "release", "unveiled", "announced"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=48,
                typical_direction="pump_then_fade",
                times_observed=300,
                last_observed="2026-01-01",
                accuracy=0.65,
                notes="Product launches often 'sell the news'. Pump before, fade after."
            ),
            
            EventType.PRODUCT_RECALL.value: EventPattern(
                event_type=EventType.PRODUCT_RECALL.value,
                keywords=["recall", "product recall", "safety recall", "defect"],
                affected_symbols=[],
                immediate_reaction=-8.0,
                peak_reaction=-15.0,
                recovery_time_hours=72,
                typical_direction="dump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Recalls hurt brand and cost money. Short term bearish."
            ),
            
            EventType.PATENT_APPROVED.value: EventPattern(
                event_type=EventType.PATENT_APPROVED.value,
                keywords=["patent approved", "patent granted", "fda approval", "regulatory approval"],
                affected_symbols=[],
                immediate_reaction=10.0,
                peak_reaction=25.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=150,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Patent/FDA approvals are major catalysts. Especially for biotech."
            ),
            
            EventType.BANKRUPTCY.value: EventPattern(
                event_type=EventType.BANKRUPTCY.value,
                keywords=["bankruptcy", "chapter 11", "chapter 7", "filed for bankruptcy"],
                affected_symbols=[],
                immediate_reaction=-50.0,
                peak_reaction=-90.0,
                recovery_time_hours=0,
                typical_direction="dump_to_zero",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.95,
                notes="Bankruptcy usually means equity is worthless. Avoid."
            ),
            
            EventType.ANALYST_UPGRADE.value: EventPattern(
                event_type=EventType.ANALYST_UPGRADE.value,
                keywords=["upgraded", "upgrade", "raised price target", "buy rating"],
                affected_symbols=[],
                immediate_reaction=3.0,
                peak_reaction=6.0,
                recovery_time_hours=24,
                typical_direction="mild_pump",
                times_observed=500,
                last_observed="2026-01-01",
                accuracy=0.60,
                notes="Analyst upgrades have modest impact. Often lagging indicators."
            ),
            
            EventType.ANALYST_DOWNGRADE.value: EventPattern(
                event_type=EventType.ANALYST_DOWNGRADE.value,
                keywords=["downgraded", "downgrade", "lowered price target", "sell rating"],
                affected_symbols=[],
                immediate_reaction=-4.0,
                peak_reaction=-8.0,
                recovery_time_hours=24,
                typical_direction="mild_dump",
                times_observed=500,
                last_observed="2026-01-01",
                accuracy=0.60,
                notes="Analyst downgrades have modest impact. Often lagging indicators."
            ),
            
            EventType.STOCK_SPLIT.value: EventPattern(
                event_type=EventType.STOCK_SPLIT.value,
                keywords=["stock split", "share split", "split announced"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=168,
                typical_direction="pump_into_split",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Splits are psychologically bullish. Usually pump into effective date."
            ),
            
            # =================================================================
            # B. SECTOR & INDUSTRY
            # =================================================================
            
            EventType.SECTOR_ROTATION.value: EventPattern(
                event_type=EventType.SECTOR_ROTATION.value,
                keywords=["sector rotation", "rotating into", "rotating out of", "flows into"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=15.0,
                recovery_time_hours=168,
                typical_direction="beneficiary_pumps",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Money flows from one sector to another. Follow the flows."
            ),
            
            EventType.TARIFF_CHANGE.value: EventPattern(
                event_type=EventType.TARIFF_CHANGE.value,
                keywords=["tariff", "tariffs", "trade war", "import duty"],
                affected_symbols=[],
                immediate_reaction=-5.0,
                peak_reaction=-10.0,
                recovery_time_hours=72,
                typical_direction="affected_dumps",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Tariffs hurt importers and exporters. Create uncertainty."
            ),
            
            EventType.LABOR_STRIKE.value: EventPattern(
                event_type=EventType.LABOR_STRIKE.value,
                keywords=["strike", "labor strike", "union strike", "walkout"],
                affected_symbols=[],
                immediate_reaction=-5.0,
                peak_reaction=-10.0,
                recovery_time_hours=168,
                typical_direction="dump",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Strikes disrupt production and cost money. Short term bearish."
            ),
            
            # =================================================================
            # C. MACRO & ECONOMIC
            # =================================================================
            
            EventType.FED_RATE_HIKE.value: EventPattern(
                event_type=EventType.FED_RATE_HIKE.value,
                keywords=["rate hike", "raised rates", "rate increase", "hawkish fed"],
                affected_symbols=["SPY", "QQQ", "BTC", "ETH"],
                immediate_reaction=-3.0,
                peak_reaction=-8.0,
                recovery_time_hours=72,
                typical_direction="risk_off",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Rate hikes are bearish for growth stocks and crypto. Risk-off."
            ),
            
            EventType.FED_RATE_CUT.value: EventPattern(
                event_type=EventType.FED_RATE_CUT.value,
                keywords=["rate cut", "lowered rates", "rate decrease", "dovish fed"],
                affected_symbols=["SPY", "QQQ", "BTC", "ETH"],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=0,
                typical_direction="risk_on",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Rate cuts are bullish. Cheap money flows into risk assets."
            ),
            
            EventType.FED_RATE_DECISION.value: EventPattern(
                event_type=EventType.FED_RATE_DECISION.value,
                keywords=["fed", "federal reserve", "rate", "fomc", "powell", "interest rate"],
                affected_symbols=["BTC", "ETH", "SPY", "QQQ"],
                immediate_reaction=-5.0,
                peak_reaction=-10.0,
                recovery_time_hours=72,
                typical_direction="dump_on_hike",
                times_observed=30,
                last_observed="2025-01-01",
                accuracy=0.75,
                notes="Rate hikes = risk-off = crypto dumps. Rate cuts = pump."
            ),
            
            EventType.INFLATION_HIGH.value: EventPattern(
                event_type=EventType.INFLATION_HIGH.value,
                keywords=["inflation high", "cpi higher", "inflation rose", "hot inflation"],
                affected_symbols=["SPY", "QQQ", "BTC", "GOLD"],
                immediate_reaction=-3.0,
                peak_reaction=-7.0,
                recovery_time_hours=48,
                typical_direction="dump",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="High inflation = more rate hikes expected. Bearish for risk assets."
            ),
            
            EventType.INFLATION_LOW.value: EventPattern(
                event_type=EventType.INFLATION_LOW.value,
                keywords=["inflation low", "cpi lower", "inflation fell", "cool inflation"],
                affected_symbols=["SPY", "QQQ", "BTC"],
                immediate_reaction=3.0,
                peak_reaction=7.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=20,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Low inflation = dovish Fed expected. Bullish for risk assets."
            ),
            
            EventType.JOBS_REPORT_STRONG.value: EventPattern(
                event_type=EventType.JOBS_REPORT_STRONG.value,
                keywords=["jobs beat", "nfp strong", "unemployment fell", "payrolls strong"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=2.0,
                peak_reaction=5.0,
                recovery_time_hours=24,
                typical_direction="mixed",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.55,
                notes="Strong jobs = good economy but also = hawkish Fed. Mixed signal."
            ),
            
            EventType.JOBS_REPORT_WEAK.value: EventPattern(
                event_type=EventType.JOBS_REPORT_WEAK.value,
                keywords=["jobs miss", "nfp weak", "unemployment rose", "payrolls weak"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=-2.0,
                peak_reaction=-5.0,
                recovery_time_hours=24,
                typical_direction="mixed",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.55,
                notes="Weak jobs = bad economy but also = dovish Fed. Mixed signal."
            ),
            
            EventType.GDP_GROWTH.value: EventPattern(
                event_type=EventType.GDP_GROWTH.value,
                keywords=["gdp growth", "gdp beat", "economy grew", "gdp higher"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=2.0,
                peak_reaction=5.0,
                recovery_time_hours=0,
                typical_direction="mild_pump",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.65,
                notes="GDP growth is bullish for equities. Shows healthy economy."
            ),
            
            EventType.GDP_CONTRACTION.value: EventPattern(
                event_type=EventType.GDP_CONTRACTION.value,
                keywords=["gdp contraction", "gdp miss", "economy shrank", "negative gdp"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=-3.0,
                peak_reaction=-8.0,
                recovery_time_hours=72,
                typical_direction="dump",
                times_observed=20,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="GDP contraction raises recession fears. Bearish."
            ),
            
            EventType.RECESSION_FEAR.value: EventPattern(
                event_type=EventType.RECESSION_FEAR.value,
                keywords=["recession", "recession fears", "economic slowdown", "hard landing"],
                affected_symbols=["SPY", "QQQ", "BTC"],
                immediate_reaction=-5.0,
                peak_reaction=-15.0,
                recovery_time_hours=168,
                typical_direction="risk_off",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Recession fears cause flight to safety. Risk assets dump."
            ),
            
            EventType.BANKING_CRISIS.value: EventPattern(
                event_type=EventType.BANKING_CRISIS.value,
                keywords=["bank failure", "bank run", "banking crisis", "bank collapse"],
                affected_symbols=["BTC", "GOLD", "XLF"],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=168,
                typical_direction="risk_off_btc_pump",
                times_observed=10,
                last_observed="2023-03-01",
                accuracy=0.80,
                notes="Banking crises paradoxically can pump BTC as safe haven. SVB effect."
            ),
            
            # =================================================================
            # D. GEOPOLITICAL & SENTIMENT
            # =================================================================
            
            EventType.WAR_CONFLICT.value: EventPattern(
                event_type=EventType.WAR_CONFLICT.value,
                keywords=["war", "invasion", "attack", "military", "conflict", "missile"],
                affected_symbols=["BTC", "GOLD", "OIL"],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=168,
                typical_direction="risk_off",
                times_observed=10,
                last_observed="2024-01-01",
                accuracy=0.75,
                notes="War = uncertainty = risk-off. But crypto can be safe haven long term."
            ),
            
            EventType.WAR_ESCALATION.value: EventPattern(
                event_type=EventType.WAR_ESCALATION.value,
                keywords=["escalation", "troops deployed", "bombing", "nuclear"],
                affected_symbols=["BTC", "GOLD", "OIL", "SPY"],
                immediate_reaction=-8.0,
                peak_reaction=-15.0,
                recovery_time_hours=72,
                typical_direction="risk_off",
                times_observed=20,
                last_observed="2024-01-01",
                accuracy=0.80,
                notes="Escalation increases uncertainty. Risk assets dump."
            ),
            
            EventType.WAR_DEESCALATION.value: EventPattern(
                event_type=EventType.WAR_DEESCALATION.value,
                keywords=["ceasefire", "peace talks", "de-escalation", "truce"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=0,
                typical_direction="relief_rally",
                times_observed=15,
                last_observed="2024-01-01",
                accuracy=0.75,
                notes="Peace = certainty = risk-on. Markets rally on de-escalation."
            ),
            
            EventType.OIL_PRICE_SPIKE.value: EventPattern(
                event_type=EventType.OIL_PRICE_SPIKE.value,
                keywords=["oil spike", "oil surge", "crude jumped", "opec cut"],
                affected_symbols=["XLE", "XOM", "CVX", "OIL"],
                immediate_reaction=-2.0,
                peak_reaction=-5.0,
                recovery_time_hours=48,
                typical_direction="energy_up_market_down",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Oil spikes hurt consumers and most businesses. Energy stocks benefit."
            ),
            
            EventType.ELECTION_RESULT.value: EventPattern(
                event_type=EventType.ELECTION_RESULT.value,
                keywords=["election", "elected", "won election", "election result"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=3.0,
                peak_reaction=10.0,
                recovery_time_hours=48,
                typical_direction="volatile_then_up",
                times_observed=20,
                last_observed="2024-11-01",
                accuracy=0.70,
                notes="Elections create volatility. Markets usually rally once uncertainty removed."
            ),
            
            EventType.GOVERNMENT_STIMULUS.value: EventPattern(
                event_type=EventType.GOVERNMENT_STIMULUS.value,
                keywords=["stimulus", "stimulus package", "relief bill", "government spending"],
                affected_symbols=["SPY", "QQQ", "BTC"],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=20,
                last_observed="2021-01-01",
                accuracy=0.85,
                notes="Free money = asset inflation. Very bullish for risk assets."
            ),
            
            EventType.PANDEMIC_OUTBREAK.value: EventPattern(
                event_type=EventType.PANDEMIC_OUTBREAK.value,
                keywords=["pandemic", "outbreak", "virus", "covid", "health emergency"],
                affected_symbols=["SPY", "QQQ", "XLV"],
                immediate_reaction=-15.0,
                peak_reaction=-35.0,
                recovery_time_hours=720,
                typical_direction="crash_then_v_recovery",
                times_observed=5,
                last_observed="2020-03-01",
                accuracy=0.90,
                notes="Pandemics crash markets hard. But stimulus follows. Buy the crash."
            ),
            
            EventType.NATURAL_DISASTER.value: EventPattern(
                event_type=EventType.NATURAL_DISASTER.value,
                keywords=["hurricane", "earthquake", "tsunami", "wildfire", "flood"],
                affected_symbols=[],
                immediate_reaction=-3.0,
                peak_reaction=-8.0,
                recovery_time_hours=72,
                typical_direction="localized_impact",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.65,
                notes="Disasters have localized impact. Affects specific companies/regions."
            ),
            
            EventType.SHORT_SQUEEZE.value: EventPattern(
                event_type=EventType.SHORT_SQUEEZE.value,
                keywords=["short squeeze", "squeeze", "shorts covering", "gme", "meme stock"],
                affected_symbols=[],
                immediate_reaction=50.0,
                peak_reaction=200.0,
                recovery_time_hours=72,
                typical_direction="violent_pump_then_crash",
                times_observed=20,
                last_observed="2024-01-01",
                accuracy=0.85,
                notes="Squeezes are violent. Don't chase. They always crash back."
            ),
            
            EventType.FLASH_CRASH.value: EventPattern(
                event_type=EventType.FLASH_CRASH.value,
                keywords=["flash crash", "circuit breaker", "halted", "plunge"],
                affected_symbols=[],
                immediate_reaction=-15.0,
                peak_reaction=-25.0,
                recovery_time_hours=4,
                typical_direction="crash_then_recovery",
                times_observed=15,
                last_observed="2024-01-01",
                accuracy=0.80,
                notes="Flash crashes are buy opportunities. Usually recover same day."
            ),
            
            # =================================================================
            # E. COMMODITY & CURRENCY
            # =================================================================
            
            EventType.GOLD_PRICE_UP.value: EventPattern(
                event_type=EventType.GOLD_PRICE_UP.value,
                keywords=["gold up", "gold rally", "gold surge", "gold breakout"],
                affected_symbols=["GOLD", "GLD", "GDX", "BTC"],
                immediate_reaction=2.0,
                peak_reaction=5.0,
                recovery_time_hours=0,
                typical_direction="safe_haven_bid",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.65,
                notes="Gold up = risk-off sentiment. BTC sometimes follows as digital gold."
            ),
            
            EventType.DOLLAR_STRENGTH.value: EventPattern(
                event_type=EventType.DOLLAR_STRENGTH.value,
                keywords=["dollar strong", "dxy up", "dollar rally", "usd strength"],
                affected_symbols=["BTC", "ETH", "GOLD"],
                immediate_reaction=-3.0,
                peak_reaction=-7.0,
                recovery_time_hours=72,
                typical_direction="inverse_to_risk",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Strong dollar is bearish for BTC and commodities. Inverse correlation."
            ),
            
            EventType.DOLLAR_WEAKNESS.value: EventPattern(
                event_type=EventType.DOLLAR_WEAKNESS.value,
                keywords=["dollar weak", "dxy down", "dollar fell", "usd weakness"],
                affected_symbols=["BTC", "ETH", "GOLD"],
                immediate_reaction=3.0,
                peak_reaction=7.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Weak dollar is bullish for BTC and commodities. Inverse correlation."
            ),
            
            # =================================================================
            # F. CRYPTO-SPECIFIC - BITCOIN & MARKET STRUCTURE
            # =================================================================
            
            EventType.BTC_PRICE_SURGE.value: EventPattern(
                event_type=EventType.BTC_PRICE_SURGE.value,
                keywords=["bitcoin surge", "btc pump", "bitcoin rally", "bitcoin breakout"],
                affected_symbols=["ETH", "SOL", "ALTS"],
                immediate_reaction=10.0,
                peak_reaction=20.0,
                recovery_time_hours=48,
                typical_direction="alts_follow",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="BTC pumps = altcoins pump harder. Risk-on across crypto."
            ),
            
            EventType.BTC_PRICE_CRASH.value: EventPattern(
                event_type=EventType.BTC_PRICE_CRASH.value,
                keywords=["bitcoin crash", "btc dump", "bitcoin plunge", "bitcoin fell"],
                affected_symbols=["ETH", "SOL", "ALTS"],
                immediate_reaction=-15.0,
                peak_reaction=-30.0,
                recovery_time_hours=72,
                typical_direction="alts_follow_harder",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="BTC crashes = altcoins crash harder. Risk-off across crypto."
            ),
            
            EventType.HALVING.value: EventPattern(
                event_type=EventType.HALVING.value,
                keywords=["halving", "halvening", "block reward", "supply reduction"],
                affected_symbols=["BTC", "ETH", "altcoins"],
                immediate_reaction=5.0,
                peak_reaction=100.0,
                recovery_time_hours=0,
                typical_direction="long_term_pump",
                times_observed=4,
                last_observed="2024-04-01",
                accuracy=1.0,
                notes="Every halving has led to new ATH within 12-18 months. HODL."
            ),
            
            EventType.BTC_DOMINANCE_UP.value: EventPattern(
                event_type=EventType.BTC_DOMINANCE_UP.value,
                keywords=["btc dominance up", "dominance rising", "btc.d up"],
                affected_symbols=["ETH", "SOL", "ALTS"],
                immediate_reaction=-5.0,
                peak_reaction=-15.0,
                recovery_time_hours=168,
                typical_direction="alts_underperform",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Rising BTC dominance = money flowing from alts to BTC. Alts dump."
            ),
            
            EventType.BTC_DOMINANCE_DOWN.value: EventPattern(
                event_type=EventType.BTC_DOMINANCE_DOWN.value,
                keywords=["btc dominance down", "dominance falling", "altseason"],
                affected_symbols=["ETH", "SOL", "ALTS"],
                immediate_reaction=10.0,
                peak_reaction=30.0,
                recovery_time_hours=0,
                typical_direction="altseason",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Falling BTC dominance = altseason. Alts outperform."
            ),
            
            EventType.WHALE_BUY.value: EventPattern(
                event_type=EventType.WHALE_BUY.value,
                keywords=["whale bought", "large purchase", "whale accumulation", "whale buying"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=10.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Whale buys are bullish signal. Smart money accumulating."
            ),
            
            EventType.WHALE_SELL.value: EventPattern(
                event_type=EventType.WHALE_SELL.value,
                keywords=["whale", "large transfer", "moved to exchange", "dump", "whale sell"],
                affected_symbols=[],
                immediate_reaction=-8.0,
                peak_reaction=-15.0,
                recovery_time_hours=12,
                typical_direction="cascade_then_recovery",
                times_observed=100,
                last_observed="2025-01-01",
                accuracy=0.70,
                notes="Whale sells trigger stop losses and cascade selling. Usually oversold."
            ),
            
            EventType.MINER_CAPITULATION.value: EventPattern(
                event_type=EventType.MINER_CAPITULATION.value,
                keywords=["miner capitulation", "miners selling", "hash rate drop"],
                affected_symbols=["BTC"],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=168,
                typical_direction="bottom_signal",
                times_observed=10,
                last_observed="2022-12-01",
                accuracy=0.80,
                notes="Miner capitulation often marks bottoms. They sell at lows."
            ),
            
            EventType.ETF_INFLOW.value: EventPattern(
                event_type=EventType.ETF_INFLOW.value,
                keywords=["etf inflow", "bitcoin etf", "institutional buying", "gbtc inflow"],
                affected_symbols=["BTC"],
                immediate_reaction=3.0,
                peak_reaction=8.0,
                recovery_time_hours=0,
                typical_direction="gradual_pump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="ETF inflows are bullish. Institutional demand."
            ),
            
            EventType.ETF_OUTFLOW.value: EventPattern(
                event_type=EventType.ETF_OUTFLOW.value,
                keywords=["etf outflow", "etf redemption", "institutional selling"],
                affected_symbols=["BTC"],
                immediate_reaction=-3.0,
                peak_reaction=-8.0,
                recovery_time_hours=48,
                typical_direction="gradual_dump",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="ETF outflows are bearish. Institutional selling."
            ),
            
            # =================================================================
            # G. CRYPTO-SPECIFIC - REGULATION & NEWS
            # =================================================================
            
            EventType.SEC_ACTION.value: EventPattern(
                event_type=EventType.SEC_ACTION.value,
                keywords=["sec", "sec lawsuit", "sec investigation", "securities violation"],
                affected_symbols=[],
                immediate_reaction=-15.0,
                peak_reaction=-30.0,
                recovery_time_hours=168,
                typical_direction="dump",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="SEC actions are very bearish. Regulatory risk is real."
            ),
            
            EventType.SEC_APPROVAL.value: EventPattern(
                event_type=EventType.SEC_APPROVAL.value,
                keywords=["sec approved", "sec approval", "regulatory approval"],
                affected_symbols=[],
                immediate_reaction=10.0,
                peak_reaction=25.0,
                recovery_time_hours=48,
                typical_direction="pump_then_fade",
                times_observed=20,
                last_observed="2024-01-01",
                accuracy=0.80,
                notes="SEC approvals are major catalysts. Buy rumor sell news often applies."
            ),
            
            EventType.ETF_APPROVED.value: EventPattern(
                event_type=EventType.ETF_APPROVED.value,
                keywords=["etf approved", "spot etf", "bitcoin etf approved"],
                affected_symbols=["BTC", "ETH"],
                immediate_reaction=10.0,
                peak_reaction=30.0,
                recovery_time_hours=72,
                typical_direction="pump_then_consolidate",
                times_observed=5,
                last_observed="2024-01-10",
                accuracy=0.80,
                notes="ETF approvals are major. BTC ETF Jan 2024 was huge catalyst."
            ),
            
            EventType.ETF_REJECTED.value: EventPattern(
                event_type=EventType.ETF_REJECTED.value,
                keywords=["etf rejected", "etf denied", "sec rejected"],
                affected_symbols=["BTC"],
                immediate_reaction=-8.0,
                peak_reaction=-15.0,
                recovery_time_hours=48,
                typical_direction="dump",
                times_observed=20,
                last_observed="2023-01-01",
                accuracy=0.80,
                notes="ETF rejections disappoint. But market usually recovers."
            ),
            
            EventType.COUNTRY_BAN.value: EventPattern(
                event_type=EventType.COUNTRY_BAN.value,
                keywords=["ban", "banned crypto", "prohibited", "illegal"],
                affected_symbols=["BTC", "ETH"],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=72,
                typical_direction="dump_then_recovery",
                times_observed=30,
                last_observed="2021-09-01",
                accuracy=0.75,
                notes="Country bans cause panic. But crypto is global. Usually recovers."
            ),
            
            EventType.COUNTRY_ADOPTION.value: EventPattern(
                event_type=EventType.COUNTRY_ADOPTION.value,
                keywords=["adopted bitcoin", "legal tender", "country adoption"],
                affected_symbols=["BTC"],
                immediate_reaction=8.0,
                peak_reaction=15.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=5,
                last_observed="2021-09-01",
                accuracy=0.80,
                notes="Country adoption is bullish. El Salvador was first."
            ),
            
            EventType.INSTITUTIONAL_ADOPTION.value: EventPattern(
                event_type=EventType.INSTITUTIONAL_ADOPTION.value,
                keywords=["blackrock", "fidelity", "institutional", "pension fund", "hedge fund"],
                affected_symbols=["BTC", "ETH"],
                immediate_reaction=5.0,
                peak_reaction=12.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Institutional adoption = legitimacy. Very bullish long term."
            ),
            
            # =================================================================
            # H. CRYPTO-SPECIFIC - PROJECT/ALTCOIN
            # =================================================================
            
            EventType.MAINNET_LAUNCH.value: EventPattern(
                event_type=EventType.MAINNET_LAUNCH.value,
                keywords=["mainnet launch", "mainnet live", "launched mainnet"],
                affected_symbols=[],
                immediate_reaction=10.0,
                peak_reaction=25.0,
                recovery_time_hours=72,
                typical_direction="pump_then_dump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Mainnet launches are buy rumor sell news. Pump into launch, dump after."
            ),
            
            EventType.NETWORK_UPGRADE.value: EventPattern(
                event_type=EventType.NETWORK_UPGRADE.value,
                keywords=["upgrade", "network upgrade", "hard fork", "v2 launch"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=15.0,
                recovery_time_hours=48,
                typical_direction="pump_into_event",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Upgrades pump into event. Sell the news often applies."
            ),
            
            EventType.TOKEN_BURN.value: EventPattern(
                event_type=EventType.TOKEN_BURN.value,
                keywords=["token burn", "burned", "supply reduction", "deflationary"],
                affected_symbols=[],
                immediate_reaction=5.0,
                peak_reaction=15.0,
                recovery_time_hours=0,
                typical_direction="pump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Burns reduce supply. Basic economics = bullish."
            ),
            
            EventType.TOKEN_UNLOCK.value: EventPattern(
                event_type=EventType.TOKEN_UNLOCK.value,
                keywords=["token unlock", "vesting", "unlock event", "tokens released"],
                affected_symbols=[],
                immediate_reaction=-5.0,
                peak_reaction=-15.0,
                recovery_time_hours=72,
                typical_direction="dump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Unlocks increase supply. Insiders often dump. Bearish."
            ),
            
            EventType.EXCHANGE_HACK.value: EventPattern(
                event_type=EventType.EXCHANGE_HACK.value,
                keywords=["hack", "exploit", "stolen", "breach", "drained"],
                affected_symbols=["BTC", "ETH"],
                immediate_reaction=-15.0,
                peak_reaction=-25.0,
                recovery_time_hours=48,
                typical_direction="flash_crash_recovery",
                times_observed=20,
                last_observed="2025-01-01",
                accuracy=0.80,
                notes="Hacks cause panic selling, but market usually recovers. Buy the dip opportunity."
            ),
            
            EventType.PROTOCOL_EXPLOIT.value: EventPattern(
                event_type=EventType.PROTOCOL_EXPLOIT.value,
                keywords=["exploit", "vulnerability", "defi hack", "protocol drained"],
                affected_symbols=[],
                immediate_reaction=-30.0,
                peak_reaction=-60.0,
                recovery_time_hours=168,
                typical_direction="dump_hard",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.90,
                notes="Protocol exploits destroy trust. Affected token often never recovers."
            ),
            
            EventType.BRIDGE_HACK.value: EventPattern(
                event_type=EventType.BRIDGE_HACK.value,
                keywords=["bridge hack", "bridge exploit", "cross-chain hack"],
                affected_symbols=[],
                immediate_reaction=-20.0,
                peak_reaction=-40.0,
                recovery_time_hours=168,
                typical_direction="dump",
                times_observed=20,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="Bridge hacks are devastating. Ronin, Wormhole, etc. Major red flag."
            ),
            
            EventType.EXCHANGE_LISTING.value: EventPattern(
                event_type=EventType.EXCHANGE_LISTING.value,
                keywords=["listed on", "listing", "binance listing", "coinbase listing"],
                affected_symbols=[],
                immediate_reaction=25.0,
                peak_reaction=50.0,
                recovery_time_hours=48,
                typical_direction="pump_then_dump",
                times_observed=200,
                last_observed="2025-01-01",
                accuracy=0.90,
                notes="Buy the rumor, sell the news. Listings pump hard then dump."
            ),
            
            EventType.EXCHANGE_DELISTING.value: EventPattern(
                event_type=EventType.EXCHANGE_DELISTING.value,
                keywords=["delisted", "delisting", "removed from", "trading suspended"],
                affected_symbols=[],
                immediate_reaction=-25.0,
                peak_reaction=-50.0,
                recovery_time_hours=168,
                typical_direction="dump_hard",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.90,
                notes="Delistings are death sentences. Liquidity disappears. Avoid."
            ),
            
            EventType.AIRDROP_ANNOUNCED.value: EventPattern(
                event_type=EventType.AIRDROP_ANNOUNCED.value,
                keywords=["airdrop", "token distribution", "free tokens", "airdrop announced"],
                affected_symbols=[],
                immediate_reaction=10.0,
                peak_reaction=30.0,
                recovery_time_hours=72,
                typical_direction="pump_then_dump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Airdrops pump anticipation. Often dump after distribution."
            ),
            
            EventType.RUG_PULL.value: EventPattern(
                event_type=EventType.RUG_PULL.value,
                keywords=["rug pull", "rugged", "scam", "exit scam", "team disappeared"],
                affected_symbols=[],
                immediate_reaction=-90.0,
                peak_reaction=-99.0,
                recovery_time_hours=0,
                typical_direction="death",
                times_observed=500,
                last_observed="2026-01-01",
                accuracy=0.99,
                notes="Rug pulls = total loss. No recovery. DYOR always."
            ),
            
            # =================================================================
            # I. SOCIAL & SENTIMENT
            # =================================================================
            
            EventType.ELON_TWEET.value: EventPattern(
                event_type=EventType.ELON_TWEET.value,
                keywords=["elon", "musk", "tesla", "doge", "dogecoin"],
                affected_symbols=["DOGE", "SHIB", "TSLA", "BTC"],
                immediate_reaction=15.0,
                peak_reaction=30.0,
                recovery_time_hours=24,
                typical_direction="pump_then_dump",
                times_observed=50,
                last_observed="2025-01-01",
                accuracy=0.85,
                notes="Elon tweets cause immediate pump, followed by dump within 24h. Don't chase the pump."
            ),
            
            EventType.CELEBRITY_MENTION.value: EventPattern(
                event_type=EventType.CELEBRITY_MENTION.value,
                keywords=["celebrity", "endorsed", "promoted", "famous"],
                affected_symbols=[],
                immediate_reaction=10.0,
                peak_reaction=30.0,
                recovery_time_hours=24,
                typical_direction="pump_then_dump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Celebrity mentions pump short term. Usually dump after. Don't chase."
            ),
            
            EventType.INFLUENCER_HYPE.value: EventPattern(
                event_type=EventType.INFLUENCER_HYPE.value,
                keywords=["influencer", "youtuber", "crypto twitter", "ct", "shilled"],
                affected_symbols=[],
                immediate_reaction=8.0,
                peak_reaction=20.0,
                recovery_time_hours=24,
                typical_direction="pump_then_dump",
                times_observed=200,
                last_observed="2026-01-01",
                accuracy=0.75,
                notes="Influencer pumps are exit liquidity for them. Be careful."
            ),
            
            EventType.VIRAL_SOCIAL.value: EventPattern(
                event_type=EventType.VIRAL_SOCIAL.value,
                keywords=["viral", "trending", "meme", "tiktok"],
                affected_symbols=[],
                immediate_reaction=20.0,
                peak_reaction=50.0,
                recovery_time_hours=48,
                typical_direction="pump_then_dump",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.80,
                notes="Viral social pumps are fast and dump fast. Pure speculation."
            ),
            
            EventType.MEME_TREND.value: EventPattern(
                event_type=EventType.MEME_TREND.value,
                keywords=["meme coin", "memecoin", "pepe", "doge", "shib", "meme season"],
                affected_symbols=["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
                immediate_reaction=30.0,
                peak_reaction=100.0,
                recovery_time_hours=72,
                typical_direction="pump_then_crash",
                times_observed=50,
                last_observed="2026-01-01",
                accuracy=0.85,
                notes="Meme seasons are violent. Quick gains, quicker losses. Gamble only."
            ),
            
            EventType.FUD_WAVE.value: EventPattern(
                event_type=EventType.FUD_WAVE.value,
                keywords=["fud", "fear", "uncertainty", "doubt", "panic"],
                affected_symbols=[],
                immediate_reaction=-10.0,
                peak_reaction=-20.0,
                recovery_time_hours=72,
                typical_direction="dump_then_recovery",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="FUD waves cause panic selling. Often best time to buy if fundamentals intact."
            ),
            
            EventType.NARRATIVE_SHIFT.value: EventPattern(
                event_type=EventType.NARRATIVE_SHIFT.value,
                keywords=["narrative", "ai coins", "rwa", "depin", "new meta"],
                affected_symbols=[],
                immediate_reaction=15.0,
                peak_reaction=50.0,
                recovery_time_hours=168,
                typical_direction="new_sector_pumps",
                times_observed=30,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Narrative shifts move whole sectors. AI, RWA, DePIN have all had cycles."
            ),
            
            EventType.FEAR_GREED_EXTREME.value: EventPattern(
                event_type=EventType.FEAR_GREED_EXTREME.value,
                keywords=["extreme fear", "extreme greed", "fear index", "greed index"],
                affected_symbols=["BTC", "ETH"],
                immediate_reaction=0.0,
                peak_reaction=0.0,
                recovery_time_hours=168,
                typical_direction="contrarian_signal",
                times_observed=100,
                last_observed="2026-01-01",
                accuracy=0.70,
                notes="Extreme fear = buy. Extreme greed = sell. Contrarian indicator."
            ),
            
            # =================================================================
            # J. SEASONAL & CYCLICAL
            # =================================================================
            
            EventType.HOLIDAY_SHOPPING.value: EventPattern(
                event_type=EventType.HOLIDAY_SHOPPING.value,
                keywords=["black friday", "cyber monday", "holiday sales", "christmas sales"],
                affected_symbols=["AMZN", "WMT", "TGT", "SHOP"],
                immediate_reaction=2.0,
                peak_reaction=5.0,
                recovery_time_hours=168,
                typical_direction="retail_pump",
                times_observed=20,
                last_observed="2025-12-01",
                accuracy=0.65,
                notes="Holiday season pumps retailers. Watch for sales data surprises."
            ),
            
            EventType.EARNINGS_SEASON.value: EventPattern(
                event_type=EventType.EARNINGS_SEASON.value,
                keywords=["earnings season", "reporting season", "q1 earnings", "q2 earnings"],
                affected_symbols=["SPY", "QQQ"],
                immediate_reaction=0.0,
                peak_reaction=5.0,
                recovery_time_hours=336,
                typical_direction="volatile",
                times_observed=40,
                last_observed="2026-01-01",
                accuracy=0.60,
                notes="Earnings season = high volatility. Individual stocks swing big."
            ),
            
            EventType.TAX_SEASON.value: EventPattern(
                event_type=EventType.TAX_SEASON.value,
                keywords=["tax season", "tax selling", "tax loss", "april taxes"],
                affected_symbols=["BTC", "ETH", "SPY"],
                immediate_reaction=-3.0,
                peak_reaction=-8.0,
                recovery_time_hours=168,
                typical_direction="mild_dump",
                times_observed=10,
                last_observed="2025-04-01",
                accuracy=0.65,
                notes="Tax season can cause selling pressure. People need cash for taxes."
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
        
        # Persist updated pattern to PostgreSQL so it survives restarts
        self._persist_pattern(event.event_type)
        
        # Generate lesson
        direction = "pumped" if event.reaction_48h > 0 else "dumped"
        event.lesson = (
            f"{event.event_type}: {event.trigger} → {event.primary_symbol} {direction} "
            f"{abs(event.reaction_48h):.1f}% over 48h (peak: {event.peak_reaction:.1f}%)"
        )
    
    def _persist_pattern(self, event_type: str):
        """
        Persist refined pattern to PostgreSQL ghost_event_patterns table.
        
        LEARNING FIX: Previously, updated patterns were in-memory only and lost
        on every restart. Now they survive deploys.
        """
        pattern = self.patterns.get(event_type)
        if not pattern or not self.db_url or 'sqlite' in self.db_url:
            return
        
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO ghost_event_patterns 
                (event_type, keywords, affected_symbols, immediate_reaction, peak_reaction,
                 recovery_time_hours, typical_direction, times_observed, last_observed,
                 accuracy, notes, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (event_type) DO UPDATE SET
                    immediate_reaction = EXCLUDED.immediate_reaction,
                    peak_reaction = EXCLUDED.peak_reaction,
                    times_observed = EXCLUDED.times_observed,
                    last_observed = EXCLUDED.last_observed,
                    accuracy = EXCLUDED.accuracy,
                    notes = EXCLUDED.notes,
                    updated_at = NOW()
            """, (
                pattern.event_type,
                json.dumps(pattern.keywords),
                json.dumps(pattern.affected_symbols),
                pattern.immediate_reaction,
                pattern.peak_reaction,
                pattern.recovery_time_hours,
                pattern.typical_direction,
                pattern.times_observed,
                pattern.last_observed,
                pattern.accuracy,
                pattern.notes,
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            LOGGER.debug(f"[EVENT_MEMORY] 💾 Persisted pattern {event_type} to PostgreSQL")
            
        except Exception as e:
            LOGGER.warning(f"[EVENT_MEMORY] Failed to persist pattern {event_type}: {e}")
    
    def _load_persisted_patterns(self):
        """
        Load refined patterns from PostgreSQL and overlay on hardcoded defaults.
        
        LEARNING FIX: Patterns refined through _learn_from_event are now
        persisted in PostgreSQL. On restart, we load them to replace the
        hardcoded defaults with empirically-refined values.
        """
        if not self.db_url or 'sqlite' in self.db_url:
            return
        
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT event_type, immediate_reaction, peak_reaction, times_observed,
                       last_observed, accuracy, notes
                FROM ghost_event_patterns
                WHERE times_observed > 0
            """)
            
            rows = cur.fetchall()
            updated = 0
            
            for event_type, imm_react, peak_react, times_obs, last_obs, accuracy, notes in rows:
                if event_type in self.patterns:
                    pattern = self.patterns[event_type]
                    # Only override if DB has more observations than the hardcoded default
                    if times_obs > pattern.times_observed or pattern.times_observed <= 500:
                        pattern.immediate_reaction = imm_react
                        pattern.peak_reaction = peak_react
                        pattern.times_observed = times_obs
                        pattern.accuracy = accuracy
                        if last_obs:
                            pattern.last_observed = str(last_obs)
                        if notes:
                            pattern.notes = notes
                        updated += 1
            
            cur.close()
            conn.close()
            
            if updated > 0:
                LOGGER.info(f"[EVENT_MEMORY] 🧠 Loaded {updated} refined patterns from PostgreSQL (survive deploys)")
            
        except Exception as e:
            LOGGER.warning(f"[EVENT_MEMORY] Failed to load persisted patterns: {e}")
    
    def _overlay_pattern_tracker_accuracy(self):
        """
        Replace hardcoded pattern accuracy with REAL accuracy from pattern_tracker.
        
        LEARNING FIX: pattern_tracker records every pattern detection and tracks 
        whether it was profitable. This data was stored in PostgreSQL but never
        fed back into event_memory's hardcoded patterns. Now it is.
        
        Example: If pattern_tracker shows ELON_TWEET has 45% real accuracy 
        (not the hardcoded 80%), we use 45%.
        """
        try:
            from core.pattern_tracker import get_pattern_accuracy
            
            real_accuracy = get_pattern_accuracy()
            
            if "error" in real_accuracy:
                return
            
            updated = 0
            for pattern_type, stats in real_accuracy.items():
                if pattern_type == "overall" or pattern_type == "pending_reconciliation":
                    continue
                
                # Find matching pattern in memory
                if pattern_type in self.patterns and stats.get("detections", 0) >= 5:
                    old_accuracy = self.patterns[pattern_type].accuracy
                    empirical_accuracy = stats["accuracy"] / 100.0  # Convert from % to fraction
                    
                    # Blend: weight empirical data more as sample size grows
                    n = stats["detections"]
                    if n >= 20:
                        # Strong sample — use 80% empirical, 20% historical
                        blended = 0.8 * empirical_accuracy + 0.2 * old_accuracy
                    elif n >= 10:
                        # Moderate sample — 50/50
                        blended = 0.5 * empirical_accuracy + 0.5 * old_accuracy
                    else:
                        # Small sample — 30% empirical, 70% historical
                        blended = 0.3 * empirical_accuracy + 0.7 * old_accuracy
                    
                    self.patterns[pattern_type].accuracy = blended
                    
                    if abs(blended - old_accuracy) > 0.05:
                        LOGGER.info(
                            f"[EVENT_MEMORY] 📊 {pattern_type} accuracy updated: "
                            f"{old_accuracy:.0%} → {blended:.0%} "
                            f"(empirical={empirical_accuracy:.0%}, n={n})"
                        )
                        updated += 1
            
            if updated > 0:
                LOGGER.info(f"[EVENT_MEMORY] 🧠 Updated {updated} patterns with real accuracy from pattern_tracker")
                
        except Exception as e:
            LOGGER.warning(f"[EVENT_MEMORY] Pattern tracker overlay failed: {e}")
    
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

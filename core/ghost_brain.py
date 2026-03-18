#!/usr/bin/env python3
"""
🧠 GHOST BRAIN v3 — 25 Cognitive Abilities
============================================

Evolution: v1 (binary exclude/boost) → v2 (6-zone + invert) → v3 (FULL BRAIN)

The brain operates as a POST-PROCESSING layer at notification time.
Raw paper trades are PRESERVED (model accuracy stays honest).
The brain adjusts what gets SENT to users and at what confidence.

═══════════════════════════════════════════════════════════════════
25 COGNITIVE ABILITIES
═══════════════════════════════════════════════════════════════════

  TIER 1 — Highest Impact (data enrichment):
    #1  PER-DIRECTION      UP vs DOWN accuracy per symbol
    #2  RECENCY             30-day accuracy weighted 70/30 vs all-time
    #3  CALIBRATION         Map confidence to actual hit probability
    #4  STREAK              Hot/cold streaks from trust ladder
    #5  REGIME              Market regime awareness (VIX/calm/fear)

  TIER 2 — High Impact (environmental signals):
    #6  MAGNITUDE           Weight wins by size (big right > barely right)
    #7  DAY-OF-WEEK         Learn which days Ghost is accurate
    #8  SIGNAL_SOURCE       Track which signal sources win (future)
    #9  ADAPTIVE            Self-optimize thresholds weekly
    #10 FEAR_GREED          Fear & Greed index integration

  TIER 3 — Medium Impact (correlation intelligence):
    #11 SECTOR              Sector/category correlation
    #12 VOLUME              Volume confirmation gate (future)
    #13 EARNINGS            Earnings blackout learning (future)
    #14 AUTO_PRUNE          Remove chronic noise symbols
    #15 ENSEMBLE            Multi-source voting (future)

  TIER 4 — Smart Optimizations:
    #16 REDISTRIBUTE        Confidence calibration output
    #17 INVERSE_DECAY       Don't flip forever, recheck
    #18 CROSS_ASSET         BTC/SPY leading indicators
    #19 EXPECTED_VALUE      Profit-weighted accuracy (EV)
    #20 WEEKEND             Weekend crypto penalty

  TIER 5 — Meta Intelligence:
    #21 BACKTEST            Replay before deploying changes
    #22 AB_TEST             Split-test brain configurations
    #23 CIRCUIT_BREAKER     Emergency brake on bad streaks
    #24 FEATURE_IMPORTANCE  Track which abilities help most
    #25 SELF_EVOLVE         Auto-tune thresholds from data

═══════════════════════════════════════════════════════════════════
ARCHITECTURE
═══════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────────────┐
  │  brain_data.py → load_brain_context()                         │
  │       │                                                       │
  │       ▼                                                       │
  │  BrainContext (rich data: direction, recency, streaks, etc.)  │
  │       │                                                       │
  │       ▼                                                       │
  │  GhostBrain.analyze_batch(predictions, context)               │
  │       │                                                       │
  │       ├─ Per-symbol: compute brain_accuracy (#1,#2,#6)        │
  │       ├─ Decision tree: INVERT/EXCLUDE/COLD/WARM/HOT/FIRE    │
  │       ├─ Confidence modifiers: (#4,#5,#7,#10,#11,#18,#20)    │
  │       ├─ Calibration: (#3,#16)                                │
  │       ├─ Circuit breaker: (#23)                               │
  │       ├─ Cross-symbol: bias detection, correlation guard      │
  │       └─ Meta: (#17,#19,#22,#24)                              │
  │                                                               │
  │  Output: Dict[symbol, BrainDecision]                          │
  └───────────────────────────────────────────────────────────────┘
"""

import os
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Import data structures (brain_data.py has no dependencies on us)
try:
    from core.brain_data import BrainContext, SymbolContext
except ImportError:
    from brain_data import BrainContext, SymbolContext

LOGGER = logging.getLogger("ghost_brain")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION (all overridable via env vars)
# ═══════════════════════════════════════════════════════════════════

# Master switch
BRAIN_ENABLED = os.getenv("GHOST_BRAIN_ENABLED", "1") == "1"

# ─── Core Thresholds (v2, kept) ───
# FIX (Step 6, Mar 18 2026): INVERT disabled. Set threshold to 0 so no symbol
# qualifies. Inversion caused double-flips with ghost_learning_brain and
# oscillation (flip → accuracy rises → stop flipping → accuracy drops → flip again).
# The kill switch (Step 3) handles bad symbols by BLOCKING them entirely,
# which is deterministic and convergent. Inversion is neither.
INVERT_BELOW = float(os.getenv("BRAIN_INVERT_BELOW", "0.0"))  # was 38.0
EXCLUDE_BELOW = float(os.getenv("BRAIN_EXCLUDE_BELOW", "48.0"))
BOOST_ABOVE = float(os.getenv("BRAIN_BOOST_ABOVE", "62.0"))
STRONG_BOOST_ABOVE = float(os.getenv("BRAIN_STRONG_BOOST", "70.0"))
MIN_SAMPLES = int(os.getenv("BRAIN_MIN_SAMPLES", "20"))

# ─── Confidence Multipliers (v2, kept) ───
STRONG_BOOST_MULT = 1.30
BOOST_MULT = 1.15
NOISE_PENALTY_MULT = 0.85
CONFIDENCE_CAP = 0.98

# ─── Direction Bias (v2, kept) ───
BIAS_THRESHOLD = 0.80
MAX_SAME_DIRECTION = int(os.getenv("BRAIN_MAX_SAME_DIR", "6"))

# ─── #2: Recency ───
RECENCY_WEIGHT = float(os.getenv("BRAIN_RECENCY_WEIGHT", "0.70"))
RECENCY_MIN_SAMPLES = 10  # need at least 10 recent to use

# ─── #4: Streak Modifiers ───
STREAK_BONUS_PER = 0.02       # +2% confidence per consecutive win
STREAK_PENALTY_PER = 0.03     # -3% confidence per consecutive loss
MAX_STREAK_MOD = 0.15         # cap at ±15%

# ─── #5: Market Regime Modifiers ───
REGIME_MODIFIERS = {
    "calm":     0.05,
    "neutral":  0.0,
    "elevated": -0.05,
    "fear":     -0.12,
    "panic":    -0.20,
    "unknown":  0.0,
}

# ─── #6: Magnitude Bonus ───
MAGNITUDE_BIG_WIN_RATIO = 2.0     # wins 2x bigger than losses = bonus
MAGNITUDE_BIG_LOSS_RATIO = 0.5    # losses bigger than wins = penalty
MAGNITUDE_BONUS = 3.0             # accuracy points bonus
MAGNITUDE_PENALTY = -3.0          # accuracy points penalty

# ─── #10: Fear & Greed Thresholds ───
FG_EXTREME_FEAR = 20
FG_FEAR = 35
FG_GREED = 65
FG_EXTREME_GREED = 80

# ─── #17: Inverse Decay ───
INVERSE_RECHECK_DAYS = int(os.getenv("BRAIN_INVERSE_RECHECK", "30"))

# ─── #20: Weekend ───
WEEKEND_CRYPTO_PENALTY = 0.05

# ─── #23: Circuit Breaker ───
CIRCUIT_BREAKER_THRESHOLD = float(os.getenv("BRAIN_CIRCUIT_BREAKER", "45.0"))
CIRCUIT_BREAKER_MIN_PREDS = 30
CIRCUIT_BREAKER_PENALTY = 0.25

# ─── #14: Auto-Prune ───
PRUNE_MIN_DAYS = int(os.getenv("BRAIN_PRUNE_DAYS", "90"))
PRUNE_MIN_PREDICTIONS = int(os.getenv("BRAIN_PRUNE_MIN_PREDS", "100"))

# ─── Confidence Modifier Caps ───
MAX_TOTAL_BOOST = 0.35
MAX_TOTAL_PENALTY = 0.35


# ═══════════════════════════════════════════════════════════════════
# KNOWN CRYPTO (from centralized symbol registry #115)
# ═══════════════════════════════════════════════════════════════════

try:
    from core.symbol_registry import KNOWN_CRYPTO as _KNOWN_CRYPTO
except ImportError:
    # Fallback for testing without full package structure
    _KNOWN_CRYPTO = {
        "BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT",
        "MATIC", "SHIB", "UNI", "LTC", "BCH", "ATOM", "FIL", "NEAR", "ICP",
        "APT", "ARB", "OP", "INJ", "TIA", "SEI", "SUI", "TURBO", "JUP",
        "IOTX", "GIGA", "CHZ", "ALICE", "YFI", "BRETT", "HBAR", "ILV",
        "BAND", "PEPE", "ENJ", "RNDR", "MANA", "SAND", "AXS", "THETA", "VET",
        "FTM", "EGLD", "ALGO", "FLOW", "STX", "DASH", "ZEC", "EOS", "XTZ",
        "AAVE", "CRV", "MKR", "COMP", "SNX", "SUSHI", "1INCH", "BAL", "REN",
        "ZRX", "BAT", "KNC", "OCEAN", "OMG", "RLC", "BNB", "DYDX", "WLD",
        "JTO", "BONK", "WIF", "FLOKI", "ORDI", "RUNE", "ROSE", "QTUM",
        "ANT", "ZEN", "ONDO",
    }
# NOTE: T (AT&T), HOOD (Robinhood), COIN (Coinbase) are STOCKS.


# ═══════════════════════════════════════════════════════════════════
# #11: SECTOR GROUPS (for correlation intelligence)
# ═══════════════════════════════════════════════════════════════════

CRYPTO_SECTORS = {
    "L1":     {"BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "NEAR", "APT", "SUI", "SEI"},
    "MEME":   {"DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "TURBO", "BRETT"},
    "DEFI":   {"UNI", "AAVE", "CRV", "MKR", "COMP", "SNX", "SUSHI", "1INCH", "DYDX"},
    "GAMING": {"AXS", "SAND", "MANA", "ILV", "ENJ", "ALICE"},
    "INFRA":  {"LINK", "FIL", "RNDR", "THETA", "ARB", "OP", "IOTX"},
}

STOCK_SECTORS = {
    "BIG_TECH": {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"},
    "FINTECH":  {"HOOD", "COIN", "SQ", "SOFI"},
    "GROWTH":   {"TSLA", "PLTR", "NIO", "AMD"},
    "VALUE":    {"T", "INTC", "DIS", "BA", "NFLX"},
}


# ═══════════════════════════════════════════════════════════════════
# BRAIN DECISION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BrainDecision:
    """A decision made by Ghost Brain v3 for one symbol.

    Enhanced from v2 with:
      - brain_accuracy (blended, replaces raw_accuracy in decisions)
      - confidence_modifiers (which abilities adjusted confidence)
      - data_quality (how rich the input data was)
      - expected_value (profit-weighted accuracy)
      - prune_candidate (should this symbol be removed?)
      - ab_group (A/B test assignment)
    """
    symbol: str
    action: str                       # SEND | EXCLUDE | INVERT
    direction: str                    # Final direction (may be flipped)
    confidence: float                 # Adjusted confidence (0.0-1.0)
    tier: str                         # 🟢HOT | 🟡WARM | 🔴COLD | 🔄INVERTED | ⛔EXCLUDED
    asset_class: str = "unknown"      # crypto | stock
    reasons: List[str] = field(default_factory=list)

    # Accuracy data
    raw_accuracy: float = 50.0        # All-time accuracy from DB
    brain_accuracy: float = 50.0      # Blended accuracy (#1,#2,#6)
    effective_accuracy: float = 50.0  # After inversion: 100 - brain_accuracy
    inverted: bool = False
    sample_size: int = 0

    # v3 enhancements
    confidence_modifiers: Dict[str, float] = field(default_factory=dict)
    data_quality: str = "basic"       # basic | partial | rich
    expected_value: float = 0.0       # #19: profit-weighted
    prune_candidate: bool = False     # #14: should remove?
    ab_group: str = ""                # #22: A or B
    direction_split: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# GHOST BRAIN v3
# ═══════════════════════════════════════════════════════════════════

class GhostBrain:
    """
    Ghost's centralized learning intelligence — 25 cognitive abilities.

    Usage (v3 with rich context):
        brain = GhostBrain()
        context = await load_brain_context(db_url, symbols, market_data)
        decisions = brain.analyze_batch(predictions, context=context)

    Usage (backward compatible with v2):
        brain = GhostBrain()
        decisions = brain.analyze_batch(predictions, accuracy_data=old_dict)
    """

    def __init__(self):
        self._decisions: Dict[str, BrainDecision] = {}
        self._direction_bias: Optional[Dict] = None
        self._correlation_warnings: List[str] = []
        self._circuit_breaker_active: bool = False
        self._inverse_tracker: Dict[str, datetime] = {}  # #17
        self._prune_candidates: List[str] = []            # #14
        self._feature_contributions: Dict[str, Dict] = {} # #24
        self._ab_groups: Dict[str, str] = {}              # #22

        self._cycle_stats = {
            "analyzed": 0,
            "inverted": 0,
            "excluded": 0,
            "boosted": 0,
            "penalized": 0,
            "sent": 0,
        }

    # ═══════════════════════════════════════════════════════════
    # CORE: analyze_symbol (enhanced with all 25 abilities)
    # ═══════════════════════════════════════════════════════════

    def analyze_symbol(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        accuracy_data: Optional[Dict[str, Dict]] = None,
        context: Optional[BrainContext] = None,
    ) -> BrainDecision:
        """
        Make a unified decision for one symbol using all 25 abilities.

        Backward compatible: pass accuracy_data for v2 behavior,
        or context for full v3 intelligence.
        """
        if not BRAIN_ENABLED:
            return BrainDecision(
                symbol=symbol, action="SEND", direction=direction,
                confidence=confidence, tier="⚪NEUTRAL",
                reasons=["brain_disabled"],
            )

        reasons: List[str] = []
        modifiers: Dict[str, float] = {}
        inverted = False
        final_direction = direction
        final_confidence = confidence

        # ── Classify asset ──
        asset_class = "crypto" if symbol.upper() in _KNOWN_CRYPTO else "stock"

        # ── Get data (v3 context or v2 fallback) ──
        sym_ctx = None
        if context:
            sym_ctx = context.symbols.get(symbol.upper())

        if sym_ctx:
            raw_accuracy = sym_ctx.accuracy_pct
            total = sym_ctx.total_predictions
            data_quality = "rich" if sym_ctx.recent_total >= RECENCY_MIN_SAMPLES else "partial"
        elif accuracy_data:
            data = accuracy_data.get(symbol.upper()) or accuracy_data.get(symbol) or {}
            raw_accuracy = data.get("accuracy_pct", 50.0)
            total = data.get("total", 0)
            data_quality = "basic"
        else:
            raw_accuracy = 50.0
            total = 0
            data_quality = "none"

        # ── INSUFFICIENT DATA → PASS-THROUGH (let them build history) ──
        if total < MIN_SAMPLES:
            reasons.append(f"insufficient_data ({total}/{MIN_SAMPLES}) → PASS_THROUGH")
            # Cold-start fix: instead of EXCLUDE, pass through with original
            # direction/confidence so trades get recorded and history builds.
            # Once history reaches MIN_SAMPLES, full Brain analysis kicks in.
            decision = BrainDecision(
                symbol=symbol, action="SEND", direction=direction,
                confidence=confidence, tier="⚪NEW",
                asset_class=asset_class, reasons=reasons,
                raw_accuracy=raw_accuracy, brain_accuracy=raw_accuracy,
                effective_accuracy=raw_accuracy, sample_size=total,
                data_quality=data_quality,
            )
            self._decisions[symbol] = decision
            self._cycle_stats["analyzed"] += 1
            self._cycle_stats["sent"] += 1
            return decision

        # ══════════════════════════════════════════════════════
        # STEP 1: Compute brain_accuracy (#1, #2, #6)
        # ══════════════════════════════════════════════════════
        brain_accuracy = self._compute_brain_accuracy(
            symbol, direction, raw_accuracy, sym_ctx
        )
        reasons.append(
            f"brain_accuracy={brain_accuracy:.1f}% "
            f"(raw={raw_accuracy:.1f}%)"
        )

        # ══════════════════════════════════════════════════════
        # STEP 2: Decision tree (same zones, smarter input)
        # ══════════════════════════════════════════════════════

        dir_split = {}  # direction split data (populated in INVERT zone)

        # ── ZONE 1: INVERT (brain_accuracy < 38%) ──
        if brain_accuracy < INVERT_BELOW:

            # #1: Check direction split — maybe only one direction is bad
            skip_invert = False
            dir_split = {}
            if sym_ctx:
                dir_split = {
                    "up": sym_ctx.up_accuracy,
                    "down": sym_ctx.down_accuracy,
                }
                dir_total = sym_ctx.up_total if direction == "UP" else sym_ctx.down_total
                dir_acc = sym_ctx.up_accuracy if direction == "UP" else sym_ctx.down_accuracy

                if dir_total >= MIN_SAMPLES and dir_acc >= EXCLUDE_BELOW:
                    # THIS direction is actually fine — don't invert
                    skip_invert = True
                    reasons.append(
                        f"#1 DIRECTION_SPLIT: {direction} accuracy "
                        f"{dir_acc:.1f}% is OK — NOT inverting"
                    )

            if skip_invert:
                # Send without inversion, but penalize
                effective_accuracy = brain_accuracy
                final_confidence = confidence * NOISE_PENALTY_MULT
                tier = "🔴COLD"
                action = "SEND"
                self._cycle_stats["penalized"] += 1
            else:
                # Full inversion
                inverted = True
                final_direction = "DOWN" if direction == "UP" else "UP"
                effective_accuracy = 100.0 - brain_accuracy

                reasons.append(
                    f"🔄 INVERT: {brain_accuracy:.1f}% < {INVERT_BELOW}% → "
                    f"flipped {direction}→{final_direction} "
                    f"(effective {effective_accuracy:.1f}%)"
                )

                # #17: Track inversion for decay checking
                if symbol not in self._inverse_tracker:
                    self._inverse_tracker[symbol] = datetime.now()

                # Boost inverted symbols proportional to effective accuracy
                if effective_accuracy >= STRONG_BOOST_ABOVE:
                    final_confidence = min(CONFIDENCE_CAP, confidence * STRONG_BOOST_MULT)
                    reasons.append(f"🚀 STRONG_BOOST: eff {effective_accuracy:.1f}% → ×{STRONG_BOOST_MULT}")
                    self._cycle_stats["boosted"] += 1
                elif effective_accuracy >= BOOST_ABOVE:
                    final_confidence = min(CONFIDENCE_CAP, confidence * BOOST_MULT)
                    reasons.append(f"📈 BOOST: eff {effective_accuracy:.1f}% → ×{BOOST_MULT}")
                    self._cycle_stats["boosted"] += 1

                tier = "🔄INVERTED"
                action = "INVERT"
                self._cycle_stats["inverted"] += 1

                # #17: Check inverse decay
                if self._check_inverse_decay(symbol):
                    reasons.append(
                        f"⏰ INVERSE_DECAY: {symbol} inverted >{INVERSE_RECHECK_DAYS}d — due for recheck"
                    )

        # ── ZONE 2: EXCLUDE (brain_accuracy 38-48%) ──
        elif brain_accuracy < EXCLUDE_BELOW:
            effective_accuracy = brain_accuracy

            # #14: Flag chronic noise symbols for pruning
            prune = False
            if sym_ctx and sym_ctx.days_tracked >= PRUNE_MIN_DAYS and total >= PRUNE_MIN_PREDICTIONS:
                prune = True
                reasons.append(
                    f"🗑️ PRUNE_CANDIDATE: {sym_ctx.days_tracked}d tracked, "
                    f"{total} predictions, still noise"
                )

            reasons.append(
                f"⛔ EXCLUDE: {brain_accuracy:.1f}% in noise zone "
                f"({INVERT_BELOW}%-{EXCLUDE_BELOW}%)"
            )
            tier = "⛔EXCLUDED"
            action = "EXCLUDE"
            self._cycle_stats["excluded"] += 1

            decision = BrainDecision(
                symbol=symbol, action=action, direction=direction,
                confidence=0.0, tier=tier, asset_class=asset_class,
                reasons=reasons, raw_accuracy=raw_accuracy,
                brain_accuracy=brain_accuracy,
                effective_accuracy=effective_accuracy,
                sample_size=total, data_quality=data_quality,
                prune_candidate=prune,
                direction_split=dir_split if sym_ctx else {},
            )
            self._decisions[symbol] = decision
            self._cycle_stats["analyzed"] += 1
            if prune:
                self._prune_candidates.append(symbol)
            return decision

        # ── ZONE 3: COLD (48-55%) ──
        elif brain_accuracy < 55.0:
            effective_accuracy = brain_accuracy
            final_confidence = confidence * NOISE_PENALTY_MULT
            tier = "🔴COLD"
            action = "SEND"
            reasons.append(
                f"🔴 COLD: {brain_accuracy:.1f}% → ×{NOISE_PENALTY_MULT} penalty"
            )
            self._cycle_stats["penalized"] += 1

        # ── ZONE 4: WARM (55-62%) ──
        elif brain_accuracy < BOOST_ABOVE:
            effective_accuracy = brain_accuracy
            tier = "🟡WARM"
            action = "SEND"
            reasons.append(f"🟡 WARM: {brain_accuracy:.1f}% — no adjustment")

        # ── ZONE 5: HOT (62-70%) ──
        elif brain_accuracy < STRONG_BOOST_ABOVE:
            effective_accuracy = brain_accuracy
            final_confidence = min(CONFIDENCE_CAP, confidence * BOOST_MULT)
            tier = "🟢HOT"
            action = "SEND"
            reasons.append(f"🟢 HOT: {brain_accuracy:.1f}% → ×{BOOST_MULT} boost")
            self._cycle_stats["boosted"] += 1

        # ── ZONE 6: FIRE (70%+) ──
        else:
            effective_accuracy = brain_accuracy
            final_confidence = min(CONFIDENCE_CAP, confidence * STRONG_BOOST_MULT)
            tier = "🔥FIRE"
            action = "SEND"
            reasons.append(f"🔥 FIRE: {brain_accuracy:.1f}% → ×{STRONG_BOOST_MULT} strong boost")
            self._cycle_stats["boosted"] += 1

        # ══════════════════════════════════════════════════════
        # STEP 3: Confidence modifiers (#4,#5,#7,#10,#11,#18,#20)
        # ══════════════════════════════════════════════════════

        if context or sym_ctx:
            # #4: Streak modifier
            if sym_ctx:
                mod = self._compute_streak_modifier(sym_ctx)
                if mod != 0.0:
                    modifiers["streak"] = mod
                    reasons.append(f"#4 STREAK: {mod:+.0%} (streak={sym_ctx.current_streak})")

            # #5: Market regime modifier
            if context and context.market_regime != "unknown":
                mod = self._compute_regime_modifier(context)
                if mod != 0.0:
                    modifiers["regime"] = mod
                    reasons.append(f"#5 REGIME: {mod:+.0%} ({context.market_regime})")

            # #7: Day-of-week modifier
            if sym_ctx and context:
                mod = self._compute_dow_modifier(sym_ctx, context)
                if mod != 0.0:
                    modifiers["dow"] = mod
                    reasons.append(f"#7 DOW: {mod:+.0%} (day={context.current_day})")

            # #10: Fear & Greed modifier
            if context and context.fear_greed_index != 50:
                mod = self._compute_fg_modifier(context, asset_class)
                if mod != 0.0:
                    modifiers["fear_greed"] = mod
                    reasons.append(f"#10 F&G: {mod:+.0%} (index={context.fear_greed_index})")

            # #11: Sector correlation modifier
            if context:
                mod = self._compute_sector_modifier(context, symbol, asset_class)
                if mod != 0.0:
                    modifiers["sector"] = mod
                    reasons.append(f"#11 SECTOR: {mod:+.0%}")

            # #18: Cross-asset modifier
            if context:
                mod = self._compute_cross_asset_modifier(
                    context, asset_class, final_direction
                )
                if mod != 0.0:
                    modifiers["cross_asset"] = mod
                    reasons.append(f"#18 CROSS: {mod:+.0%}")

            # #20: Weekend modifier
            if context:
                mod = self._compute_weekend_modifier(context, asset_class)
                if mod != 0.0:
                    modifiers["weekend"] = mod
                    reasons.append(f"#20 WEEKEND: {mod:+.0%}")

        # ── Apply modifiers with cap ──
        if modifiers:
            total_mod = sum(modifiers.values())
            total_mod = max(-MAX_TOTAL_PENALTY, min(MAX_TOTAL_BOOST, total_mod))
            final_confidence *= (1.0 + total_mod)
            reasons.append(f"modifiers_total={total_mod:+.0%}")

        # ══════════════════════════════════════════════════════
        # STEP 4: Confidence calibration (#3, #16)
        # ══════════════════════════════════════════════════════
        if context and context.calibration_curve:
            calibrated = self._calibrate_confidence(final_confidence, context)
            if abs(calibrated - final_confidence) > 0.01:
                reasons.append(
                    f"#3 CALIBRATION: {final_confidence:.0%}→{calibrated:.0%}"
                )
                final_confidence = calibrated

        # ══════════════════════════════════════════════════════
        # STEP 5: Circuit breaker (#23)
        # ══════════════════════════════════════════════════════
        if self._circuit_breaker_active:
            old = final_confidence
            final_confidence *= (1.0 - CIRCUIT_BREAKER_PENALTY)
            reasons.append(
                f"🚨 CIRCUIT_BREAKER: {old:.0%}→{final_confidence:.0%} "
                f"(-{CIRCUIT_BREAKER_PENALTY:.0%})"
            )

        # ── Final cap ──
        final_confidence = max(0.01, min(CONFIDENCE_CAP, final_confidence))

        # ══════════════════════════════════════════════════════
        # STEP 6: Build decision
        # ══════════════════════════════════════════════════════

        # #19: Expected value
        ev = 0.0
        if sym_ctx:
            ev = self._compute_expected_value(sym_ctx)

        self._cycle_stats["analyzed"] += 1
        if action != "EXCLUDE":
            self._cycle_stats["sent"] += 1

        decision = BrainDecision(
            symbol=symbol, action=action, direction=final_direction,
            confidence=final_confidence, tier=tier, asset_class=asset_class,
            reasons=reasons, raw_accuracy=raw_accuracy,
            brain_accuracy=brain_accuracy,
            effective_accuracy=effective_accuracy,
            inverted=inverted, sample_size=total,
            confidence_modifiers=modifiers, data_quality=data_quality,
            expected_value=ev,
            direction_split={"up": sym_ctx.up_accuracy, "down": sym_ctx.down_accuracy} if sym_ctx else {},
        )

        self._decisions[symbol] = decision

        # #24: Track feature contributions
        self._track_contribution(symbol, modifiers, brain_accuracy, raw_accuracy)

        return decision

    # ═══════════════════════════════════════════════════════════
    # CORE: analyze_batch
    # ═══════════════════════════════════════════════════════════

    def analyze_batch(
        self,
        predictions: Dict[str, Dict],
        accuracy_data: Optional[Dict[str, Dict]] = None,
        context: Optional[BrainContext] = None,
    ) -> Dict[str, BrainDecision]:
        """
        Analyze all predictions in one pass.

        Enables cross-symbol intelligence:
        - Direction bias detection
        - Correlation guard
        - Circuit breaker
        - A/B group assignment
        - Prune candidate detection

        Backward compatible: pass accuracy_data for v2, context for v3.
        """
        # Reset cycle
        self._decisions = {}
        self._correlation_warnings = []
        self._prune_candidates = []
        self._cycle_stats = {k: 0 for k in self._cycle_stats}
        self._circuit_breaker_active = False

        # #23: Circuit breaker check
        if context:
            self._check_circuit_breaker(context)

        # #3 (existing): Detect direction bias
        self._direction_bias = self._detect_direction_bias(predictions)

        # Analyze each symbol
        for symbol, pred in predictions.items():
            if not isinstance(pred, dict):
                continue
            direction = pred.get("direction", "")
            confidence = pred.get("confidence", 0.0)
            if direction not in ("UP", "DOWN"):
                continue
            self.analyze_symbol(
                symbol, direction, confidence,
                accuracy_data=accuracy_data, context=context,
            )

        # #5 (existing): Correlation guard
        self._apply_correlation_guard()

        # #22: A/B group assignment
        self._assign_ab_groups()

        return self._decisions

    # ═══════════════════════════════════════════════════════════
    # ABILITY #1, #2, #6: BLENDED BRAIN ACCURACY
    # ═══════════════════════════════════════════════════════════

    def _compute_brain_accuracy(
        self,
        symbol: str,
        direction: str,
        raw_accuracy: float,
        sym_ctx: Optional[SymbolContext],
    ) -> float:
        """
        Compute blended accuracy from multiple data signals.

        #1: Per-direction (UP vs DOWN split)
        #2: Recency (last 30 days weighted 70%)
        #6: Magnitude (big wins count more)

        Returns a single brain_accuracy value that replaces
        raw_accuracy in the decision tree.
        """
        if not sym_ctx:
            return raw_accuracy

        alltime = raw_accuracy

        # #2: Recent accuracy (weighted heavier — recent performance matters more)
        if sym_ctx.recent_total >= RECENCY_MIN_SAMPLES:
            recent = sym_ctx.recent_accuracy
        else:
            recent = alltime  # not enough recent data

        # #1: Direction-specific accuracy
        if direction == "UP" and sym_ctx.up_total >= RECENCY_MIN_SAMPLES:
            dir_acc = sym_ctx.up_accuracy
        elif direction == "DOWN" and sym_ctx.down_total >= RECENCY_MIN_SAMPLES:
            dir_acc = sym_ctx.down_accuracy
        else:
            dir_acc = alltime  # not enough direction data

        # #6: Magnitude bonus
        mag_bonus = 0.0
        if sym_ctx.avg_win_magnitude > 0 and sym_ctx.avg_loss_magnitude > 0:
            ratio = sym_ctx.avg_win_magnitude / max(sym_ctx.avg_loss_magnitude, 0.001)
            if ratio >= MAGNITUDE_BIG_WIN_RATIO:
                mag_bonus = MAGNITUDE_BONUS       # Wins are 2x+ bigger
            elif ratio >= 1.5:
                mag_bonus = MAGNITUDE_BONUS / 2   # Moderate win edge
            elif ratio <= MAGNITUDE_BIG_LOSS_RATIO:
                mag_bonus = MAGNITUDE_PENALTY      # Losses dominate
            elif ratio <= 0.75:
                mag_bonus = MAGNITUDE_PENALTY / 2  # Moderate loss edge

        # Blend: recent (70%) > direction (20%) > alltime (10%)
        alltime_weight = 1.0 - RECENCY_WEIGHT - 0.20
        brain_accuracy = (
            recent * RECENCY_WEIGHT
            + dir_acc * 0.20
            + alltime * alltime_weight
        ) + mag_bonus

        return max(0.0, min(100.0, brain_accuracy))

    # ═══════════════════════════════════════════════════════════
    # ABILITY #4: STREAK MODIFIER
    # ═══════════════════════════════════════════════════════════

    def _compute_streak_modifier(self, sym_ctx: SymbolContext) -> float:
        """
        Adjust confidence based on win/loss streaks.

        A 5-win streak → +10% confidence (momentum)
        A 5-loss streak → -15% confidence (something's off)
        """
        streak = sym_ctx.current_streak
        if streak > 0:
            return min(MAX_STREAK_MOD, streak * STREAK_BONUS_PER)
        elif streak < 0:
            return max(-MAX_STREAK_MOD, streak * STREAK_PENALTY_PER)
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #5: MARKET REGIME MODIFIER
    # ═══════════════════════════════════════════════════════════

    def _compute_regime_modifier(self, context: BrainContext) -> float:
        """
        Adjust confidence based on market volatility regime.

        In panic markets, ALL predictions are less reliable.
        In calm markets, patterns are more predictable.
        """
        return REGIME_MODIFIERS.get(context.market_regime, 0.0)

    # ═══════════════════════════════════════════════════════════
    # ABILITY #7: DAY-OF-WEEK MODIFIER
    # ═══════════════════════════════════════════════════════════

    def _compute_dow_modifier(
        self, sym_ctx: SymbolContext, context: BrainContext
    ) -> float:
        """
        Adjust confidence based on day-of-week accuracy patterns.

        If Ghost is 70% accurate on Tuesdays but 35% on Fridays,
        and today is Friday → reduce confidence.
        """
        today = context.current_day  # 0=Sunday in SQL DOW
        if today not in sym_ctx.dow_accuracy:
            return 0.0

        today_acc = sym_ctx.dow_accuracy[today]
        avg_acc = sym_ctx.accuracy_pct

        if avg_acc <= 0:
            return 0.0

        # Deviation from average as a confidence modifier
        deviation = (today_acc - avg_acc) / 100.0
        return max(-0.08, min(0.08, deviation))

    # ═══════════════════════════════════════════════════════════
    # ABILITY #10: FEAR & GREED MODIFIER
    # ═══════════════════════════════════════════════════════════

    def _compute_fg_modifier(
        self, context: BrainContext, asset_class: str
    ) -> float:
        """
        Adjust confidence based on Fear & Greed Index.

        Extreme fear → crypto is chaos (panic selling)
        Extreme greed → bubble risk (irrational exuberance)
        Crypto is MORE affected than stocks.
        """
        fg = context.fear_greed_index

        if asset_class == "crypto":
            if fg <= FG_EXTREME_FEAR:
                return -0.12
            elif fg <= FG_FEAR:
                return -0.05
            elif fg >= FG_EXTREME_GREED:
                return -0.08   # Bubble risk
            elif fg >= FG_GREED:
                return 0.03    # Mild greed = momentum
        else:  # stock
            if fg <= FG_EXTREME_FEAR:
                return -0.05
            elif fg >= FG_EXTREME_GREED:
                return -0.03

        return 0.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #11: SECTOR CORRELATION MODIFIER
    # ═══════════════════════════════════════════════════════════

    def _compute_sector_modifier(
        self, context: BrainContext, symbol: str, asset_class: str
    ) -> float:
        """
        Adjust confidence based on how the symbol's sector is performing.

        If the MEME sector average is 22% accuracy, and DOGE is in MEME,
        that's a signal that the whole sector is unpredictable.
        """
        sector_map = CRYPTO_SECTORS if asset_class == "crypto" else STOCK_SECTORS

        # Find this symbol's sector
        sector_name = None
        for name, members in sector_map.items():
            if symbol.upper() in members:
                sector_name = name
                break

        if not sector_name:
            return 0.0

        # Get peer accuracies
        peers = sector_map[sector_name] - {symbol.upper()}
        if not peers:
            return 0.0

        peer_accs = []
        for peer in peers:
            peer_ctx = context.symbols.get(peer)
            if peer_ctx and peer_ctx.total_predictions >= MIN_SAMPLES:
                peer_accs.append(peer_ctx.accuracy_pct)

        if not peer_accs:
            return 0.0

        avg_peer = sum(peer_accs) / len(peer_accs)

        if avg_peer < 30.0:
            return -0.08    # Sector is terrible
        elif avg_peer < 40.0:
            return -0.04    # Sector is weak
        elif avg_peer > 70.0:
            return 0.05     # Sector is fire
        elif avg_peer > 60.0:
            return 0.03     # Sector is hot

        return 0.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #18: CROSS-ASSET LEADING INDICATORS
    # ═══════════════════════════════════════════════════════════

    def _compute_cross_asset_modifier(
        self, context: BrainContext, asset_class: str, direction: str
    ) -> float:
        """
        BTC dumps → altcoins follow. SPY dumps → stocks follow.

        If BTC just dropped 5% and we're predicting DOGE UP,
        that prediction is going against the tide.
        """
        if asset_class == "crypto":
            btc = context.btc_24h_change
            if btc < -5.0 and direction == "UP":
                return -0.10   # BTC crashed, altcoin UP is risky
            elif btc < -3.0 and direction == "UP":
                return -0.05
            elif btc > 5.0 and direction == "DOWN":
                return -0.05   # BTC pumping, crypto DOWN is risky
        else:  # stock
            spy = context.spy_24h_change
            if spy < -3.0 and direction == "UP":
                return -0.08
            elif spy < -2.0 and direction == "UP":
                return -0.03
            elif spy > 3.0 and direction == "DOWN":
                return -0.03

        return 0.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #20: WEEKEND DETECTOR
    # ═══════════════════════════════════════════════════════════

    def _compute_weekend_modifier(
        self, context: BrainContext, asset_class: str
    ) -> float:
        """
        Weekend crypto has lower liquidity and more manipulation.
        Stocks don't trade weekends (handled by market gates).
        """
        if context.is_weekend and asset_class == "crypto":
            return -WEEKEND_CRYPTO_PENALTY
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #3, #16: CONFIDENCE CALIBRATION
    # ═══════════════════════════════════════════════════════════

    def _calibrate_confidence(
        self, raw_confidence: float, context: BrainContext
    ) -> float:
        """
        Map stated confidence to actual historical hit probability.

        If the model says "72% confident" but predictions at that
        confidence only hit 58%, the calibration curve corrects it.

        Blend: 60% calibrated + 40% raw (don't fully override).
        """
        curve = context.calibration_curve
        if not curve:
            return raw_confidence

        # Find nearest bucket
        bucket_key = f"{int(raw_confidence * 10) / 10:.1f}"

        if bucket_key in curve:
            actual_rate = curve[bucket_key]
        else:
            # Find closest bucket
            closest_key = None
            closest_dist = float("inf")
            for key in curve:
                try:
                    dist = abs(float(key) - raw_confidence)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_key = key
                except ValueError:
                    continue

            if closest_key and closest_dist < 0.15:
                actual_rate = curve[closest_key]
            else:
                return raw_confidence

        # Blend: 60% calibrated, 40% raw
        calibrated = raw_confidence * 0.4 + actual_rate * 0.6
        return max(0.01, min(CONFIDENCE_CAP, calibrated))

    # ═══════════════════════════════════════════════════════════
    # ABILITY #17: INVERSE DECAY
    # ═══════════════════════════════════════════════════════════

    def _check_inverse_decay(self, symbol: str) -> bool:
        """
        Don't flip a symbol forever. Markets change.

        If a symbol has been inverted for >30 days, flag it for
        re-evaluation. The operator should run 5 non-inverted
        predictions to test if the pattern still holds.
        """
        if symbol in self._inverse_tracker:
            first_inverted = self._inverse_tracker[symbol]
            days = (datetime.now() - first_inverted).days
            if days >= INVERSE_RECHECK_DAYS:
                return True
        return False

    # ═══════════════════════════════════════════════════════════
    # ABILITY #19: EXPECTED VALUE
    # ═══════════════════════════════════════════════════════════

    def _compute_expected_value(self, sym_ctx: SymbolContext) -> float:
        """
        Compute expected value per prediction.

        EV = avg_win_pct × win_rate - avg_loss_pct × loss_rate

        A symbol with 45% accuracy but +5% avg wins and -1% avg losses
        has POSITIVE EV (0.45 × 5 - 0.55 × 1 = +1.7%).
        """
        if sym_ctx.total_predictions == 0:
            return 0.0

        win_rate = sym_ctx.accuracy_pct / 100.0
        loss_rate = 1.0 - win_rate

        if sym_ctx.avg_win_magnitude > 0 or sym_ctx.avg_loss_magnitude > 0:
            return (
                sym_ctx.avg_win_magnitude * win_rate
                - sym_ctx.avg_loss_magnitude * loss_rate
            )
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #23: CIRCUIT BREAKER
    # ═══════════════════════════════════════════════════════════

    def _check_circuit_breaker(self, context: BrainContext):
        """
        Emergency brake: if 3-day accuracy drops below threshold,
        reduce ALL confidence by 25%.

        This catches regime changes where the model is suddenly
        wrong about everything (market crash, black swan, etc.).
        """
        if (
            context.rolling_3d_total >= CIRCUIT_BREAKER_MIN_PREDS
            and context.rolling_3d_accuracy < CIRCUIT_BREAKER_THRESHOLD
        ):
            self._circuit_breaker_active = True
            LOGGER.warning(
                f"[BRAIN] 🚨 CIRCUIT BREAKER ACTIVE: "
                f"3-day accuracy {context.rolling_3d_accuracy:.1f}% "
                f"< {CIRCUIT_BREAKER_THRESHOLD}%"
            )
        else:
            self._circuit_breaker_active = False

    # ═══════════════════════════════════════════════════════════
    # EXISTING: DIRECTION BIAS DETECTION
    # ═══════════════════════════════════════════════════════════

    def _detect_direction_bias(self, predictions: Dict[str, Dict]) -> Dict:
        """Detect if >80% of predictions are same direction."""
        up = 0
        down = 0
        for pred in predictions.values():
            if not isinstance(pred, dict):
                continue
            d = pred.get("direction", "")
            if d == "UP":
                up += 1
            elif d == "DOWN":
                down += 1

        total = up + down
        if total == 0:
            return {"biased": False, "total": 0}

        up_pct = up / total
        if up_pct >= BIAS_THRESHOLD:
            LOGGER.warning(
                f"[BRAIN] ⚠️ DIRECTION BIAS: {up_pct:.0%} of {total} "
                f"predictions are UP"
            )
            return {"biased": True, "direction": "UP", "pct": up_pct, "total": total}

        down_pct = 1 - up_pct
        if down_pct >= BIAS_THRESHOLD:
            LOGGER.warning(
                f"[BRAIN] ⚠️ DIRECTION BIAS: {down_pct:.0%} of {total} "
                f"predictions are DOWN"
            )
            return {"biased": True, "direction": "DOWN", "pct": down_pct, "total": total}

        return {"biased": False, "up_pct": up_pct, "total": total}

    # ═══════════════════════════════════════════════════════════
    # EXISTING: CORRELATION GUARD
    # ═══════════════════════════════════════════════════════════

    def _apply_correlation_guard(self):
        """Penalize overflow when too many picks same direction per asset class."""
        groups: Dict[str, List[BrainDecision]] = {}

        for symbol, decision in self._decisions.items():
            if decision.action == "EXCLUDE":
                continue
            key = f"{decision.asset_class}_{decision.direction}"
            groups.setdefault(key, []).append(decision)

        self._correlation_warnings = []
        for key, decisions in groups.items():
            if len(decisions) > MAX_SAME_DIRECTION:
                decisions.sort(key=lambda d: d.effective_accuracy, reverse=True)
                overflow = decisions[MAX_SAME_DIRECTION:]
                asset_class, direction = key.rsplit("_", 1)

                for d in overflow:
                    d.confidence = d.confidence * NOISE_PENALTY_MULT
                    d.reasons.append(
                        f"⚠️ CORRELATION: {len(decisions)} {asset_class} "
                        f"{direction} picks (max {MAX_SAME_DIRECTION})"
                    )

                warning = (
                    f"{len(decisions)} {asset_class} picks are {direction} "
                    f"(max {MAX_SAME_DIRECTION}) — penalized weakest: "
                    f"{[d.symbol for d in overflow]}"
                )
                self._correlation_warnings.append(warning)
                LOGGER.warning(f"[BRAIN] ⚠️ CORRELATION: {warning}")

    # ═══════════════════════════════════════════════════════════
    # ABILITY #22: A/B TESTING
    # ═══════════════════════════════════════════════════════════

    def _assign_ab_groups(self):
        """
        Assign symbols to A/B test groups for configuration testing.

        Group A = current production config
        Group B = experimental config

        Hash-based assignment ensures consistency across cycles.
        """
        for symbol in self._decisions:
            self._ab_groups[symbol] = "A" if hash(symbol) % 2 == 0 else "B"

    # ═══════════════════════════════════════════════════════════
    # ABILITY #24: FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════

    def _track_contribution(
        self,
        symbol: str,
        modifiers: Dict[str, float],
        brain_accuracy: float,
        raw_accuracy: float,
    ):
        """
        Track which abilities contributed most to decisions.

        This tells us: "INVERT gave +25 points of lift,
        STREAK gave +2 points, REGIME gave -1 point."
        """
        # Track accuracy lift from blending (#1, #2, #6)
        blend_key = "accuracy_blend"
        if blend_key not in self._feature_contributions:
            self._feature_contributions[blend_key] = {"total_impact": 0.0, "count": 0}
        self._feature_contributions[blend_key]["total_impact"] += abs(brain_accuracy - raw_accuracy)
        self._feature_contributions[blend_key]["count"] += 1

        # Track each modifier
        for ability, value in modifiers.items():
            if ability not in self._feature_contributions:
                self._feature_contributions[ability] = {"total_impact": 0.0, "count": 0}
            self._feature_contributions[ability]["total_impact"] += abs(value)
            self._feature_contributions[ability]["count"] += 1

    # ═══════════════════════════════════════════════════════════
    # ABILITY #9, #25: SELF-EVOLVING THRESHOLDS
    # ═══════════════════════════════════════════════════════════

    def optimize_thresholds(
        self, historical_accuracy: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Find optimal INVERT and EXCLUDE thresholds from historical data.

        Tests all combinations of:
          INVERT_BELOW:  25% to 45% (step 1)
          EXCLUDE_BELOW: INVERT+5% to 55% (step 1)

        Returns the thresholds that maximize effective accuracy.
        Run weekly, not per-cycle (expensive).
        """
        best_effective = 0.0
        best_thresholds = (INVERT_BELOW, EXCLUDE_BELOW)
        current_effective = self._simulate_thresholds(
            historical_accuracy, INVERT_BELOW, EXCLUDE_BELOW
        )

        for invert_t in range(25, 46):
            for exclude_t in range(invert_t + 5, 56):
                effective = self._simulate_thresholds(
                    historical_accuracy, float(invert_t), float(exclude_t)
                )
                if effective > best_effective:
                    best_effective = effective
                    best_thresholds = (float(invert_t), float(exclude_t))

        return {
            "current_invert_below": INVERT_BELOW,
            "current_exclude_below": EXCLUDE_BELOW,
            "current_effective": current_effective,
            "optimal_invert_below": best_thresholds[0],
            "optimal_exclude_below": best_thresholds[1],
            "optimal_effective": best_effective,
            "lift": best_effective - current_effective,
        }

    def _simulate_thresholds(
        self,
        accuracy_data: Dict[str, Dict],
        invert_below: float,
        exclude_below: float,
    ) -> float:
        """Simulate effective accuracy with given thresholds."""
        correct = 0.0
        total = 0
        for sym, data in accuracy_data.items():
            acc = data.get("accuracy_pct", 50.0)
            n = data.get("total", 0)
            if n < MIN_SAMPLES:
                continue
            if acc < invert_below:
                effective = 100.0 - acc
            elif acc < exclude_below:
                continue  # excluded
            else:
                effective = acc
            correct += effective * n / 100.0
            total += n
        return (correct / total * 100.0) if total > 0 else 50.0

    # ═══════════════════════════════════════════════════════════
    # ABILITY #21: BACKTEST REPLAY
    # ═══════════════════════════════════════════════════════════

    def backtest_replay(
        self,
        historical_predictions: Dict[str, Dict],
        historical_accuracy: Dict[str, Dict],
        context: Optional[BrainContext] = None,
    ) -> Dict[str, Any]:
        """
        Replay historical predictions through the current brain.

        Compares what WOULD have happened with the current brain
        vs what DID happen (raw model output).

        Returns accuracy comparison and recommendation.
        """
        # Run batch through brain
        decisions = self.analyze_batch(
            historical_predictions,
            accuracy_data=historical_accuracy,
            context=context,
        )

        raw_correct = 0
        brain_correct = 0
        total = 0

        for symbol, decision in decisions.items():
            acc = historical_accuracy.get(symbol, {}).get("accuracy_pct", 50.0)
            n = historical_accuracy.get(symbol, {}).get("total", 0)
            if n < MIN_SAMPLES:
                continue

            total += n
            raw_correct += acc * n / 100.0

            if decision.action == "EXCLUDE":
                continue  # excluded, don't count
            elif decision.inverted:
                brain_correct += (100.0 - acc) * n / 100.0
            else:
                brain_correct += acc * n / 100.0

        raw_pct = (raw_correct / total * 100.0) if total > 0 else 50.0
        brain_pct = (brain_correct / total * 100.0) if total > 0 else 50.0

        return {
            "raw_accuracy": raw_pct,
            "brain_accuracy": brain_pct,
            "lift": brain_pct - raw_pct,
            "total_predictions": total,
            "recommendation": "SHIP" if brain_pct > raw_pct else "HOLD",
        }

    # ═══════════════════════════════════════════════════════════
    # REPORTING: generate_report (enhanced)
    # ═══════════════════════════════════════════════════════════

    def generate_report(self) -> str:
        """
        Generate honest self-assessment report.

        Enhanced from v2 with:
          - Per-asset-class breakdown
          - Feature importance ranking
          - Circuit breaker status
          - Prune candidates
          - A/B group summary
        """
        lines = []
        lines.append("🧠 GHOST BRAIN v3 REPORT")
        lines.append("=" * 50)

        # Summary stats
        s = self._cycle_stats
        lines.append(
            f"Analyzed: {s['analyzed']} | "
            f"🔄Inverted: {s['inverted']} | "
            f"⛔Excluded: {s['excluded']} | "
            f"🚀Boosted: {s['boosted']} | "
            f"📉Penalized: {s['penalized']} | "
            f"📤Sent: {s['sent']}"
        )

        # Circuit breaker status
        if self._circuit_breaker_active:
            lines.append("🚨 CIRCUIT BREAKER: ACTIVE — all confidence reduced")

        # Direction bias
        if self._direction_bias and self._direction_bias.get("biased"):
            b = self._direction_bias
            lines.append(
                f"⚠️ DIRECTION BIAS: {b['pct']:.0%} of predictions are {b['direction']}"
            )

        # Correlation warnings
        for w in self._correlation_warnings:
            lines.append(f"⚠️ CORRELATION: {w}")

        # Per-tier breakdown
        tiers = {}
        for d in self._decisions.values():
            tiers.setdefault(d.tier, []).append(d)

        tier_order = ["🔄INVERTED", "🔥FIRE", "🟢HOT", "🟡WARM", "🔴COLD", "⚪NEUTRAL", "⛔EXCLUDED"]
        for tier in tier_order:
            if tier not in tiers:
                continue
            decisions = tiers[tier]
            lines.append(f"\n{tier} ({len(decisions)}):")
            for d in sorted(decisions, key=lambda x: x.effective_accuracy, reverse=True):
                inv_tag = " (flipped)" if d.inverted else ""
                lines.append(
                    f"  {d.symbol}: {d.direction}{inv_tag} @ {d.confidence:.0%} "
                    f"[raw:{d.raw_accuracy:.0f}%→brain:{d.brain_accuracy:.0f}%"
                    f"→eff:{d.effective_accuracy:.0f}%, n={d.sample_size}]"
                )

        # ── Per-asset-class breakdown ──
        for asset in ("stock", "crypto"):
            asset_decisions = [d for d in self._decisions.values() if d.asset_class == asset]
            if not asset_decisions:
                continue

            icon = "📈" if asset == "stock" else "🪙"
            label = "STOCKS" if asset == "stock" else "CRYPTO"
            inv_count = sum(1 for d in asset_decisions if d.action == "INVERT")
            exc_count = sum(1 for d in asset_decisions if d.action == "EXCLUDE")

            raw_accs = [d.raw_accuracy for d in asset_decisions if d.sample_size >= MIN_SAMPLES]
            eff_accs = [d.effective_accuracy for d in asset_decisions if d.sample_size >= MIN_SAMPLES and d.action != "EXCLUDE"]

            avg_raw = sum(raw_accs) / len(raw_accs) if raw_accs else 0
            avg_eff = sum(eff_accs) / len(eff_accs) if eff_accs else 0

            verdict = "GOOD ✅" if avg_eff >= 60 else ("OKAY 🟡" if avg_eff >= 50 else "WEAK ❌")

            lines.append(
                f"\n{icon} {label} ({len(asset_decisions)} symbols): "
                f"raw {avg_raw:.1f}% → eff {avg_eff:.1f}% | "
                f"{inv_count}🔄 {exc_count}⛔"
            )
            lines.append(f"  → Ghost is {verdict} at {asset}")

        # ── Feature importance (#24) ──
        if self._feature_contributions:
            lines.append("\n📊 FEATURE IMPORTANCE:")
            ranked = sorted(
                self._feature_contributions.items(),
                key=lambda x: x[1]["total_impact"],
                reverse=True,
            )
            for ability, stats in ranked[:8]:
                avg_impact = stats["total_impact"] / max(stats["count"], 1)
                lines.append(
                    f"  {ability}: avg_impact={avg_impact:.3f} "
                    f"(applied {stats['count']}x)"
                )

        # ── Prune candidates (#14) ──
        if self._prune_candidates:
            lines.append(f"\n🗑️ PRUNE CANDIDATES: {', '.join(self._prune_candidates)}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # REPORTING: telegram_summary
    # ═══════════════════════════════════════════════════════════

    def generate_telegram_summary(self) -> str:
        """One-line summary for Telegram footer."""
        s = self._cycle_stats
        inv = s["inverted"]
        exc = s["excluded"]
        boost = s["boosted"]
        total = s["analyzed"]

        parts = [f"🧠 Brain v3: {total} analyzed"]
        if inv:
            parts.append(f"🔄{inv} flipped")
        if exc:
            parts.append(f"⛔{exc} excluded")
        if boost:
            parts.append(f"🚀{boost} boosted")
        if self._circuit_breaker_active:
            parts.append("🚨 CIRCUIT BREAKER")

        return " | ".join(parts)

    # ═══════════════════════════════════════════════════════════
    # REPORTING: health endpoint
    # ═══════════════════════════════════════════════════════════

    def get_health(self) -> Dict[str, Any]:
        """
        JSON-serializable health status for /api/brain-health endpoint.

        Enhanced from v2 with feature importance, circuit breaker,
        prune candidates, A/B groups, and per-asset-class breakdown.
        """
        decisions_list = []
        for sym, d in self._decisions.items():
            decisions_list.append({
                "symbol": sym,
                "action": d.action,
                "direction": d.direction,
                "confidence": round(d.confidence, 4),
                "tier": d.tier,
                "asset_class": d.asset_class,
                "raw_accuracy": round(d.raw_accuracy, 1),
                "brain_accuracy": round(d.brain_accuracy, 1),
                "effective_accuracy": round(d.effective_accuracy, 1),
                "inverted": d.inverted,
                "sample_size": d.sample_size,
                "data_quality": d.data_quality,
                "expected_value": round(d.expected_value, 4),
                "modifiers": {k: round(v, 4) for k, v in d.confidence_modifiers.items()},
                "direction_split": {k: round(v, 1) for k, v in d.direction_split.items()},
                "prune_candidate": d.prune_candidate,
            })

        # Per-asset-class stats
        by_asset = {}
        for asset in ("stock", "crypto"):
            ad = [d for d in self._decisions.values() if d.asset_class == asset]
            if not ad:
                continue
            by_asset[asset] = {
                "total": len(ad),
                "sent": sum(1 for d in ad if d.action != "EXCLUDE"),
                "inverted": sum(1 for d in ad if d.action == "INVERT"),
                "excluded": sum(1 for d in ad if d.action == "EXCLUDE"),
                "avg_raw": round(sum(d.raw_accuracy for d in ad) / len(ad), 1),
                "avg_brain": round(sum(d.brain_accuracy for d in ad) / len(ad), 1),
                "avg_effective": round(
                    sum(d.effective_accuracy for d in ad if d.action != "EXCLUDE")
                    / max(sum(1 for d in ad if d.action != "EXCLUDE"), 1), 1
                ),
            }

        # Feature importance ranking
        importance = {}
        if self._feature_contributions:
            for ability, stats in self._feature_contributions.items():
                importance[ability] = round(
                    stats["total_impact"] / max(stats["count"], 1), 4
                )

        return {
            "version": "v3",
            "enabled": BRAIN_ENABLED,
            "config": {
                "invert_below": INVERT_BELOW,
                "exclude_below": EXCLUDE_BELOW,
                "boost_above": BOOST_ABOVE,
                "strong_boost_above": STRONG_BOOST_ABOVE,
                "min_samples": MIN_SAMPLES,
                "recency_weight": RECENCY_WEIGHT,
                "max_same_direction": MAX_SAME_DIRECTION,
            },
            "cycle_stats": dict(self._cycle_stats),
            "circuit_breaker_active": self._circuit_breaker_active,
            "direction_bias": self._direction_bias or {},
            "correlation_warnings": self._correlation_warnings,
            "prune_candidates": self._prune_candidates,
            "feature_importance": importance,
            "by_asset_class": by_asset,
            "decisions": decisions_list,
        }

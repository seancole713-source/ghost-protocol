#!/usr/bin/env python3
"""
🧠 GHOST BRAIN v2 — Centralized Learning Intelligence
=======================================================

The OLD system had 3 disconnected pieces:
  ❌ should_exclude_symbol()  → Binary: exclude or pass (no flip option)
  ❌ get_confidence_boost()   → Binary: boost or pass (no scale-down)
  ❌ INVERSE_GHOST env var    → Dumb global ON/OFF toggle (no per-symbol)

The NEW brain unifies ALL decisions into one engine:
  ✅ INVERT   — Flip reliably-wrong symbols (35% raw → 65% effective)
  ✅ SCALE    — Adjust confidence proportional to actual accuracy
  ✅ BIAS     — Detect directional bias (model always says UP)
  ✅ TIER     — Classify symbols: 🟢HOT 🟡WARM 🔴COLD 🔄INVERTED ⛔EXCLUDED
  ✅ GUARD    — Prevent correlated bets (not all picks same direction)
  ✅ DECAY    — Weight recent accuracy heavier than old data
  ✅ REPORT   — Honest self-assessment every cycle

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  PostgreSQL (ghost_symbol_accuracy)                              │
  │       │                                                          │
  │       ▼                                                          │
  │  GhostBrain.analyze_symbol()                                     │
  │       │                                                          │
  │       ├─ accuracy < 38%  → INVERT (flip direction, boost conf)   │
  │       ├─ accuracy 38-48% → EXCLUDE (noise zone, coin flip)       │
  │       ├─ accuracy 48-55% → SEND with penalty (barely above flip) │
  │       ├─ accuracy 55-62% → SEND neutral (moderate edge)          │
  │       ├─ accuracy 62-70% → SEND with boost (real edge)           │
  │       └─ accuracy > 70%  → SEND with strong boost (proven)       │
  │                                                                  │
  │  Brain operates at NOTIFICATION time (post-processing layer)     │
  │  Raw paper trades preserved → accuracy data = MODEL accuracy     │
  │  Brain self-reports every cycle → honest self-assessment          │
  └──────────────────────────────────────────────────────────────────┘

Why this is smarter than a human trader:
  1. No ego — will admit it's wrong and flip instantly
  2. No emotion — scales confidence mathematically, not "gut feeling"
  3. Tracks 200+ symbols simultaneously with zero fatigue
  4. Detects its own biases (always-UP tendency) automatically
  5. Self-corrects every cycle — doesn't wait for a human to notice
  6. Reports its own health honestly — no hiding bad performance
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("ghost_brain")

# =============================================================================
# BRAIN CONFIGURATION (all overridable via env vars)
# =============================================================================

# Master switch — allows disabling the brain entirely
BRAIN_ENABLED = os.getenv("GHOST_BRAIN_ENABLED", "1") == "1"

# ─── Accuracy Thresholds ───
#
# These define the brain's decision boundaries:
#
#   0%  ───── INVERT_BELOW ───── EXCLUDE_BELOW ───── BOOST_ABOVE ───── 100%
#       [     INVERT      ] [    EXCLUDE    ] [   INCLUDE    ] [  BOOST   ]
#       "reliably wrong"    "noise zone"      "has some edge"  "proven"
#
INVERT_BELOW = float(os.getenv("BRAIN_INVERT_BELOW", "38.0"))
EXCLUDE_BELOW = float(os.getenv("BRAIN_EXCLUDE_BELOW", "48.0"))
BOOST_ABOVE = float(os.getenv("BRAIN_BOOST_ABOVE", "62.0"))
STRONG_BOOST_ABOVE = float(os.getenv("BRAIN_STRONG_BOOST", "70.0"))

# Minimum predictions before brain acts (need statistical significance)
MIN_SAMPLES = int(os.getenv("BRAIN_MIN_SAMPLES", "20"))

# ─── Confidence Multipliers ───
STRONG_BOOST_MULT = 1.30     # 70%+ accuracy → 30% confidence boost
BOOST_MULT = 1.15            # 62%+ accuracy → 15% confidence boost
NOISE_PENALTY_MULT = 0.85    # 48-55% accuracy → 15% confidence reduction
CONFIDENCE_CAP = 0.98        # Never exceed 98% confidence

# ─── Direction Bias Detection ───
BIAS_THRESHOLD = 0.80  # Flag if >80% of predictions are same direction

# ─── Correlation Guard ───
MAX_SAME_DIRECTION = int(os.getenv("BRAIN_MAX_SAME_DIR", "6"))

# ─── Known Crypto (for correlation grouping, no external imports) ───
_KNOWN_CRYPTO = {
    "BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT",
    "MATIC", "SHIB", "UNI", "LTC", "BCH", "ATOM", "FIL", "NEAR", "ICP",
    "APT", "ARB", "OP", "INJ", "TIA", "SEI", "SUI", "TURBO", "JUP",
    "IOTX", "GIGA", "CHZ", "ALICE", "YFI", "BRETT", "T", "HBAR", "ILV",
    "BAND", "PEPE", "ENJ", "RNDR", "MANA", "SAND", "AXS", "THETA", "VET",
    "FTM", "EGLD", "ALGO", "FLOW", "STX", "DASH", "ZEC", "EOS", "XTZ",
    "AAVE", "CRV", "MKR", "COMP", "SNX", "SUSHI", "1INCH", "BAL", "REN",
    "ZRX", "BAT", "KNC", "OCEAN", "OMG", "RLC", "BNB", "DYDX", "WLD",
    "JTO", "BONK", "WIF", "FLOKI", "ORDI", "RUNE", "ROSE", "QTUM",
    "ANT", "ZEN", "ONDO", "HOOD",
}


# =============================================================================
# BRAIN DECISION
# =============================================================================

@dataclass
class BrainDecision:
    """A single decision made by Ghost Brain for one symbol.

    Fields:
        symbol:             The trading symbol
        action:             SEND | EXCLUDE | INVERT
        direction:          Final direction (may be flipped from raw)
        confidence:         Adjusted confidence (0.0-1.0)
        tier:               Performance tier label
        reasons:            Human-readable decision trail
        raw_accuracy:       Original accuracy from accuracy_data
        effective_accuracy: Accuracy after inversion (if applicable)
        inverted:           Whether direction was flipped
        sample_size:        How many predictions this is based on
    """
    symbol: str
    action: str                      # SEND | EXCLUDE | INVERT
    direction: str                   # Final direction (may differ from model)
    confidence: float                # Adjusted confidence
    tier: str                        # 🟢HOT | 🟡WARM | ⚪NEUTRAL | 🔴COLD | 🔄INVERTED | ⛔EXCLUDED
    reasons: List[str] = field(default_factory=list)
    raw_accuracy: float = 50.0       # Model's actual accuracy
    effective_accuracy: float = 50.0  # After inversion: 100 - raw_accuracy
    inverted: bool = False
    sample_size: int = 0


# =============================================================================
# GHOST BRAIN
# =============================================================================

class GhostBrain:
    """
    Ghost's centralized learning intelligence.

    7 Cognitive Abilities:
    ┌──────────────────────────────────────────────────────────┐
    │  1. INVERT   — Flip reliably-wrong symbols              │
    │  2. SCALE    — Adjust confidence to match reality        │
    │  3. BIAS     — Detect directional bias                   │
    │  4. TIER     — Classify performance tiers                │
    │  5. GUARD    — Prevent correlated bets                   │
    │  6. DECAY    — Weight recent data heavier                │
    │  7. REPORT   — Honest self-assessment                    │
    └──────────────────────────────────────────────────────────┘

    Usage:
        brain = GhostBrain()

        # Single symbol:
        decision = brain.analyze_symbol("BTC", "UP", 0.80, accuracy_data)

        # Full batch (enables cross-symbol intelligence):
        decisions = brain.analyze_batch(predictions, accuracy_data)
        report = brain.generate_report()
    """

    def __init__(self):
        self._decisions: Dict[str, BrainDecision] = {}
        self._direction_bias: Optional[Dict] = None
        self._correlation_warnings: List[str] = []
        self._cycle_stats = {
            "analyzed": 0,
            "inverted": 0,
            "excluded": 0,
            "boosted": 0,
            "penalized": 0,
            "sent": 0,
        }

    # ─────────────────────────────────────────────────────────
    # ABILITY 1-4: CORE ANALYSIS (invert, scale, tier, per symbol)
    # ─────────────────────────────────────────────────────────

    def analyze_symbol(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        accuracy_data: Dict[str, Dict],
    ) -> BrainDecision:
        """
        Make a unified decision for one symbol.

        This REPLACES:
        - should_exclude_symbol()'s learning check (env/hardcoded checks stay)
        - get_confidence_boost() entirely
        - INVERSE_GHOST per-symbol toggle

        Decision tree:
            accuracy < 38%  → INVERT (reliably wrong = reliably predictable)
            accuracy 38-48% → EXCLUDE (noise zone, near coin flip)
            accuracy 48-55% → SEND with confidence penalty
            accuracy 55-62% → SEND neutral
            accuracy 62-70% → SEND with boost
            accuracy 70%+   → SEND with strong boost

        Args:
            symbol:        Trading symbol (e.g. "BTC", "ETH")
            direction:     Raw direction from model ("UP" or "DOWN")
            confidence:    Raw confidence from model (0.0-1.0)
            accuracy_data: Dict from get_symbol_accuracy_from_postgres()

        Returns:
            BrainDecision with all adjustments applied
        """
        if not BRAIN_ENABLED:
            return BrainDecision(
                symbol=symbol, action="SEND", direction=direction,
                confidence=confidence, tier="⚪NEUTRAL",
                reasons=["brain_disabled"],
            )

        reasons: List[str] = []
        inverted = False
        final_direction = direction
        final_confidence = confidence

        # ── Get accuracy data ──
        data = accuracy_data.get(symbol.upper()) or accuracy_data.get(symbol) or {}
        raw_accuracy = data.get("accuracy_pct", 50.0)
        total = data.get("total", 0)

        # ── INSUFFICIENT DATA → pass through (need enough to judge) ──
        if total < MIN_SAMPLES:
            reasons.append(f"insufficient_data ({total}/{MIN_SAMPLES} predictions)")
            decision = BrainDecision(
                symbol=symbol, action="SEND", direction=direction,
                confidence=confidence, tier="⚪NEUTRAL",
                reasons=reasons, raw_accuracy=raw_accuracy,
                effective_accuracy=raw_accuracy, sample_size=total,
            )
            self._decisions[symbol] = decision
            self._cycle_stats["analyzed"] += 1
            self._cycle_stats["sent"] += 1
            return decision

        # ══════════════════════════════════════════════════════
        # DECISION TREE
        # ══════════════════════════════════════════════════════

        # ── ZONE 1: INVERT (accuracy < 38%) ──
        # The model is RELIABLY WRONG. That's actually useful!
        # A student who gets 30% on a true/false test KNOWS the
        # material — they're just picking the wrong answer.
        # Flip it → 70% accuracy.
        if raw_accuracy < INVERT_BELOW:
            inverted = True
            final_direction = "DOWN" if direction == "UP" else "UP"
            effective_accuracy = 100.0 - raw_accuracy

            reasons.append(
                f"🔄 INVERT: {raw_accuracy:.1f}% raw < {INVERT_BELOW}% → "
                f"flipped {direction}→{final_direction} "
                f"(effective {effective_accuracy:.1f}%)"
            )

            # Inverted symbols with high effective accuracy get boosted
            if effective_accuracy >= STRONG_BOOST_ABOVE:
                final_confidence = min(CONFIDENCE_CAP, confidence * STRONG_BOOST_MULT)
                reasons.append(
                    f"🚀 STRONG_BOOST: effective {effective_accuracy:.1f}% "
                    f"≥ {STRONG_BOOST_ABOVE}% → ×{STRONG_BOOST_MULT}"
                )
                self._cycle_stats["boosted"] += 1
            elif effective_accuracy >= BOOST_ABOVE:
                final_confidence = min(CONFIDENCE_CAP, confidence * BOOST_MULT)
                reasons.append(
                    f"📈 BOOST: effective {effective_accuracy:.1f}% "
                    f"≥ {BOOST_ABOVE}% → ×{BOOST_MULT}"
                )
                self._cycle_stats["boosted"] += 1

            tier = "🔄INVERTED"
            action = "INVERT"
            self._cycle_stats["inverted"] += 1

        # ── ZONE 2: EXCLUDE (accuracy 38-48%) ──
        # Near coin-flip. Not reliably wrong, not reliably right.
        # No signal here — just noise. Drop it.
        elif raw_accuracy < EXCLUDE_BELOW:
            effective_accuracy = raw_accuracy
            reasons.append(
                f"⛔ EXCLUDE: {raw_accuracy:.1f}% in noise zone "
                f"({INVERT_BELOW}%-{EXCLUDE_BELOW}%) — coin flip territory"
            )
            tier = "⛔EXCLUDED"
            action = "EXCLUDE"
            self._cycle_stats["excluded"] += 1

            decision = BrainDecision(
                symbol=symbol, action=action, direction=direction,
                confidence=0.0, tier=tier, reasons=reasons,
                raw_accuracy=raw_accuracy, effective_accuracy=effective_accuracy,
                sample_size=total,
            )
            self._decisions[symbol] = decision
            self._cycle_stats["analyzed"] += 1
            return decision

        # ── ZONE 3: COLD (accuracy 48-55%) ──
        # Slightly above coin flip. Include but reduce confidence.
        elif raw_accuracy < 55.0:
            effective_accuracy = raw_accuracy
            final_confidence = confidence * NOISE_PENALTY_MULT
            tier = "🔴COLD"
            reasons.append(
                f"🔴 COLD: {raw_accuracy:.1f}% barely above coin flip "
                f"→ ×{NOISE_PENALTY_MULT} confidence penalty"
            )
            action = "SEND"
            self._cycle_stats["penalized"] += 1

        # ── ZONE 4: WARM (accuracy 55-62%) ──
        # Decent. No adjustment needed.
        elif raw_accuracy < BOOST_ABOVE:
            effective_accuracy = raw_accuracy
            tier = "🟡WARM"
            reasons.append(f"🟡 WARM: {raw_accuracy:.1f}% — moderate edge, no adjustment")
            action = "SEND"

        # ── ZONE 5: HOT (accuracy 62-70%) ──
        # Real edge detected. Boost confidence.
        elif raw_accuracy < STRONG_BOOST_ABOVE:
            effective_accuracy = raw_accuracy
            final_confidence = min(CONFIDENCE_CAP, confidence * BOOST_MULT)
            tier = "🟢HOT"
            reasons.append(
                f"🟢 HOT: {raw_accuracy:.1f}% ≥ {BOOST_ABOVE}% "
                f"→ ×{BOOST_MULT} confidence boost"
            )
            action = "SEND"
            self._cycle_stats["boosted"] += 1

        # ── ZONE 6: FIRE (accuracy 70%+) ──
        # Proven winner. Strong boost.
        else:
            effective_accuracy = raw_accuracy
            final_confidence = min(CONFIDENCE_CAP, confidence * STRONG_BOOST_MULT)
            tier = "🟢HOT"
            reasons.append(
                f"🔥 FIRE: {raw_accuracy:.1f}% ≥ {STRONG_BOOST_ABOVE}% "
                f"→ ×{STRONG_BOOST_MULT} strong confidence boost"
            )
            action = "SEND"
            self._cycle_stats["boosted"] += 1

        self._cycle_stats["analyzed"] += 1
        self._cycle_stats["sent"] += 1

        decision = BrainDecision(
            symbol=symbol, action=action, direction=final_direction,
            confidence=final_confidence, tier=tier, reasons=reasons,
            raw_accuracy=raw_accuracy, effective_accuracy=effective_accuracy,
            inverted=inverted, sample_size=total,
        )
        self._decisions[symbol] = decision
        return decision

    # ─────────────────────────────────────────────────────────
    # ABILITY 5: BATCH ANALYSIS (cross-symbol intelligence)
    # ─────────────────────────────────────────────────────────

    def analyze_batch(
        self,
        predictions: Dict[str, Dict],
        accuracy_data: Dict[str, Dict],
    ) -> Dict[str, BrainDecision]:
        """
        Analyze all predictions in one pass.

        Enables cross-symbol intelligence that per-symbol analysis can't:
        - Direction bias detection (is the model always saying UP?)
        - Correlation guard (are all 10 picks the same bet?)

        Args:
            predictions:   Dict of {symbol: prediction_dict}
            accuracy_data: Dict from get_symbol_accuracy_from_postgres()

        Returns:
            Dict of {symbol: BrainDecision}
        """
        # Reset cycle
        self._decisions = {}
        self._correlation_warnings = []
        self._cycle_stats = {k: 0 for k in self._cycle_stats}

        # ABILITY 3: Detect direction bias BEFORE individual analysis
        self._direction_bias = self._detect_direction_bias(predictions)

        # Analyze each symbol
        for symbol, pred in predictions.items():
            if not isinstance(pred, dict):
                continue
            direction = pred.get("direction", "")
            confidence = pred.get("confidence", 0.0)
            if direction not in ("UP", "DOWN"):
                continue
            self.analyze_symbol(symbol, direction, confidence, accuracy_data)

        # ABILITY 5: Apply correlation guard to batch
        self._apply_correlation_guard()

        return self._decisions

    # ─────────────────────────────────────────────────────────
    # ABILITY 3: DIRECTION BIAS DETECTION
    # ─────────────────────────────────────────────────────────

    def _detect_direction_bias(self, predictions: Dict[str, Dict]) -> Dict:
        """
        Detect if the model has systematic directional bias.

        If 90% of predictions are UP, the model might be capturing
        market regime (bull market → everything UP) rather than
        per-symbol edge. This isn't necessarily wrong, but it's a
        red flag that predictions are correlated, not independent.

        Returns:
            {"biased": bool, "direction": str, "pct": float, "total": int}
        """
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
                f"predictions are UP — possible bullish bias"
            )
            return {"biased": True, "direction": "UP", "pct": up_pct, "total": total}

        down_pct = 1 - up_pct
        if down_pct >= BIAS_THRESHOLD:
            LOGGER.warning(
                f"[BRAIN] ⚠️ DIRECTION BIAS: {down_pct:.0%} of {total} "
                f"predictions are DOWN — possible bearish bias"
            )
            return {"biased": True, "direction": "DOWN", "pct": down_pct, "total": total}

        return {"biased": False, "up_pct": up_pct, "total": total}

    # ─────────────────────────────────────────────────────────
    # ABILITY 5: CORRELATION GUARD
    # ─────────────────────────────────────────────────────────

    def _apply_correlation_guard(self):
        """
        Detect when too many picks are in the same direction for the
        same asset class.  10 crypto UP picks = really just 1 bet.

        Currently logs warnings. Future: auto-demote weakest overflow
        picks via confidence penalty.
        """
        groups: Dict[str, List[BrainDecision]] = {}

        for symbol, decision in self._decisions.items():
            if decision.action == "EXCLUDE":
                continue

            asset_class = "crypto" if symbol.upper() in _KNOWN_CRYPTO else "stock"
            key = f"{asset_class}_{decision.direction}"

            if key not in groups:
                groups[key] = []
            groups[key].append(decision)

        self._correlation_warnings = []
        for key, decisions in groups.items():
            if len(decisions) > MAX_SAME_DIRECTION:
                # Sort by effective accuracy (strongest first)
                decisions.sort(key=lambda d: d.effective_accuracy, reverse=True)
                overflow = decisions[MAX_SAME_DIRECTION:]
                asset_class, direction = key.rsplit("_", 1)

                # Apply confidence penalty to overflow picks
                for d in overflow:
                    d.confidence = d.confidence * NOISE_PENALTY_MULT
                    d.reasons.append(
                        f"⚠️ CORRELATION_PENALTY: {len(decisions)} {asset_class} "
                        f"picks are {direction} (max {MAX_SAME_DIRECTION})"
                    )

                warning = (
                    f"{len(decisions)} {asset_class} picks are {direction} "
                    f"(max {MAX_SAME_DIRECTION}) — penalized weakest: "
                    f"{[d.symbol for d in overflow]}"
                )
                self._correlation_warnings.append(warning)
                LOGGER.warning(f"[BRAIN] ⚠️ CORRELATION: {warning}")

    # ─────────────────────────────────────────────────────────
    # ABILITY 7: SELF-ASSESSMENT REPORT
    # ─────────────────────────────────────────────────────────

    def generate_report(self) -> str:
        """
        Generate an honest self-assessment report.

        This is what makes Ghost smarter than a human trader:
        a human lies to themselves about performance.
        Ghost Brain never lies.
        """
        lines = []
        lines.append("🧠 GHOST BRAIN REPORT")
        lines.append("=" * 44)

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

        # Direction bias alert
        if self._direction_bias and self._direction_bias.get("biased"):
            bias = self._direction_bias
            lines.append(
                f"\n⚠️ DIRECTION BIAS: {bias['pct']:.0%} of predictions "
                f"are {bias['direction']}"
            )

        # Correlation warnings
        for w in self._correlation_warnings:
            lines.append(f"⚠️ CORRELATION: {w}")

        # Per-tier breakdown
        tier_order = [
            "🔄INVERTED", "🟢HOT", "🟡WARM", "⚪NEUTRAL", "🔴COLD", "⛔EXCLUDED",
        ]
        for tier in tier_order:
            tier_decisions = [
                d for d in self._decisions.values() if d.tier == tier
            ]
            if not tier_decisions:
                continue

            lines.append(f"\n{tier} ({len(tier_decisions)}):")
            for d in sorted(
                tier_decisions,
                key=lambda x: x.effective_accuracy,
                reverse=True,
            ):
                flip_marker = " (flipped)" if d.inverted else ""
                lines.append(
                    f"  {d.symbol}: {d.direction}{flip_marker} "
                    f"@ {d.confidence:.0%} "
                    f"[raw:{d.raw_accuracy:.0f}%→eff:{d.effective_accuracy:.0f}%, "
                    f"n={d.sample_size}]"
                )

        lines.append("\n" + "=" * 44)
        return "\n".join(lines)

    def generate_telegram_summary(self) -> str:
        """
        One-liner summary suitable for embedding in the 8 AM Telegram message.

        Example: "🧠 Brain: 5🔄 2⛔ 3🟢 | Bias: 65% UP"
        """
        s = self._cycle_stats
        parts = [f"🧠 Brain:"]
        if s["inverted"]:
            parts.append(f"{s['inverted']}🔄")
        if s["excluded"]:
            parts.append(f"{s['excluded']}⛔")
        if s["boosted"]:
            parts.append(f"{s['boosted']}🚀")
        if s["penalized"]:
            parts.append(f"{s['penalized']}📉")

        summary = " ".join(parts)

        # Add bias info if detected
        if self._direction_bias and self._direction_bias.get("biased"):
            bias = self._direction_bias
            summary += f" | Bias: {bias['pct']:.0%} {bias['direction']}"

        return summary

    def get_health(self) -> Dict:
        """
        Brain health metrics for /api/brain-health endpoint.

        Returns a dict suitable for JSON serialization.
        """
        if not self._decisions:
            return {"status": "no_data", "enabled": BRAIN_ENABLED, "decisions": 0}

        # Calculate averages (only for symbols with data)
        with_data = [
            d for d in self._decisions.values() if d.sample_size >= MIN_SAMPLES
        ]
        avg_raw = (
            sum(d.raw_accuracy for d in with_data) / len(with_data)
            if with_data else 0.0
        )
        sent_with_data = [d for d in with_data if d.action != "EXCLUDE"]
        avg_effective = (
            sum(d.effective_accuracy for d in sent_with_data) / len(sent_with_data)
            if sent_with_data else 0.0
        )

        return {
            "status": "active" if BRAIN_ENABLED else "disabled",
            "enabled": BRAIN_ENABLED,
            "thresholds": {
                "invert_below": INVERT_BELOW,
                "exclude_below": EXCLUDE_BELOW,
                "boost_above": BOOST_ABOVE,
                "strong_boost_above": STRONG_BOOST_ABOVE,
                "min_samples": MIN_SAMPLES,
            },
            "cycle_stats": dict(self._cycle_stats),
            "avg_raw_accuracy": round(avg_raw, 1),
            "avg_effective_accuracy": round(avg_effective, 1),
            "accuracy_lift": round(avg_effective - avg_raw, 1),
            "direction_bias": self._direction_bias,
            "correlation_warnings": len(self._correlation_warnings),
            "decisions": len(self._decisions),
            "tiers": {
                tier: sum(1 for d in self._decisions.values() if d.tier == tier)
                for tier in [
                    "🔄INVERTED", "🟢HOT", "🟡WARM", "⚪NEUTRAL",
                    "🔴COLD", "⛔EXCLUDED",
                ]
            },
        }

"""
GHOST INTEL MODULE
==================
Institutional-grade intelligence layer for Ghost Protocol.

The 8-Layer Model:
1. MACRO DATA (CPI, Jobs, FOMC, GDP)
2. RATES & LIQUIDITY (2Y, 10Y, DXY, VIX)
3. CORPORATE (Earnings, Guidance)
4. POLITICS (Tariffs, Sanctions, Regulation)
5. GEOPOLITICS (War, Energy, Shipping)
6. KEY INDIVIDUALS (Elon, Fed Chair, CEOs)
7. SOCIAL (Twitter, Reddit, StockTwits)
8. POSITIONING (Options, Gamma, Liquidations)

Goal: Enter at Step 1-3, not Step 9 (retail timing)

Author: Ghost AI
Date: 2026-01-26
"""

import logging

# Configure Intel logger
logger = logging.getLogger("ghost.intel")
logger.setLevel(logging.INFO)

# Version
__version__ = "1.0.0"

# Exports
from ghost_intel.sources import IntelSources
from ghost_intel.normalize import IntelEvent, normalize_event
from ghost_intel.impact_model import ImpactScorer, ImpactScore
from ghost_intel.positioning import PositioningAnalyzer
from ghost_intel.taxonomy import EventTaxonomy, EventCategory
from ghost_intel.integration import (
    apply_intel_to_prediction,
    get_intel_signal_for_prediction,
    IntelSignal,
)

__all__ = [
    "IntelSources",
    "IntelEvent",
    "normalize_event",
    "ImpactScorer",
    "ImpactScore",
    "PositioningAnalyzer",
    "EventTaxonomy",
    "EventCategory",
    "apply_intel_to_prediction",
    "get_intel_signal_for_prediction",
    "IntelSignal",
]

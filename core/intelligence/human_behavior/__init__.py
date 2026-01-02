"""
Ghost Protocol - Human Behavior Module
Understand WHY humans buy and sell
"""

from .narrative_detector import detect_narratives, get_narrative_detector
from .influencer_tracker import check_influencers, get_influencer_tracker

__all__ = [
    "detect_narratives",
    "get_narrative_detector",
    "check_influencers",
    "get_influencer_tracker",
]

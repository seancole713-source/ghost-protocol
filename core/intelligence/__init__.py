"""
Ghost Protocol - Intelligence Module
Central hub for market intelligence
"""

from .ghost_brain import analyze_with_intelligence, get_ghost_brain
from .opus_brain import (
    opus_analyze, 
    opus_research, 
    opus_explain, 
    opus_compare,
    get_opus_brain
)
from .ghost_news_brain import (
    GhostNewsBrain,
    get_news_brain,
    analyze_breaking_news,
)

__all__ = [
    "analyze_with_intelligence",
    "get_ghost_brain",
    "opus_analyze",
    "opus_research",
    "opus_explain",
    "opus_compare",
    "get_opus_brain",
    "GhostNewsBrain",
    "get_news_brain",
    "analyze_breaking_news",
]

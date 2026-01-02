"""
Ghost Protocol - Research Module
Comprehensive research for smarter predictions
"""

from .deep_researcher import deep_research, batch_research, get_researcher
from .earnings_calendar import check_earnings_risk, get_earnings_calendar
from .news_analyzer import analyze_news, get_news_analyzer
from .seasonal_patterns import analyze_seasonal, get_seasonal_analyzer
from .historical_analyzer import analyze_historical, get_historical_analyzer
from .integration import (
    get_research_enhancement,
    apply_research_adjustment,
    should_skip_prediction,
    RESEARCH_ENABLED
)

__all__ = [
    "deep_research",
    "batch_research",
    "get_researcher",
    "check_earnings_risk",
    "get_earnings_calendar",
    "analyze_news",
    "get_news_analyzer",
    "analyze_seasonal",
    "get_seasonal_analyzer",
    "analyze_historical",
    "get_historical_analyzer",
    "get_research_enhancement",
    "apply_research_adjustment",
    "should_skip_prediction",
    "RESEARCH_ENABLED",
]

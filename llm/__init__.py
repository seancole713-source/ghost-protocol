"""LLM modules for Ghost Protocol"""

from .gpt4_analyst import (
    is_enabled,
    analyze_prediction,
    get_market_commentary,
    get_status,
)

__all__ = [
    "is_enabled",
    "analyze_prediction",
    "get_market_commentary",
    "get_status",
]

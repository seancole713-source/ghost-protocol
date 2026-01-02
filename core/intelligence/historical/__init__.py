"""
Ghost Protocol - Historical Intelligence Module
Learn from what happened before
"""

from .event_outcomes import (
    get_historical_outcomes, 
    what_happened_last_time, 
    get_event_database
)

__all__ = [
    "get_historical_outcomes",
    "what_happened_last_time",
    "get_event_database",
]

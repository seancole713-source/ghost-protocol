"""
core/beast_scheduler.py — Compatibility stub
═════════════════════════════════════════════

beast_scheduler was superseded by engines/app_config.py and
core/auto_prediction_loop.py in Step 12.  This stub re-exports the
symbols that the rest of the codebase still imports by name, so that
existing `from core.beast_scheduler import …` calls keep working
without modification.

Mutable attributes (REDIS_CLIENT, LOGGER, etc.) remain here as module-
level variables so that orchestrator.py can set them with dot-notation:

    import core.beast_scheduler as beast_scheduler
    beast_scheduler.REDIS_CLIENT = redis_client
    beast_scheduler.start_beast_scheduler()

The actual scheduling is performed by core/auto_prediction_loop.py.
"""

import logging

# ── Re-export symbol lists from canonical config ─────────────────────────
from engines.app_config import (
    STOCK_SYMBOLS,
    CRYPTO_SYMBOLS,
    HUNTER_STOCK_SYMBOLS,
    HUNTER_CRYPTO_SYMBOLS,
    DEFAULT_STOCK_SYMBOLS,
    DEFAULT_CRYPTO_SYMBOLS,
)

# ── Mutable handles set by orchestrator before calling start/stop ─────────
REDIS_CLIENT = None
LOGGER = logging.getLogger("ghost.beast_scheduler")
GET_PRICE_FUNC = None
RUN_PREDICTION_FUNC = None
TELEGRAM_ALERTS_MODULE = None


# ── Lifecycle stubs ───────────────────────────────────────────────────────
def start_beast_scheduler() -> None:
    """No-op: scheduling is now handled by core/auto_prediction_loop.py."""
    LOGGER.info("[beast_scheduler] start_beast_scheduler() called — delegating to auto_prediction_loop")
    try:
        from core.auto_prediction_loop import start_prediction_loop
        start_prediction_loop()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning(f"[beast_scheduler] Could not start auto_prediction_loop: {exc}")


def stop_beast_scheduler() -> None:
    """No-op: scheduling is now handled by core/auto_prediction_loop.py."""
    LOGGER.info("[beast_scheduler] stop_beast_scheduler() called — delegating to auto_prediction_loop")
    try:
        from core.auto_prediction_loop import stop_prediction_loop
        stop_prediction_loop()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning(f"[beast_scheduler] Could not stop auto_prediction_loop: {exc}")

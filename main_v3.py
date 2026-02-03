"""
Application entrypoint with logging configuration.

Run with:
    python main_v3.py
    
Or for development:
    uvicorn main_v3:app --reload --port 8001
"""
import sys
from loguru import logger

from config.settings import settings

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Remove default handler
logger.remove()

# Console logging with color
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    format=log_format,
    level="DEBUG" if settings.DEBUG else "INFO",
    colorize=True,
)

# File logging (optional, only if logs directory exists)
try:
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/ghost_v3_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
except Exception:
    pass  # File logging optional

# =============================================================================
# APP IMPORT (after logging configured)
# =============================================================================

from api.app import app

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"V3 Mode: {'ENABLED' if settings.V3_ENABLED else 'DISABLED'}")
    logger.info(f"Min Confidence: {settings.V3_MIN_CONFIDENCE:.0%}")
    
    uvicorn.run(
        "main_v3:app",
        host="0.0.0.0",
        port=port,
        reload=settings.DEBUG,
    )

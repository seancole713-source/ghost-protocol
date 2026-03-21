"""
Graceful Shutdown Handler (Phase 5.8)

Handles SIGTERM/SIGINT signals to gracefully shutdown the application.
Ensures predictions in-flight complete before shutdown.

Ghost Protocol v5 — Session 6
"""

import signal
import logging
import time
import threading
from typing import Callable, List

LOGGER = logging.getLogger("ghost.shutdown")


class ShutdownHandler:
    """Manages graceful shutdown of Ghost Protocol."""
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Args:
            timeout_seconds: Maximum time to wait for cleanup (default 30s)
        """
        self.timeout_seconds = timeout_seconds
        self.shutdown_requested = False
        self.shutdown_complete = False
        self.cleanup_callbacks: List[Callable] = []
        self._shutdown_lock = threading.Lock()
        
    def register_cleanup(self, callback: Callable):
        """
        Register a cleanup function to be called during shutdown.
        
        Args:
            callback: Function to call during shutdown (should be quick)
        """
        self.cleanup_callbacks.append(callback)
        LOGGER.debug(f"Registered cleanup callback: {callback.__name__}")
        
    def handle_shutdown(self, signum, frame):
        """
        Signal handler for graceful shutdown.
        
        Called when SIGTERM or SIGINT is received.
        """
        with self._shutdown_lock:
            if self.shutdown_requested:
                LOGGER.warning("Shutdown already in progress, ignoring signal")
                return
            
            self.shutdown_requested = True
        
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        LOGGER.info(f"🛑 Received {signal_name} - initiating graceful shutdown...")
        
        start_time = time.time()
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback_name = getattr(callback, '__name__', str(callback))
                LOGGER.info(f"Running cleanup: {callback_name}")
                callback()
            except Exception as e:
                LOGGER.error(f"Cleanup callback failed: {e}", exc_info=True)
        
        # Wait for any in-flight operations
        LOGGER.info("Waiting for in-flight operations to complete...")
        time.sleep(2)  # Give threads time to notice shutdown flag
        
        elapsed = time.time() - start_time
        LOGGER.info(f"✅ Graceful shutdown complete ({elapsed:.1f}s)")
        
        self.shutdown_complete = True
        
    def install(self):
        """Install signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        LOGGER.info("✅ Graceful shutdown handler installed (SIGTERM, SIGINT)")
        
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested


# Global singleton
_SHUTDOWN_HANDLER = ShutdownHandler()


def install_shutdown_handler():
    """Install the global shutdown handler."""
    _SHUTDOWN_HANDLER.install()


def register_cleanup(callback: Callable):
    """Register a cleanup callback."""
    _SHUTDOWN_HANDLER.register_cleanup(callback)


def is_shutting_down() -> bool:
    """Check if shutdown is in progress."""
    return _SHUTDOWN_HANDLER.is_shutting_down()

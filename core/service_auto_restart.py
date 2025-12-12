"""
🔄 SERVICE AUTO-RESTART MODULE
Wraps background tasks with automatic restart logic on failures
Monitors service health and restarts crashed services
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)

# Track service restarts
_SERVICE_RESTART_COUNTS = {}
_SERVICE_LAST_RESTART_TIME = {}
_MAX_RESTARTS_PER_HOUR = 5


async def auto_restart_wrapper(
    service_name: str,
    service_func: Callable,
    *args: Any,
    max_restarts: int = 999,
    restart_delay_seconds: int = 30,
    **kwargs: Any
) -> None:
    """
    Wrapper that automatically restarts a service if it crashes.
    
    Args:
        service_name: Friendly name for logging
        service_func: Async function to run
        *args: Arguments to pass to service_func
        max_restarts: Maximum number of restarts before giving up
        restart_delay_seconds: Delay between restart attempts
        **kwargs: Keyword arguments to pass to service_func
    """
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            # Check restart rate limit (max 5 restarts per hour)
            current_time = time.time()
            if service_name in _SERVICE_LAST_RESTART_TIME:
                time_since_last_restart = current_time - _SERVICE_LAST_RESTART_TIME[service_name]
                
                if time_since_last_restart < 3600:  # Within last hour
                    restart_count_in_hour = _SERVICE_RESTART_COUNTS.get(service_name, 0)
                    
                    if restart_count_in_hour >= _MAX_RESTARTS_PER_HOUR:
                        LOGGER.error(
                            f"🛑 {service_name}: Restart rate limit exceeded "
                            f"({restart_count_in_hour} restarts in last hour). Stopping service."
                        )
                        return
                else:
                    # Reset counter after 1 hour
                    _SERVICE_RESTART_COUNTS[service_name] = 0
            
            # Log startup
            if restart_count == 0:
                LOGGER.info(f"🚀 {service_name}: Starting...")
            else:
                LOGGER.warning(
                    f"🔄 {service_name}: Restarting (attempt {restart_count + 1}/{max_restarts})..."
                )
            
            # Update restart tracking
            _SERVICE_LAST_RESTART_TIME[service_name] = current_time
            _SERVICE_RESTART_COUNTS[service_name] = _SERVICE_RESTART_COUNTS.get(service_name, 0) + 1
            
            # Run the service
            await service_func(*args, **kwargs)
            
            # If service completes normally (not expected for background loops), exit gracefully
            LOGGER.info(f"✅ {service_name}: Completed normally")
            return
        
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            LOGGER.info(f"⏹️ {service_name}: Cancelled")
            return
        
        except Exception as e:
            restart_count += 1
            
            LOGGER.error(
                f"❌ {service_name}: Crashed with error: {e}",
                exc_info=True,
                extra={"restart_count": restart_count, "max_restarts": max_restarts}
            )
            
            if restart_count >= max_restarts:
                LOGGER.error(
                    f"🛑 {service_name}: Maximum restart limit reached ({max_restarts}). "
                    f"Service stopped."
                )
                return
            
            # Wait before restarting
            LOGGER.info(f"⏳ {service_name}: Waiting {restart_delay_seconds}s before restart...")
            await asyncio.sleep(restart_delay_seconds)


async def health_check_monitor(
    service_name: str,
    health_check_func: Callable[[], bool],
    check_interval_seconds: int = 300,
    restart_callback: Optional[Callable] = None
) -> None:
    """
    Monitor service health and trigger restart if health check fails.
    
    Args:
        service_name: Friendly name for logging
        health_check_func: Function that returns True if service is healthy
        check_interval_seconds: Interval between health checks
        restart_callback: Optional callback to trigger service restart
    """
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            await asyncio.sleep(check_interval_seconds)
            
            # Run health check
            is_healthy = health_check_func()
            
            if is_healthy:
                consecutive_failures = 0
                LOGGER.debug(f"✅ {service_name}: Health check passed")
            else:
                consecutive_failures += 1
                LOGGER.warning(
                    f"⚠️ {service_name}: Health check failed "
                    f"({consecutive_failures}/{max_consecutive_failures})"
                )
                
                if consecutive_failures >= max_consecutive_failures:
                    LOGGER.error(
                        f"🚨 {service_name}: Health check failed {consecutive_failures} times. "
                        f"Triggering restart..."
                    )
                    
                    if restart_callback:
                        await restart_callback()
                    
                    # Reset counter after triggering restart
                    consecutive_failures = 0
        
        except Exception as e:
            LOGGER.error(f"Health check monitor error for {service_name}: {e}", exc_info=True)
            await asyncio.sleep(60)  # Wait 1 min on error


def get_service_restart_stats() -> dict[str, Any]:
    """
    Get statistics about service restarts.
    
    Returns:
        {
            "service_name": {
                "total_restarts": int,
                "last_restart_time": float (epoch),
                "seconds_since_last_restart": int
            }
        }
    """
    stats = {}
    current_time = time.time()
    
    for service_name, restart_count in _SERVICE_RESTART_COUNTS.items():
        last_restart = _SERVICE_LAST_RESTART_TIME.get(service_name, 0)
        
        stats[service_name] = {
            "total_restarts": restart_count,
            "last_restart_time": last_restart,
            "seconds_since_last_restart": int(current_time - last_restart) if last_restart > 0 else None
        }
    
    return stats


# Export main functions
__all__ = [
    "auto_restart_wrapper",
    "health_check_monitor",
    "get_service_restart_stats"
]

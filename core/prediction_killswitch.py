"""
Ghost Protocol - Prediction Killswitch
Emergency stop for all predictions until system is verified.

Must be explicitly enabled via PREDICTIONS_ENABLED=true after fixes are verified.
"""

import os
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class PredictionKillswitch:
    """
    Emergency stop for all predictions.
    Must be explicitly enabled after fixes are verified.
    
    Environment Variables:
        PREDICTIONS_ENABLED: 'true' to allow predictions, 'false' (default) to block
        KILLSWITCH_REASON: Reason shown when predictions are blocked
    """
    
    def __init__(self):
        self._reload_config()
    
    def _reload_config(self):
        """Reload configuration from environment"""
        self.enabled = os.environ.get('PREDICTIONS_ENABLED', 'false').lower() == 'true'
        self.override_reason = os.environ.get(
            'KILLSWITCH_REASON', 
            'System under maintenance - accuracy tracking being implemented'
        )
    
    def can_send_prediction(self) -> bool:
        """
        Check if predictions are allowed.
        Returns True only if PREDICTIONS_ENABLED=true
        """
        # Refresh config in case env vars changed
        self._reload_config()
        
        if not self.enabled:
            logger.warning(f"⛔ Prediction BLOCKED: {self.override_reason}")
            return False
        return True
    
    def get_status(self) -> Dict:
        """Get current killswitch status"""
        self._reload_config()
        
        return {
            'predictions_enabled': self.enabled,
            'killswitch_active': not self.enabled,
            'reason': self.override_reason if not self.enabled else 'System operational',
            'timestamp': datetime.utcnow().isoformat(),
            'env_var': os.environ.get('PREDICTIONS_ENABLED', 'not set')
        }
    
    def force_enable(self):
        """Force enable predictions (for testing only)"""
        logger.warning("⚠️ Force enabling predictions - use with caution!")
        self.enabled = True
        return self.get_status()
    
    def force_disable(self, reason: str = None):
        """Force disable predictions"""
        logger.info(f"🛑 Force disabling predictions: {reason or 'Manual override'}")
        self.enabled = False
        if reason:
            self.override_reason = reason
        return self.get_status()


# Singleton instance
_killswitch: PredictionKillswitch | None = None


def get_killswitch() -> PredictionKillswitch:
    """Get the singleton killswitch instance"""
    global _killswitch
    if _killswitch is None:
        _killswitch = PredictionKillswitch()
    return _killswitch

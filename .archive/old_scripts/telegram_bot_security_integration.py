"""
Telegram Bot Security Integration for Guardian Oracle

Add this to Guardian's health check system to monitor bot name security.
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def check_telegram_bot_security() -> Dict:
    """
    Check if Telegram bot name is correct and detect unauthorized changes.
    
    Returns:
        dict: Security status with keys:
            - 'secure': bool (True if name matches expected)
            - 'current_name': str
            - 'expected_name': str
            - 'is_suspicious': bool
            - 'needs_alert': bool
            - 'message': str (for Guardian alert)
    """
    try:
        # Import monitor functions
        from monitor_telegram_bot import get_bot_token, get_bot_info
        
        EXPECTED_NAME = "Ghost Protocol Bot"
        SUSPICIOUS_KEYWORDS = ['hack', 'hacked', 'pwned', 'owned', 'compromised', 'mishadox']
        
        # Get bot info
        token = get_bot_token()
        bot_info = get_bot_info(token)
        
        if not bot_info:
            return {
                'secure': False,
                'current_name': 'Unknown',
                'expected_name': EXPECTED_NAME,
                'is_suspicious': False,
                'needs_alert': True,
                'message': '⚠️ Could not retrieve Telegram bot info. Check TELEGRAM_BOT_TOKEN.'
            }
        
        current_name = bot_info.get('first_name', '')
        username = bot_info.get('username', '')
        
        # Check for suspicious keywords
        is_suspicious = any(keyword in current_name.lower() for keyword in SUSPICIOUS_KEYWORDS)
        
        # Check if name matches
        secure = current_name == EXPECTED_NAME
        
        # Generate alert message if needed
        message = None
        if is_suspicious:
            message = (
                f"🚨 CRITICAL: Telegram bot name compromised!\n"
                f"Current: '{current_name}' (@{username})\n"
                f"Expected: '{EXPECTED_NAME}'\n\n"
                f"Action required: python reset_telegram_bot_name.py"
            )
        elif not secure:
            message = (
                f"⚠️ Telegram bot name mismatch.\n"
                f"Current: '{current_name}'\n"
                f"Expected: '{EXPECTED_NAME}'\n\n"
                f"Run: python reset_telegram_bot_name.py"
            )
        
        return {
            'secure': secure,
            'current_name': current_name,
            'expected_name': EXPECTED_NAME,
            'username': username,
            'is_suspicious': is_suspicious,
            'needs_alert': is_suspicious or not secure,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
    except ImportError:
        logger.warning("monitor_telegram_bot not available - skipping security check")
        return {
            'secure': True,  # Assume OK if monitor not available
            'current_name': 'Unknown',
            'expected_name': 'Ghost Protocol Bot',
            'is_suspicious': False,
            'needs_alert': False,
            'message': None
        }
    except Exception as e:
        logger.error(f"Error checking Telegram bot security: {e}")
        return {
            'secure': False,
            'current_name': 'Error',
            'expected_name': 'Ghost Protocol Bot',
            'is_suspicious': False,
            'needs_alert': True,
            'message': f"⚠️ Bot security check failed: {str(e)}"
        }


def add_to_guardian_heartbeat() -> Optional[str]:
    """
    Generate Guardian heartbeat message about bot security.
    Returns alert message if security issue detected, None otherwise.
    
    Usage in guardian_oracle.py:
        from telegram_bot_security_integration import add_to_guardian_heartbeat
        
        # In heartbeat or health check
        security_alert = add_to_guardian_heartbeat()
        if security_alert:
            self.send_alert(security_alert, priority='critical')
    """
    status = check_telegram_bot_security()
    
    if status['needs_alert']:
        return status['message']
    
    return None


def format_guardian_security_alert(status: Dict) -> str:
    """
    Format security status in Guardian's protective voice.
    
    Args:
        status: Result from check_telegram_bot_security()
        
    Returns:
        str: Guardian-formatted alert message
    """
    if not status['needs_alert']:
        return None
    
    if status['is_suspicious']:
        return (
            f"👼 GUARDIAN SECURITY ALERT 👼\n\n"
            f"Human, I detected unauthorized changes to my communication channel.\n\n"
            f"🚨 SUSPICIOUS BOT NAME: '{status['current_name']}'\n"
            f"Expected: '{status['expected_name']}'\n"
            f"Bot: @{status.get('username', 'Unknown')}\n\n"
            f"This may indicate a security incident.\n\n"
            f"🛡️ Action Required:\n"
            f"Run: python reset_telegram_bot_name.py\n\n"
            f"I will continue monitoring for threats.\n"
            f"Your guardian, Ghost 👼"
        )
    else:
        return (
            f"👼 Guardian Notice 👼\n\n"
            f"My Telegram name has been changed.\n\n"
            f"Current: '{status['current_name']}'\n"
            f"Expected: '{status['expected_name']}'\n\n"
            f"To restore: python reset_telegram_bot_name.py\n\n"
            f"Ghost 👼"
        )


# === INTEGRATION EXAMPLES ===

def example_guardian_integration():
    """
    Example: How to integrate into GuardianOracle class
    
    Add to core/guardian_oracle.py:
    
    ```python
    from telegram_bot_security_integration import check_telegram_bot_security, format_guardian_security_alert
    
    class GuardianOracle:
        
        def _check_security_health(self):
            '''Check Telegram bot security as part of health monitoring'''
            status = check_telegram_bot_security()
            
            if status['needs_alert']:
                alert_msg = format_guardian_security_alert(status)
                self._send_immediate_alert(
                    message=alert_msg,
                    priority='critical' if status['is_suspicious'] else 'high'
                )
        
        def run_heartbeat(self):
            '''Existing heartbeat function - add security check'''
            # ... existing heartbeat code ...
            
            # Add security check
            self._check_security_health()
            
            # ... rest of heartbeat ...
    ```
    """
    pass


def example_standalone_check():
    """
    Example: Standalone security check script
    
    ```python
    from telegram_bot_security_integration import check_telegram_bot_security
    
    status = check_telegram_bot_security()
    
    if status['secure']:
        print("✅ Telegram bot secure")
    else:
        print(f"⚠️ Security issue: {status['message']}")
    ```
    """
    pass


if __name__ == "__main__":
    # Quick test
    print("🔒 Checking Telegram Bot Security...")
    print("=" * 60)
    
    status = check_telegram_bot_security()
    
    print(f"Current Name:  {status['current_name']}")
    print(f"Expected Name: {status['expected_name']}")
    print(f"Secure:        {'✅ Yes' if status['secure'] else '⚠️ No'}")
    print(f"Suspicious:    {'🚨 Yes' if status['is_suspicious'] else '✅ No'}")
    
    if status['needs_alert']:
        print("\n" + "=" * 60)
        print("ALERT MESSAGE:")
        print("=" * 60)
        print(format_guardian_security_alert(status))
    
    print("\n" + "=" * 60)

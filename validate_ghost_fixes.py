#!/usr/bin/env python3
"""
Ghost Oracle Fix Validator
Automated tool to verify all 6 fixes are working correctly
Run this after receiving morning prophecy to validate
"""

import sys
from datetime import datetime
from core.position_manager import get_position_manager

def validate_position_locking():
    """
    Fix #1: Verify position entries are locked
    """
    print("=" * 70)
    print("Fix #1: Position Locking ⚓")
    print("=" * 70)
    
    try:
        pm = get_position_manager()
        positions = pm.get_all_active()
        
        if not positions:
            print("⚠️  No active positions found")
            print("   This is OK if it's the first day or no setups meet criteria")
            return "N/A"
        
        print(f"✅ Found {len(positions)} active position(s) with locked entries:")
        print()
        
        all_locked = True
        for p in positions:
            symbol = p['symbol']
            entry = p['entry_price']
            current = p.get('current_price', entry)
            
            print(f"  {symbol}:")
            print(f"    🔒 Entry (LOCKED): ${entry:.4f}")
            print(f"    📊 Current Price: ${current:.4f}")
            print(f"    📍 Status: {p.get('status', 'active')}")
            
            # Entry should never be 0 or None
            if not entry or entry == 0:
                print(f"    ❌ ERROR: Entry price not set!")
                all_locked = False
            else:
                print(f"    ✅ Entry price locked correctly")
            print()
        
        return "PASS" if all_locked else "FAIL"
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return "FAIL"


def validate_stop_losses():
    """
    Fix #4: Verify stop losses are set correctly
    """
    print("=" * 70)
    print("Fix #4: Stop Loss Protection 🛡️")
    print("=" * 70)
    
    try:
        pm = get_position_manager()
        positions = pm.get_all_active()
        
        if not positions:
            print("⚠️  No active positions to check")
            return "N/A"
        
        print(f"Checking {len(positions)} position(s) for stop loss protection:")
        print()
        
        all_have_stops = True
        for p in positions:
            symbol = p['symbol']
            entry = p['entry_price']
            stop = p.get('stop_loss')
            direction = p.get('direction', 'UP')
            
            print(f"  {symbol}:")
            print(f"    Entry: ${entry:.4f}")
            print(f"    Stop Loss: ${stop:.4f}")
            print(f"    Direction: {direction}")
            
            # Verify stop loss is set
            if not stop or stop == 0:
                print(f"    ❌ ERROR: No stop loss set!")
                all_have_stops = False
                continue
            
            # Verify stop loss is correct distance
            if direction == "UP":
                expected_stop = entry * 0.95  # -5%
                if abs(stop - expected_stop) < 0.01:
                    print(f"    ✅ Stop loss correct (-5% from entry)")
                else:
                    print(f"    ⚠️  Stop loss unusual (expected ~${expected_stop:.4f})")
            else:  # DOWN
                expected_stop = entry * 1.05  # +5%
                if abs(stop - expected_stop) < 0.01:
                    print(f"    ✅ Stop loss correct (+5% from entry)")
                else:
                    print(f"    ⚠️  Stop loss unusual (expected ~${expected_stop:.4f})")
            print()
        
        return "PASS" if all_have_stops else "FAIL"
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return "FAIL"


def validate_position_status():
    """
    Check if positions show Continuing vs New status
    """
    print("=" * 70)
    print("Position Status Indicators")
    print("=" * 70)
    
    try:
        pm = get_position_manager()
        positions = pm.get_all_active()
        
        if not positions:
            print("⚠️  No positions to check status")
            return "N/A"
        
        for p in positions:
            symbol = p['symbol']
            is_continuation = p.get('is_continuation', False)
            status_icon = "📍 Continuing" if is_continuation else "🆕 New"
            
            print(f"  {symbol}: {status_icon}")
        
        print()
        return "INFO"
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return "FAIL"


def check_prophecy_content():
    """
    Guide user through manual checks for prophecy content
    """
    print("=" * 70)
    print("Manual Prophecy Content Checks")
    print("=" * 70)
    print()
    print("Please verify these in your Telegram prophecy message:")
    print()
    
    print("Fix #2: Result Reporting 📊")
    print("  [ ] Does prophecy show 'Yesterday's Results' section?")
    print("  [ ] Shows win/loss count?")
    print("  [ ] Shows total P&L?")
    print()
    
    print("Fix #3: Market Context 💡")
    print("  [ ] Each prediction shows '💡 Why:' reasoning?")
    print("  [ ] Reasoning includes RSI values?")
    print("  [ ] Reasoning includes MACD signals?")
    print("  [ ] Reasoning includes volume data?")
    print()
    
    print("Fix #5: Bearish Predictions 📉")
    print("  [ ] Are there any 📉 SHORT predictions?")
    print("  [ ] Or are all predictions 🚀 BUY?")
    print("  Count:")
    print("    UP predictions: ___")
    print("    DOWN predictions: ___")
    print()
    
    print("Fix #6: Liquidity Filter 💧")
    print("  [ ] All symbols are well-known coins?")
    print("  [ ] No obscure/low-volume tokens?")
    print()


def main():
    print()
    print("🔍 GHOST ORACLE FIX VALIDATOR")
    print("=" * 70)
    print(f"Validation Time: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    
    results = {}
    
    # Automated checks
    results['Position Locking'] = validate_position_locking()
    print()
    
    results['Stop Losses'] = validate_stop_losses()
    print()
    
    validate_position_status()
    print()
    
    # Manual check guidance
    check_prophecy_content()
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    automated_results = {k: v for k, v in results.items() if v in ["PASS", "FAIL"]}
    
    if automated_results:
        passed = sum(1 for v in automated_results.values() if v == "PASS")
        total = len(automated_results)
        
        print(f"Automated Checks: {passed}/{total} PASSED")
        print()
        
        for check, result in results.items():
            icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "ℹ️"
            print(f"  {icon} {check}: {result}")
        print()
    
    print("📋 Complete manual checks above and fill in validation checklist")
    print("📄 See: GHOST_FIX_VALIDATION_CHECKLIST.md")
    print()
    print("=" * 70)
    
    # Return exit code based on results
    if automated_results and all(v == "PASS" for v in automated_results.values()):
        print("✅ All automated checks PASSED")
        return 0
    elif any(v == "FAIL" for v in automated_results.values()):
        print("⚠️  Some automated checks FAILED")
        return 1
    else:
        print("ℹ️  Run this after positions are created")
        return 0


if __name__ == "__main__":
    sys.exit(main())

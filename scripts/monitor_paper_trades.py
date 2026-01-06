#!/usr/bin/env python3
"""
Paper Trade Resolution Monitor
==============================
Monitors paper trades and sends alerts when trades start resolving.

Usage:
    python scripts/monitor_paper_trades.py           # Check status once
    python scripts/monitor_paper_trades.py --watch   # Watch mode (check every 5 min)
    python scripts/monitor_paper_trades.py --alert   # Send Telegram alert on resolution

The system tracks:
- Total trades logged
- Pending vs resolved counts
- Win rate when trades resolve
- First resolution timestamp
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta

# Production API
PRODUCTION_URL = os.getenv("GHOST_API_URL", "https://ghost-protocol-production.up.railway.app")

# Telegram config (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram_alert(message: str) -> bool:
    """Send alert via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ALERT] Telegram not configured, skipping alert")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[ALERT] Failed to send Telegram: {e}")
        return False


def get_paper_stats() -> dict:
    """Fetch paper trading stats from production API."""
    try:
        r = requests.get(f"{PRODUCTION_URL}/api/v3/paper/stats?days=365", timeout=30)
        data = r.json()
        return data.get("stats", {})
    except Exception as e:
        print(f"[ERROR] Failed to get stats: {e}")
        return {}


def get_sample_trades(limit: int = 10) -> list:
    """Fetch sample trades from production API."""
    try:
        r = requests.get(f"{PRODUCTION_URL}/api/v3/paper/trades?days=365", timeout=30)
        data = r.json()
        return data.get("trades", [])[:limit]
    except Exception as e:
        print(f"[ERROR] Failed to get trades: {e}")
        return []


def check_status(verbose: bool = True) -> dict:
    """Check paper trading status and return metrics."""
    stats = get_paper_stats()
    
    if not stats:
        print("[ERROR] Could not fetch paper trading stats")
        return {}
    
    total = stats.get("total_trades", 0)
    resolved = stats.get("resolved_trades", 0)
    pending = stats.get("pending_trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    stopped = stats.get("stopped", 0)
    win_rate = stats.get("win_rate", 0)
    total_pnl = stats.get("total_pnl", 0)
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "total": total,
        "resolved": resolved,
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "stopped": stopped,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "has_resolutions": resolved > 0
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("📊 PAPER TRADING STATUS")
        print("=" * 50)
        print(f"  Timestamp:   {result['timestamp']}")
        print(f"  Total:       {total:,}")
        print(f"  Pending:     {pending:,}")
        print(f"  Resolved:    {resolved:,}")
        print()
        
        if resolved > 0:
            print("  📈 RESOLUTION METRICS:")
            print(f"    Wins:      {wins} ({win_rate:.1%})")
            print(f"    Losses:    {losses}")
            print(f"    Stopped:   {stopped}")
            print(f"    Total P&L: ${total_pnl:,.2f}")
        else:
            # Calculate time to first resolution
            trades = get_sample_trades(1)
            if trades:
                oldest = trades[-1]
                target_time = oldest.get("target_time", "")
                if target_time:
                    try:
                        tt = datetime.fromisoformat(target_time.replace("Z", "+00:00").replace("+00:00", ""))
                        now = datetime.utcnow()
                        hours_left = (tt - now).total_seconds() / 3600
                        if hours_left > 0:
                            print(f"  ⏳ First resolution in: {hours_left:.1f} hours")
                        else:
                            print(f"  ⚠️ Trades should be resolving (target time passed)")
                    except:
                        pass
        print("=" * 50)
    
    return result


def watch_mode(interval_minutes: int = 5, send_alerts: bool = False):
    """
    Watch paper trades and alert on state changes.
    
    Args:
        interval_minutes: How often to check (default 5 min)
        send_alerts: Whether to send Telegram alerts
    """
    print(f"\n🔭 WATCH MODE: Checking every {interval_minutes} minutes")
    print("   Press Ctrl+C to stop\n")
    
    last_resolved = 0
    first_resolution_alerted = False
    
    while True:
        try:
            status = check_status(verbose=True)
            
            if not status:
                print("[WARN] Failed to get status, retrying...")
                time.sleep(interval_minutes * 60)
                continue
            
            resolved = status.get("resolved", 0)
            
            # Alert on first resolution
            if resolved > 0 and not first_resolution_alerted:
                first_resolution_alerted = True
                message = (
                    "🎉 <b>PAPER TRADES RESOLVING!</b>\n\n"
                    f"First batch of paper trades has resolved:\n"
                    f"• Resolved: {resolved}\n"
                    f"• Wins: {status.get('wins', 0)}\n"
                    f"• Win Rate: {status.get('win_rate', 0):.1%}\n"
                    f"• Total P&L: ${status.get('total_pnl', 0):,.2f}\n\n"
                    f"🔗 Check: {PRODUCTION_URL}/api/v3/paper/stats"
                )
                print(f"\n🎉 FIRST RESOLUTION DETECTED!")
                
                if send_alerts:
                    send_telegram_alert(message)
            
            # Alert on significant changes
            if resolved > last_resolved and last_resolved > 0:
                new_resolutions = resolved - last_resolved
                message = (
                    f"📊 <b>Paper Trade Update</b>\n\n"
                    f"• New resolutions: +{new_resolutions}\n"
                    f"• Total resolved: {resolved}\n"
                    f"• Win rate: {status.get('win_rate', 0):.1%}\n"
                    f"• P&L: ${status.get('total_pnl', 0):,.2f}"
                )
                
                if send_alerts and new_resolutions >= 10:
                    send_telegram_alert(message)
            
            last_resolved = resolved
            
            print(f"\n⏰ Next check in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Watch mode stopped")
            break
        except Exception as e:
            print(f"[ERROR] Watch loop error: {e}")
            time.sleep(60)


def trigger_reconciliation():
    """Manually trigger paper trade reconciliation."""
    print("\n🔄 Triggering paper trade reconciliation...")
    
    try:
        r = requests.post(
            f"{PRODUCTION_URL}/api/v3/paper/check_all",
            timeout=120
        )
        data = r.json()
        
        if data.get("ok"):
            resolved = data.get("resolved_count", 0)
            print(f"✅ Reconciliation complete: {resolved} trades resolved")
            return data
        else:
            print(f"❌ Reconciliation failed: {data.get('error', 'Unknown error')}")
            return data
    except requests.Timeout:
        print("⚠️ Reconciliation timed out (this is normal for large batches)")
        print("   The process may still be running in the background.")
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ Reconciliation error: {e}")
        return {"ok": False, "error": str(e)}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trade Resolution Monitor")
    parser.add_argument("--watch", action="store_true", help="Watch mode (continuous monitoring)")
    parser.add_argument("--alert", action="store_true", help="Send Telegram alerts")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in minutes (default: 5)")
    parser.add_argument("--reconcile", action="store_true", help="Trigger manual reconciliation")
    args = parser.parse_args()
    
    if args.reconcile:
        trigger_reconciliation()
    elif args.watch:
        watch_mode(interval_minutes=args.interval, send_alerts=args.alert)
    else:
        check_status()


if __name__ == "__main__":
    main()

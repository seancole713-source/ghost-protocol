#!/usr/bin/env python3
"""
Ghost Protocol - Daily Monitoring Script
=========================================

Quick daily check of Ghost prediction accuracy and system health.

Run: python scripts/daily_monitor.py

Uses the production Railway API endpoints.

Author: Ghost AI
Date: December 28, 2025
"""

import json
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Configuration
BASE_URL = "https://ghost-protocol-production.up.railway.app"

def colored(text: str, color: str) -> str:
    """Add ANSI color to text."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
        "bold": "\033[1m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def fetch_endpoint(endpoint: str) -> dict:
    """Fetch data from a production endpoint."""
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(colored(f"  {title}", "bold"))
    print("=" * 60)

def print_metric(label: str, value, status: str = "neutral"):
    """Print a metric with status color."""
    color = {"good": "green", "bad": "red", "warn": "yellow", "neutral": "cyan"}.get(status, "cyan")
    print(f"  {label}: {colored(str(value), color)}")

def check_health():
    """Check if the system is running."""
    print_header("🏥 SYSTEM HEALTH")
    
    data = fetch_endpoint("/health")
    if "error" in data:
        print_metric("Status", "OFFLINE ❌", "bad")
        return False
    
    print_metric("Status", "ONLINE ✅", "good")
    print_metric("Server", data.get("status", "unknown"))
    return True

def check_accuracy():
    """Check prediction accuracy from clean database."""
    print_header("📊 PREDICTION ACCURACY")
    
    data = fetch_endpoint("/debug/db-audit")
    if not data.get("ok"):
        print_metric("Error", data.get("error", "Unknown"), "bad")
        return
    
    stats = data.get("outcomes_stats", {})
    overview = data.get("overview", {})
    
    total = stats.get("total", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    accuracy = stats.get("accuracy_pct", 0)
    no_data = stats.get("no_data", 0)
    
    # Determine status
    if accuracy >= 60:
        status = "good"
        emoji = "🟢"
    elif accuracy >= 50:
        status = "warn"
        emoji = "🟡"
    else:
        status = "bad"
        emoji = "🔴"
    
    print_metric("Total Outcomes", total)
    print_metric("Wins", wins, "good" if wins > losses else "bad")
    print_metric("Losses", losses, "bad" if losses > wins else "good")
    print_metric("No Data", no_data, "bad" if no_data > 0 else "good")
    print_metric(f"Accuracy {emoji}", f"{accuracy}%", status)
    
    # Date range
    print(f"\n  📅 Date Range:")
    print(f"     Earliest: {overview.get('earliest', 'N/A')}")
    print(f"     Latest: {overview.get('latest', 'N/A')}")
    
    # Corrupt data check
    corrupt = data.get("total_corrupt", 0)
    if corrupt > 0:
        print_metric("\n  ⚠️ Corrupt Records", corrupt, "bad")
    else:
        print_metric("\n  ✅ Data Quality", "Clean", "good")
    
    return accuracy

def check_inverse_status():
    """Check INVERSE mode configuration."""
    print_header("🔄 INVERSE MODE STATUS")
    
    data = fetch_endpoint("/debug/inverse-status")
    if not data.get("ok"):
        print_metric("Error", data.get("error", "Unknown"), "bad")
        return
    
    enabled = data.get("inverse_ghost_enabled", False)
    skip_count = data.get("inverse_skip_count", 0)
    
    print_metric("INVERSE Mode", "ENABLED ✅" if enabled else "DISABLED ❌", "good" if enabled else "warn")
    print_metric("Skip Symbols", f"{skip_count} symbols use RAW predictions")
    
    # Sample symbol modes
    symbol_modes = data.get("symbol_modes", {})
    if symbol_modes:
        print(f"\n  📊 Sample Symbol Modes:")
        for sym, mode in list(symbol_modes.items())[:6]:
            indicator = "🔄" if "INVERTED" in mode else "⏭️"
            print(f"     {indicator} {sym}: {mode.split('(')[0].strip()}")

def check_recent_predictions():
    """Check recent TOP 10 predictions."""
    print_header("📈 RECENT PREDICTIONS")
    
    data = fetch_endpoint("/debug/top10-preview")
    if "error" in data:
        print_metric("Error", data.get("error", "Unknown"), "bad")
        return
    
    predictions = data.get("predictions", [])
    if not predictions:
        print_metric("Status", "No predictions available", "warn")
        return
    
    print(f"  Latest TOP 10 Preview ({len(predictions)} symbols):\n")
    
    for i, pred in enumerate(predictions[:10], 1):
        symbol = pred.get("symbol", "???")
        direction = pred.get("direction", "???")
        confidence = pred.get("confidence", 0)
        
        dir_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        
        print(f"     {i:2}. {symbol:6} {dir_emoji} {direction:4} [{conf_bar}] {confidence:.0%}")

def check_notification_loop():
    """Check if notification loop is running."""
    print_header("🔔 NOTIFICATION SYSTEM")
    
    data = fetch_endpoint("/debug/notification-loop-status")
    if "error" in data:
        print_metric("Error", data.get("error", "Unknown"), "bad")
        return
    
    running = data.get("loop_running", False)
    last_run = data.get("last_run_time", "Never")
    next_run = data.get("next_run_time", "Unknown")
    
    print_metric("Loop Status", "RUNNING ✅" if running else "STOPPED ❌", "good" if running else "bad")
    print_metric("Last Run", last_run)
    print_metric("Next Run", next_run)

def generate_summary(accuracy: float):
    """Generate a summary with recommendations."""
    print_header("📋 DAILY SUMMARY")
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  Report Generated: {now}\n")
    
    if accuracy is None:
        print(colored("  ⚠️ Could not calculate accuracy - check database connection", "yellow"))
        return
    
    # Status assessment
    if accuracy >= 65:
        print(colored("  ✅ EXCELLENT: Ghost is performing well!", "green"))
        print("     • Continue monitoring daily")
        print("     • Consider expanding symbol coverage")
    elif accuracy >= 55:
        print(colored("  🟢 GOOD: Ghost is above random chance", "green"))
        print("     • Monitor for improvement")
        print("     • Review INVERSE_SKIP_SYMBOLS configuration")
    elif accuracy >= 45:
        print(colored("  🟡 FAIR: Near random chance", "yellow"))
        print("     • Review recent predictions manually")
        print("     • Check INVERSE mode is working correctly")
        print("     • Consider adding more symbols to exclusions")
    else:
        print(colored("  🔴 NEEDS ATTENTION: Below random chance", "red"))
        print("     • Check if INVERSE mode should be toggled")
        print("     • Review per-symbol accuracy")
        print("     • Consider model retraining")
    
    print("\n  📊 Monitor Commands:")
    print("     curl -s 'https://ghost-protocol-production.up.railway.app/debug/db-audit' | jq")
    print("     curl -s 'https://ghost-protocol-production.up.railway.app/debug/outcome-data-audit' | jq")

def main():
    """Run all monitoring checks."""
    print("\n" + "🔮" * 30)
    print(colored("       GHOST PROTOCOL - DAILY MONITOR", "bold"))
    print("🔮" * 30)
    print(f"\n  {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
    
    # Run checks
    if not check_health():
        print(colored("\n❌ System is offline - cannot continue checks", "red"))
        return 1
    
    accuracy = check_accuracy()
    check_inverse_status()
    check_recent_predictions()
    check_notification_loop()
    generate_summary(accuracy)
    
    print("\n" + "=" * 60)
    print(colored("  ✅ Daily monitoring complete!", "green"))
    print("=" * 60 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

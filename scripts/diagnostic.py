#!/usr/bin/env python3
"""
Ghost Protocol - Prediction Loop Diagnostic
Checks why predictions aren't generating
"""

import asyncio
import aiohttp
from datetime import datetime

BASE_URL = "https://ghost-protocol-production.up.railway.app"


async def check_system_status():
    """Comprehensive status check"""
    print(f"🔍 Ghost Protocol Diagnostic - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 80)
    
    async with aiohttp.ClientSession() as session:
        # 1. Health Score
        print("\n📊 HEALTH STATUS")
        try:
            async with session.get(f"{BASE_URL}/integrity/audit/readonly", timeout=30) as resp:
                data = await resp.json()
                health = data.get("health_score", 0)
                print(f"  Health Score: {health:.1f}/100")
                
                breakdown = data.get("score_breakdown", [])
                for component in breakdown:
                    comp_name = component.get("component", "unknown")
                    count = component.get("count", 0)
                    penalty = component.get("penalty", 0)
                    print(f"  {comp_name.upper()}: {count} issues (penalty: {penalty})")
                    for detail in component.get("details", [])[:3]:
                        print(f"    - {detail}")
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
        
        # 2. Heartbeat Status
        print("\n💓 HEARTBEAT STATUS")
        try:
            async with session.get(f"{BASE_URL}/api/v3/heartbeat/status", timeout=10) as resp:
                data = await resp.json()
                tasks = data.get("tasks", {})
                
                prediction_cycle = tasks.get("prediction-cycle", {})
                if prediction_cycle:
                    age_s = prediction_cycle.get("age_s", 999999)
                    alive = prediction_cycle.get("alive", False)
                    print(f"  Prediction Cycle: {'🟢 ALIVE' if alive else '🔴 DEAD'} (last pulse {age_s:.1f}s ago)")
                
                # Show other critical tasks
                for task_name in ["alert-worker", "price-recorder", "outcome-reconciler"]:
                    task = tasks.get(task_name, {})
                    if task:
                        age_s = task.get("age_s", 999999)
                        alive = task.get("alive", False)
                        print(f"  {task_name}: {'🟢' if alive else '🔴'} (age: {age_s:.1f}s)")
        except Exception as e:
            print(f"  ❌ Heartbeat check failed: {e}")
        
        # 3. Recent Predictions
        print("\n🎯 RECENT PREDICTIONS")
        try:
            async with session.get(f"{BASE_URL}/api/v4/picks", timeout=10) as resp:
                data = await resp.json()
                picks = data.get("picks", [])
                print(f"  Total picks in DB: {len(picks)}")
                
                if picks:
                    print(f"  Latest picks:")
                    for pick in picks[:5]:
                        symbol = pick.get("symbol", "???")
                        direction = pick.get("direction", "?")
                        confidence = pick.get("confidence", 0)
                        # Try to get timestamp
                        created = pick.get("created_at", pick.get("run_at", "unknown"))
                        print(f"    {symbol:6s} {direction:4s} {confidence:.0%} (created: {created})")
                else:
                    print("  ⚠️  No picks found in database")
        except Exception as e:
            print(f"  ❌ Picks check failed: {e}")
        
        # 4. Accuracy Status
        print("\n📈 ACCURACY STATUS")
        try:
            async with session.get(f"{BASE_URL}/api/v3/accuracy/summary", timeout=10) as resp:
                data = await resp.json()
                accuracy = data.get("accuracy_pct", 0)
                total = data.get("total_predictions", 0)
                wins = data.get("total_wins", 0)
                losses = data.get("total_losses", 0)
                last_pred_age = data.get("last_prediction_age_min", "N/A")
                
                print(f"  Overall Accuracy: {accuracy:.1f}% ({wins}W / {losses}L / {total} total)")
                print(f"  Last Prediction: {last_pred_age} minutes ago")
        except Exception as e:
            print(f"  ❌ Accuracy check failed: {e}")
        
        # 5. Watchlist Status
        print("\n👁️  WATCHLIST STATUS")
        try:
            async with session.get(f"{BASE_URL}/api/v3/watchlist/enriched", timeout=10) as resp:
                data = await resp.json()
                items = data.get("items", data.get("watchlist", []))
                print(f"  Total symbols: {len(items)}")
                
                active_predictions = [
                    item for item in items
                    if item.get("ghost_direction") not in ["HOLD", None, ""]
                ]
                print(f"  Active predictions: {len(active_predictions)}")
                
                if active_predictions:
                    print(f"  Sample:")
                    for item in active_predictions[:5]:
                        symbol = item.get("symbol", "???")
                        direction = item.get("ghost_direction", "?")
                        confidence = item.get("ghost_confidence", 0)
                        print(f"    {symbol:6s} {direction:4s} {confidence:.0%}")
        except Exception as e:
            print(f"  ❌ Watchlist check failed: {e}")
        
        # 6. Performance Gate Status
        print("\n🚪 PERFORMANCE GATE STATUS")
        try:
            async with session.get(f"{BASE_URL}/api/v3/accuracy/symbols", timeout=10) as resp:
                data = await resp.json()
                symbols = data.get("symbols", [])
                
                killed = [s for s in symbols if s.get("status") == "KILLED"]
                warned = [s for s in symbols if s.get("status") == "WARNED"]
                active = [s for s in symbols if s.get("status") == "ACTIVE"]
                
                print(f"  Active: {len(active)}, Warned: {len(warned)}, Killed: {len(killed)}")
                
                if killed:
                    print(f"  Killed symbols: {', '.join([s.get('symbol', '?') for s in killed[:5]])}")
        except Exception as e:
            print(f"  ❌ Performance gate check failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Diagnostic complete\n")


if __name__ == "__main__":
    asyncio.run(check_system_status())

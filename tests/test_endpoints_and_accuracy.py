#!/usr/bin/env python3
"""
Test Endpoints and Monitor Accuracy Ledger
==========================================
Comprehensive testing of all new endpoints and accuracy tracking.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_accuracy_database():
    """Check the forecast accuracy database"""
    print_section("📊 ACCURACY LEDGER - FORECAST DATABASE")
    
    db_path = Path(__file__).parent / "data" / "forecast_accuracy.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Database found: {db_path}")
    print(f"   Size: {db_path.stat().st_size:,} bytes")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total forecasts
    cursor.execute("SELECT COUNT(*) FROM forecasts")
    total = cursor.fetchone()[0]
    print(f"\n📈 Total Forecasts Recorded: {total}")
    
    if total == 0:
        print("   No forecasts in database yet")
        conn.close()
        return
    
    # Get forecasts with actuals (completed predictions)
    cursor.execute("""
        SELECT COUNT(*) FROM forecasts 
        WHERE actual_price IS NOT NULL
    """)
    completed = cursor.fetchone()[0]
    print(f"✅ Completed (with actual price): {completed}")
    print(f"⏳ Pending (waiting for actual): {total - completed}")
    
    # Get accuracy stats
    cursor.execute("""
        SELECT 
            symbol,
            COUNT(*) as count,
            AVG(percentage_error) as avg_error,
            MIN(percentage_error) as min_error,
            MAX(percentage_error) as max_error
        FROM forecasts
        WHERE actual_price IS NOT NULL
        GROUP BY symbol
        ORDER BY count DESC
    """)
    
    stats = cursor.fetchall()
    if stats:
        print("\n📊 Accuracy by Symbol:")
        print(f"{'Symbol':<8} {'Count':<8} {'Avg Error':<12} {'Min Error':<12} {'Max Error':<12}")
        print("-" * 60)
        for symbol, count, avg_err, min_err, max_err in stats:
            accuracy = 100 - avg_err if avg_err else 0
            print(f"{symbol:<8} {count:<8} {accuracy:>6.2f}% acc  {min_err:>6.2f}%  {max_err:>6.2f}%")
    
    # Get recent forecasts (last 10)
    cursor.execute("""
        SELECT 
            symbol,
            forecast_price,
            actual_price,
            percentage_error,
            forecast_horizon_hours,
            datetime(forecast_timestamp, 'unixepoch') as forecast_time,
            datetime(actual_timestamp, 'unixepoch') as actual_time
        FROM forecasts
        WHERE actual_price IS NOT NULL
        ORDER BY id DESC
        LIMIT 10
    """)
    
    recent = cursor.fetchall()
    if recent:
        print("\n📋 Recent Completed Forecasts (Last 10):")
        print(f"{'Symbol':<8} {'Forecast':<10} {'Actual':<10} {'Error':<10} {'Horizon':<10} {'Date'}")
        print("-" * 80)
        for symbol, forecast, actual, error, horizon, f_time, a_time in recent:
            status = "✅" if error < 5 else "⚠️" if error < 10 else "❌"
            print(f"{status} {symbol:<6} ${forecast:<8.2f} ${actual:<8.2f} {error:>6.2f}%  {horizon}h    {f_time}")
    
    # Get pending forecasts
    cursor.execute("""
        SELECT 
            symbol,
            forecast_price,
            forecast_horizon_hours,
            datetime(forecast_timestamp, 'unixepoch') as forecast_time,
            confidence
        FROM forecasts
        WHERE actual_price IS NULL
        ORDER BY forecast_timestamp DESC
        LIMIT 5
    """)
    
    pending = cursor.fetchall()
    if pending:
        print("\n⏳ Pending Forecasts (Waiting for Actual Price):")
        print(f"{'Symbol':<8} {'Forecast':<12} {'Horizon':<10} {'Confidence':<12} {'Date'}")
        print("-" * 70)
        for symbol, forecast, horizon, f_time, confidence in pending:
            conf_str = f"{confidence:.0%}" if confidence else "N/A"
            print(f"   {symbol:<6} ${forecast:<10.2f} {horizon}h      {conf_str:<10} {f_time}")
    
    conn.close()


def check_prediction_database():
    """Check the predictions database (wolf_app.py predictor)"""
    print_section("🔮 PREDICTION DATABASE - MARKET PREDICTIONS")
    
    db_path = Path(__file__).parent / "data" / "predictions.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Database found: {db_path}")
    print(f"   Size: {db_path.stat().st_size:,} bytes")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    print(f"\n🎯 Total Predictions: {total}")
    
    if total == 0:
        print("   No predictions in database yet")
        conn.close()
        return
    
    # Get recent predictions
    cursor.execute("""
        SELECT 
            symbol,
            direction,
            confidence,
            datetime(run_at, 'unixepoch') as pred_time,
            horizon_h,
            method
        FROM predictions
        ORDER BY run_at DESC
        LIMIT 10
    """)
    
    predictions = cursor.fetchall()
    if predictions:
        print("\n📈 Recent Predictions (Last 10):")
        print(f"{'Symbol':<8} {'Direction':<10} {'Confidence':<12} {'Horizon':<10} {'Time':<20} {'Method'}")
        print("-" * 90)
        for symbol, direction, confidence, pred_time, horizon, method in predictions:
            emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
            conf_str = f"{confidence:.0%}" if confidence else "N/A"
            print(f"{emoji} {symbol:<6} {direction:<8} {conf_str:<10} {horizon}h      {pred_time:<18} {method}")
    
    # Get prediction outcomes
    cursor.execute("""
        SELECT 
            p.symbol,
            p.direction,
            o.mae,
            o.map,
            o.hit_direction,
            datetime(o.closed_at, 'unixepoch') as outcome_time
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        ORDER BY o.closed_at DESC
        LIMIT 10
    """)
    
    outcomes = cursor.fetchall()
    if outcomes:
        print("\n✅ Recent Prediction Outcomes (Last 10):")
        print(f"{'Symbol':<8} {'Direction':<10} {'Hit?':<8} {'MAE':<10} {'MAP':<10} {'Time'}")
        print("-" * 70)
        for symbol, direction, mae, map_err, hit, outcome_time in outcomes:
            hit_emoji = "✅" if hit else "❌"
            dir_emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
            print(f"{hit_emoji} {dir_emoji} {symbol:<6} {direction:<8} {hit:<6} {mae:<8.2f} {map_err:<8.2f}% {outcome_time}")
    
    conn.close()


def check_ai_advisor_accuracy():
    """Check AI Advisor accuracy tracker"""
    print_section("🤖 AI ADVISOR - DECISION ACCURACY")
    
    db_path = Path(__file__).parent / "data" / "ai_advisor.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Database found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total decisions
    cursor.execute("SELECT COUNT(*) FROM ai_decisions")
    total = cursor.fetchone()[0]
    print(f"\n🎯 Total AI Decisions: {total}")
    
    if total == 0:
        print("   No AI decisions recorded yet")
        conn.close()
        return
    
    # Get accuracy stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN outcome_return_pct IS NOT NULL THEN outcome_return_pct ELSE 0 END) as avg_return
        FROM ai_decisions
        WHERE outcome_checked = 1
    """)
    
    stats = cursor.fetchone()
    if stats and stats[0] > 0:
        total_checked, correct, avg_return = stats
        accuracy = (correct / total_checked * 100) if total_checked > 0 else 0
        print(f"\n📊 Decision Accuracy:")
        print(f"   Total Checked: {total_checked}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Avg Return: {avg_return:.2f}%")
    
    # Get recent decisions
    cursor.execute("""
        SELECT 
            asset,
            decision,
            confidence,
            outcome_correct,
            outcome_return_pct,
            datetime(created_at, 'unixepoch') as decision_time
        FROM ai_decisions
        WHERE outcome_checked = 1
        ORDER BY created_at DESC
        LIMIT 10
    """)
    
    decisions = cursor.fetchall()
    if decisions:
        print("\n📋 Recent AI Decisions (Last 10):")
        print(f"{'Asset':<8} {'Decision':<8} {'Confidence':<12} {'Correct?':<10} {'Return':<10} {'Time'}")
        print("-" * 75)
        for asset, decision, confidence, correct, return_pct, dec_time in decisions:
            correct_emoji = "✅" if correct else "❌"
            dec_emoji = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "⚪"
            conf_str = f"{confidence:.0%}" if confidence else "N/A"
            ret_str = f"{return_pct:+.2f}%" if return_pct else "N/A"
            print(f"{correct_emoji} {dec_emoji} {asset:<6} {decision:<6} {conf_str:<10} {correct:<8} {ret_str:<8} {dec_time}")
    
    conn.close()


def generate_console_test_code():
    """Generate JavaScript code for browser console testing"""
    print_section("🌐 BROWSER CONSOLE TEST CODE")
    
    console_code = """
// Copy and paste this into your browser console at:
// https://ghost-sniper-bot-seancole713-production.up.railway.app

console.log('🎭 Ghost Protocol - Endpoint Test Suite');
console.log('=' .repeat(60));

// Test 1: World Context
fetch('/api/world/context')
  .then(r => r.json())
  .then(data => {
    console.log('✅ World Context:', data);
    console.log('   SPY:', data.spy_price);
    console.log('   VIX:', data.vix_level);
    console.log('   Mood:', data.market_mood);
  })
  .catch(e => console.error('❌ World Context failed:', e));

// Test 2: Goals Tracker
fetch('/api/goals/all')
  .then(r => r.json())
  .then(data => {
    console.log('✅ Goals Tracker:', data);
    if (data.goals) {
      console.log('   Daily:', data.goals.daily);
      console.log('   Weekly:', data.goals.weekly);
      console.log('   Monthly:', data.goals.monthly);
    }
  })
  .catch(e => console.error('❌ Goals Tracker failed:', e));

// Test 3: XRP Tracker
fetch('/api/xrp/tracker')
  .then(r => r.json())
  .then(data => {
    console.log('✅ XRP Tracker:', data);
    console.log('   Price:', data.price);
    console.log('   Signal:', data.signal);
    console.log('   Bullish Eye:', data.bullish_eye);
    console.log('   Confidence:', data.confidence);
  })
  .catch(e => console.error('❌ XRP Tracker failed:', e));

// Test 4: VIP Coins
fetch('/api/vip/coins')
  .then(r => r.json())
  .then(data => {
    console.log('✅ VIP Coins:', data);
    if (data.coins) {
      data.coins.forEach(coin => {
        console.log(`   ${coin.symbol}: $${coin.price} (${coin.change_24h > 0 ? '+' : ''}${coin.change_24h.toFixed(2)}%)`);
      });
    }
  })
  .catch(e => console.error('❌ VIP Coins failed:', e));

// Test 5: Portfolio Positions
fetch('/api/portfolio/positions')
  .then(r => r.json())
  .then(data => {
    console.log('✅ Portfolio:', data);
    console.log('   Positions:', data.positions?.length || 0);
    console.log('   Total Value:', data.total_value);
    console.log('   Total P&L:', data.total_pnl);
  })
  .catch(e => console.error('❌ Portfolio failed:', e));

// Test 6: Accuracy Ledger
fetch('/api/stage2/forecasts?limit=10')
  .then(r => r.json())
  .then(data => {
    console.log('✅ Accuracy Ledger:', data);
    if (data.forecasts) {
      console.log(`   Total Forecasts: ${data.forecasts.length}`);
      data.forecasts.slice(0, 3).forEach(f => {
        console.log(`   ${f.symbol}: Forecast $${f.forecast_price?.toFixed(2) || 'N/A'} vs Actual $${f.actual_price?.toFixed(2) || 'Pending'}`);
      });
    }
  })
  .catch(e => console.error('❌ Accuracy Ledger failed:', e));

console.log('=' .repeat(60));
console.log('🎭 Test suite complete! Check results above.');
"""
    
    print(console_code)
    
    # Save to file
    script_path = Path(__file__).parent / "browser_console_test.js"
    with open(script_path, 'w') as f:
        f.write(console_code)
    print(f"\n✅ Saved to: {script_path}")


def show_verification_steps():
    """Show manual verification steps"""
    print_section("📋 VERIFICATION STEPS")
    
    steps = """
STEP 1: Check Railway Deployment
---------------------------------
1. Open Railway dashboard
2. Check deployment logs for errors
3. Verify app is running (not crashed)

STEP 2: Test Live Endpoints
----------------------------
Run these curl commands:

  curl https://ghost-sniper-bot-seancole713-production.up.railway.app/api/world/context
  curl https://ghost-sniper-bot-seancole713-production.up.railway.app/api/goals/all
  curl https://ghost-sniper-bot-seancole713-production.up.railway.app/api/xrp/tracker
  curl https://ghost-sniper-bot-seancole713-production.up.railway.app/api/vip/coins
  curl https://ghost-sniper-bot-seancole713-production.up.railway.app/api/portfolio/positions

STEP 3: Browser Console Test
------------------------------
1. Open: https://ghost-sniper-bot-seancole713-production.up.railway.app
2. Press F12 → Console tab
3. Copy/paste code from: browser_console_test.js
4. Look for ✅ checkmarks and data output

STEP 4: Monitor Accuracy Ledger
--------------------------------
1. Run this script periodically: python3 test_endpoints_and_accuracy.py
2. Check for new forecasts being recorded
3. Monitor accuracy percentages improving over time
4. Watch for auto-tuning adjustments

STEP 5: Telegram Integration
------------------------------
1. Send /predict command to bot
2. Wait for prediction response
3. Check accuracy ledger for new entry
4. Monitor throughout trading day
"""
    
    print(steps)


def main():
    """Run all checks"""
    print("\n")
    print("🎭" * 40)
    print("  GHOST PROTOCOL - ENDPOINT & ACCURACY TEST SUITE")
    print("🎭" * 40)
    
    # Check all databases
    check_accuracy_database()
    check_prediction_database()
    check_ai_advisor_accuracy()
    
    # Generate test code
    generate_console_test_code()
    
    # Show verification steps
    show_verification_steps()
    
    print("\n")
    print("🎭" * 40)
    print("  TEST SUITE COMPLETE")
    print("🎭" * 40)
    print("\n")


if __name__ == "__main__":
    main()

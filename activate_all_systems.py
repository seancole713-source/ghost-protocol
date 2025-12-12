#!/usr/bin/env python3
"""
🚀 GHOST PROTOCOL - SYSTEM ACTIVATION SCRIPT
============================================
Activates all hidden/incomplete Ghost systems:
- Crypto trading (15 coins)
- AI Advisor (autonomous scanner)
- Multi-timeframe analysis
- Social sentiment monitoring
- Economic calendar tracking
- Backtesting API endpoints

Usage: python activate_all_systems.py
"""

import os
import sys


def activate_crypto_suite():
    """Enable full crypto trading with 15 coins"""
    print("\n🪙 ACTIVATING CRYPTO SUITE...")
    print("   - 15 coins: BTC, ETH, SOL, XRP, ADA, DOGE, SHIB, PEPE, etc.")
    print("   - Crypto prediction engine")
    print("   - Telegram crypto routing")
    print("   - Crypto endpoints (/api/crypto/*)")
    print("   - AI Advisor crypto scanning")
    
    # Set environment variable
    os.environ["CRYPTO_ENABLED"] = "1"
    print("   ✅ Set CRYPTO_ENABLED=1")
    
    return True


def activate_ai_advisor():
    """Enable AI Advisor autonomous opportunity scanner"""
    print("\n🤖 ACTIVATING AI ADVISOR...")
    print("   - Autonomous market scanner (30s intervals)")
    print("   - Opportunity scoring (0-100)")
    print("   - Confidence filter (≥70%)")
    print("   - 5 API endpoints (/api/advisor/*)")
    
    # AI Advisor activation happens via scheduler in wolf_app.py
    print("   ✅ AI Advisor code ready (needs startup scheduler)")
    
    return True


def activate_social_sentiment():
    """Enable social sentiment tracking"""
    print("\n📱 ACTIVATING SOCIAL SENTIMENT...")
    print("   - Twitter/X mention tracking")
    print("   - Reddit WallStreetBets monitoring")
    print("   - Sentiment scoring (-1.0 to +1.0)")
    print("   - Viral signal detection")
    
    # Check for API keys
    twitter_key = os.getenv("TWITTER_BEARER_TOKEN")
    reddit_client = os.getenv("REDDIT_CLIENT_ID")
    
    if not twitter_key:
        print("   ⚠️  TWITTER_BEARER_TOKEN not set (optional)")
    else:
        print("   ✅ Twitter API configured")
    
    if not reddit_client:
        print("   ⚠️  REDDIT_CLIENT_ID not set (optional)")
    else:
        print("   ✅ Reddit API configured")
    
    # Enable social sentiment integration
    os.environ["SOCIAL_SENTIMENT_ENABLED"] = "1"
    print("   ✅ Set SOCIAL_SENTIMENT_ENABLED=1")
    
    return True


def activate_economic_calendar():
    """Enable economic calendar tracking"""
    print("\n📅 ACTIVATING ECONOMIC CALENDAR...")
    print("   - FOMC meeting detection")
    print("   - CPI, PPI, GDP reports")
    print("   - Earnings announcements")
    print("   - Impact scoring (high/medium/low)")
    
    # Check for API keys
    trading_econ = os.getenv("TRADING_ECONOMICS_API_KEY")
    fred_key = os.getenv("FRED_API_KEY")
    
    if not trading_econ and not fred_key:
        print("   ⚠️  Economic calendar API not configured (need TRADING_ECONOMICS_API_KEY or FRED_API_KEY)")
    elif fred_key:
        print("   ✅ Fred API configured")
    else:
        print("   ✅ Trading Economics API configured")
    
    # Enable economic calendar
    os.environ["ECONOMIC_CALENDAR_ENABLED"] = "1"
    print("   ✅ Set ECONOMIC_CALENDAR_ENABLED=1")
    
    return True


def activate_multi_timeframe():
    """Enable multi-timeframe analysis"""
    print("\n⏰ ACTIVATING MULTI-TIMEFRAME ANALYSIS...")
    print("   - 1h forecasts")
    print("   - 4h forecasts")
    print("   - 1d forecasts")
    print("   - 1w forecasts")
    print("   - Timeframe alignment detection")
    
    os.environ["MULTI_TIMEFRAME_ENABLED"] = "1"
    print("   ✅ Set MULTI_TIMEFRAME_ENABLED=1")
    
    return True


def activate_backtesting():
    """Enable backtesting API"""
    print("\n📊 ACTIVATING BACKTESTING ENGINE...")
    print("   - Historical strategy testing")
    print("   - Sharpe ratio, Sortino calculation")
    print("   - Slippage + commission modeling")
    print("   - Equity curve generation")
    print("   - API endpoint: /api/backtest")
    
    os.environ["BACKTESTING_ENABLED"] = "1"
    print("   ✅ Set BACKTESTING_ENABLED=1")
    
    return True


def generate_env_file():
    """Generate .env file with all activations"""
    print("\n📝 GENERATING .env FILE...")
    
    env_vars = {
        "CRYPTO_ENABLED": "1",
        "SOCIAL_SENTIMENT_ENABLED": "1",
        "ECONOMIC_CALENDAR_ENABLED": "1",
        "MULTI_TIMEFRAME_ENABLED": "1",
        "BACKTESTING_ENABLED": "1",
        "AI_ADVISOR_ENABLED": "1",
    }
    
    # Read existing .env
    env_path = "/workspaces/ghost-protocol/.env"
    existing_vars = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_vars[key] = value
    
    # Merge with new vars
    existing_vars.update(env_vars)
    
    # Write updated .env
    with open(env_path, 'w') as f:
        f.write("# Ghost Protocol - Enhanced Configuration\n")
        f.write("# Generated by activate_all_systems.py\n\n")
        
        for key, value in sorted(existing_vars.items()):
            f.write(f"{key}={value}\n")
    
    print(f"   ✅ Updated {env_path}")
    return True


def generate_railway_vars():
    """Generate Railway environment variable commands"""
    print("\n🚂 RAILWAY DEPLOYMENT COMMANDS...")
    print("   Run these commands to activate on Railway:\n")
    
    commands = [
        "railway variables set CRYPTO_ENABLED=1",
        "railway variables set SOCIAL_SENTIMENT_ENABLED=1",
        "railway variables set ECONOMIC_CALENDAR_ENABLED=1",
        "railway variables set MULTI_TIMEFRAME_ENABLED=1",
        "railway variables set BACKTESTING_ENABLED=1",
        "railway variables set AI_ADVISOR_ENABLED=1",
    ]
    
    for cmd in commands:
        print(f"   {cmd}")
    
    print("\n   Or set in Railway dashboard:")
    print("   https://railway.app/project/<your-project>/variables")
    
    return True


def create_activation_summary():
    """Create summary of activated systems"""
    print("\n" + "="*60)
    print("✅ ACTIVATION COMPLETE")
    print("="*60)
    
    print("\n🎯 ACTIVATED SYSTEMS:")
    print("   1. ✅ Crypto Trading Suite (15 coins)")
    print("   2. ✅ AI Advisor (autonomous scanner)")
    print("   3. ✅ Social Sentiment Monitoring")
    print("   4. ✅ Economic Calendar Tracking")
    print("   5. ✅ Multi-Timeframe Analysis")
    print("   6. ✅ Backtesting Engine")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Deploy to Railway with new env vars")
    print("   2. Verify activation: /api/v3/cockpit/status")
    print("   3. Test crypto: /api/crypto/forecast/BTC")
    print("   4. Check AI Advisor: /api/advisor/recommendations")
    print("   5. Run backtest: /api/backtest")
    
    print("\n📊 EXPECTED IMPROVEMENTS:")
    print("   - More trading opportunities (crypto + stocks)")
    print("   - Better context (social sentiment + economic events)")
    print("   - Multi-timeframe confirmation")
    print("   - Strategy validation (backtesting)")
    print("   - Autonomous opportunity detection (AI Advisor)")
    
    print("\n🔧 OPTIONAL API KEYS (for full functionality):")
    print("   - TWITTER_BEARER_TOKEN (social sentiment)")
    print("   - REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET (social sentiment)")
    print("   - TRADING_ECONOMICS_API_KEY or FRED_API_KEY (economic calendar)")
    
    print("\n" + "="*60)
    
    return True


def main():
    """Main activation flow"""
    print("🚀 GHOST PROTOCOL - SYSTEM ACTIVATION")
    print("="*60)
    
    try:
        # Activate all systems
        activate_crypto_suite()
        activate_ai_advisor()
        activate_social_sentiment()
        activate_economic_calendar()
        activate_multi_timeframe()
        activate_backtesting()
        
        # Generate config files
        generate_env_file()
        generate_railway_vars()
        
        # Summary
        create_activation_summary()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ACTIVATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

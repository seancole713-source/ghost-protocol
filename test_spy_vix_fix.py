#!/usr/bin/env python3
"""
Quick test to verify SPY/VIX fallback is working with yfinance.
"""

import sys
sys.path.insert(0, '/app')

def test_yfinance_directly():
    """Test yfinance directly to see if it works"""
    print("=" * 80)
    print("TESTING YFINANCE DIRECTLY")
    print("=" * 80)
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
        
        # Test SPY
        print("\n1. Testing SPY:")
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="2d")
            if not spy_data.empty:
                price = float(spy_data['Close'].iloc[-1])
                print(f"   ✅ SPY: ${price:.2f}")
            else:
                print("   ❌ SPY: No data returned")
        except Exception as e:
            print(f"   ❌ SPY error: {e}")
        
        # Test ^GSPC (S&P 500 index)
        print("\n2. Testing ^GSPC (S&P 500 Index):")
        try:
            gspc = yf.Ticker("^GSPC")
            gspc_data = gspc.history(period="2d")
            if not gspc_data.empty:
                price = float(gspc_data['Close'].iloc[-1])
                print(f"   ✅ ^GSPC: ${price:.2f}")
            else:
                print("   ❌ ^GSPC: No data returned")
        except Exception as e:
            print(f"   ❌ ^GSPC error: {e}")
        
        # Test ^VIX
        print("\n3. Testing ^VIX (Volatility Index):")
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="2d")
            if not vix_data.empty:
                level = float(vix_data['Close'].iloc[-1])
                print(f"   ✅ ^VIX: {level:.2f}")
            else:
                print("   ❌ ^VIX: No data returned")
        except Exception as e:
            print(f"   ❌ ^VIX error: {e}")
            
    except ImportError as e:
        print(f"❌ Cannot import yfinance: {e}")
        print("   Run: pip install yfinance")
        return False
    
    return True


def test_world_context():
    """Test the world_context module"""
    print("\n" + "=" * 80)
    print("TESTING WORLD CONTEXT MODULE")
    print("=" * 80)
    
    try:
        from core.world_context import get_world_context
        print("✅ world_context module imported")
        
        context = get_world_context()
        
        print("\n📊 SPY Data:")
        spy = context.get("spy", {})
        spy_price = spy.get("price")
        spy_change = spy.get("change_pct")
        spy_provider = spy.get("provider")
        
        if spy_price:
            print(f"   ✅ Price: ${spy_price}")
            print(f"   ✅ Change: {spy_change:+.2f}%" if spy_change else "   ⚠️  Change: N/A")
            print(f"   ✅ Provider: {spy_provider}")
        else:
            print("   ❌ SPY price is NULL")
        
        print("\n📊 VIX Data:")
        vix = context.get("vix", {})
        vix_level = vix.get("level")
        vix_change = vix.get("change")
        vix_status = vix.get("status")
        
        if vix_level:
            print(f"   ✅ Level: {vix_level}")
            print(f"   ✅ Change: {vix_change:+.2f}" if vix_change else "   ⚠️  Change: N/A")
            print(f"   ✅ Status: {vix_status}")
        else:
            print("   ❌ VIX level is NULL")
        
        print("\n📊 Market Mood:")
        mood = context.get("market_mood", {})
        print(f"   Sentiment: {mood.get('sentiment', 'unknown')}")
        print(f"   Score: {mood.get('score', 0)}")
        print(f"   Factors: {', '.join(mood.get('factors', []))}")
        
        # Final verdict
        print("\n" + "=" * 80)
        print("VERDICT:")
        print("=" * 80)
        
        if spy_price and vix_level:
            print("✅ BOTH SPY AND VIX WORKING!")
            print(f"   SPY: ${spy_price} via {spy_provider}")
            print(f"   VIX: {vix_level} ({vix_status})")
            return True
        elif spy_price:
            print("⚠️  SPY working, VIX failed")
            return False
        elif vix_level:
            print("⚠️  VIX working, SPY failed")
            return False
        else:
            print("❌ BOTH SPY AND VIX FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error testing world_context: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Testing SPY/VIX Fix\n")
    
    # Test yfinance directly first
    yf_works = test_yfinance_directly()
    
    # Test world_context module
    context_works = test_world_context()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"yfinance direct test: {'✅ PASS' if yf_works else '❌ FAIL'}")
    print(f"world_context test: {'✅ PASS' if context_works else '❌ FAIL'}")
    
    sys.exit(0 if (yf_works and context_works) else 1)

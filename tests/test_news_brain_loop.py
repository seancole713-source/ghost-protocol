#!/usr/bin/env python3
"""
Test News Brain loop startup.

Verifies that the News Brain analysis loop starts correctly
and the diagnostic logging appears.
"""

import asyncio
import os
import sys


async def test_news_brain_startup():
    """Test News Brain startup sequence"""
    print("=" * 60)
    print("🧪 Testing News Brain Loop Startup")
    print("=" * 60)
    
    # Set environment variables
    os.environ["NEWS_ANALYSIS_ENABLED"] = "1"
    os.environ["NEWS_ANALYSIS_INTERVAL_MINUTES"] = "5"  # 5 min for testing
    
    print("\n📝 Environment Variables:")
    print(f"   NEWS_ANALYSIS_ENABLED = {os.getenv('NEWS_ANALYSIS_ENABLED')}")
    print(f"   NEWS_ANALYSIS_INTERVAL_MINUTES = {os.getenv('NEWS_ANALYSIS_INTERVAL_MINUTES')}")
    
    # Test import
    print("\n🔍 Testing import...")
    try:
        from core.intelligence.ghost_news_brain import get_news_brain
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test News Brain instantiation
    print("\n🔍 Testing News Brain instantiation...")
    try:
        brain = get_news_brain()
        print(f"✅ News Brain created: {brain}")
    except Exception as e:
        print(f"❌ Failed to create News Brain: {e}")
        return False
    
    # Test loop creation (without actually running it for 30 minutes)
    print("\n🔍 Testing loop function creation...")
    try:
        NEWS_ANALYSIS_INTERVAL_MINUTES = int(os.getenv("NEWS_ANALYSIS_INTERVAL_MINUTES", "30"))
        
        async def _news_analysis_loop():
            """Automatic news analysis every 30 minutes"""
            print(f"📰 News Analysis Loop: STARTING (every {NEWS_ANALYSIS_INTERVAL_MINUTES} min)")
            
            # Run once for testing
            try:
                print("📰 Running automatic news analysis...")
                brain = get_news_brain()
                result = await brain.analyze_news()
                
                major_events = result.get("major_events", [])
                predictions_at_risk = result.get("predictions_at_risk", [])
                
                print(
                    f"📰 News analysis complete: {len(major_events)} events, "
                    f"{len(predictions_at_risk)} predictions at risk"
                )
                
                return True
            except Exception as e:
                print(f"❌ News analysis error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Create task
        task = asyncio.create_task(_news_analysis_loop())
        print("✅ Loop task created")
        
        # Wait for first run to complete
        print("\n⏳ Waiting for first analysis to complete...")
        result = await task
        
        if result:
            print("\n✅ News Brain loop successfully executed!")
            return True
        else:
            print("\n❌ News Brain loop failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create loop: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_news_brain_startup())
    
    print("\n" + "=" * 60)
    if success:
        print("✅ NEWS BRAIN LOOP TEST PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ NEWS BRAIN LOOP TEST FAILED")
        print("=" * 60)
        sys.exit(1)

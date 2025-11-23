#!/usr/bin/env python3
"""
Initialize default trading goals in goals.db
Sets realistic targets based on typical trading objectives
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.goals_tracker import GoalsTracker

def main():
    print("🎯 Initializing Ghost Protocol Goals...")
    
    # Default goals (conservative but achievable)
    goals = [
        ("daily", 500.0),      # $500/day = $10k/month at 20 trading days
        ("weekly", 2500.0),    # $2.5k/week = ~$10k/month  
        ("monthly", 10000.0),  # $10k/month = $120k/year
        ("yearly", 120000.0),  # $120k/year target
    ]
    
    try:
        tracker = GoalsTracker()
        
        # Check existing goals
        existing = tracker.get_all_goals()
        has_goals = any(g['target'] > 0 for g in existing.values())
        
        if has_goals:
            print(f"✓ Goals already configured:")
            for period in ["daily", "weekly", "monthly", "yearly"]:
                g = existing[period]
                print(f"   {period.capitalize()}: ${g['target']:,.2f} (current: ${g['current']:,.2f})")
            print("\nSkipping initialization (use force_init_goals.py to reset)")
            return
        
        # Set new goals
        created = 0
        for period, target in goals:
            result = tracker.set_goal(period, target)
            print(f"  ✓ {period.capitalize()}: ${target:,.2f}")
            created += 1
        
        # Verify
        final = tracker.get_all_goals()
        print(f"\n✅ Goals initialized: {created} periods configured")
        print(f"   Total yearly target: ${final['yearly']['target']:,.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

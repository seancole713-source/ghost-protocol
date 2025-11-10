#!/usr/bin/env python3
"""Add WOLF to watchlist"""

import os
import sys

# Add Ghost to path
sys.path.insert(0, "/workspaces/GHOST")
os.chdir("/workspaces/GHOST")

try:
    from core.watchlist_manager import WatchlistManager

    print("📋 Adding WOLF to watchlist...")
    wm = WatchlistManager()
    result = wm.add_symbol("WOLF", name="Wolfspeed Inc.")
    print(f"✅ WOLF added to watchlist: {result}")

    # Verify
    watchlist = wm.get_watchlist()
    wolf_in_list = any(s.get("symbol") == "WOLF" for s in watchlist)
    print(f"✅ Verification: WOLF in watchlist = {wolf_in_list}")
    print(f"✅ Total symbols in watchlist: {len(watchlist)}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

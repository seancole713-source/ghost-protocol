#!/usr/bin/env python3
"""
Add all user's stocks to Ghost personal watchlist
"""
import requests
import time

BASE_URL = "https://ghost-protocol-production.up.railway.app"
AUTH_HEADER = {"Authorization": "Bearer 4V0bC_tThjnq7G81lKrWOb0ym6pzB7ZA40To1Sln_7w"}

# User's stocks to add
STOCKS = [
    # Tech giants
    "AMD", "NFLX", "LLY", "PLTR", "ORCL", "GOOG", "GOOGL", "AMZN", "MU",
    
    # Gold miners
    "BTG", "KGC", "FSM", "GOLD", "NEM", "EGO", "AEM", "GFI",
    
    # Silver miners
    "PAAS", "AG", "CDE", "SSRM", "HL",
    
    # Small caps / Speculative
    "RARE", "BMBL", "GME", "SOUN", "ABCL", "TLRY", "DJT", "NOK", "HIMS",
    "TGTX", "DUOL", "CVNA", "ITRI", "XPO", "YMM", "RIOT", "SPCE", "OTRK",
    "AMC", "WOLF", "IQ", "TME", "ARCT", "RDFN", "BILL", "TAL", "PFE",
    "PLTK", "HOOD",
    
    # Additional tech/growth
    "NVDA", "TSLA", "META", "MSFT", "AAPL", "CRM", "NOW", "SNOW", "DDOG",
    "NET", "ZS", "CRWD", "PANW",
    
    # More growth names
    "SQ", "PYPL", "SHOP", "COIN", "AFRM", "UPST", "SOFI", "PATH",
    "SMCI", "ARM", "MRVL",
]

# Remove duplicates
STOCKS = list(set(STOCKS))

print(f"📋 Adding {len(STOCKS)} stocks to Ghost Watchlist...")
print("=" * 60)

success = 0
failed = 0
failed_symbols = []

for symbol in sorted(STOCKS):
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v3/watchlist/add",
            headers=AUTH_HEADER,
            json={"symbol": symbol, "asset_type": "stock"},
            timeout=10
        )
        
        if resp.status_code == 200:
            data = resp.json()
            action = data.get("action", "added")
            print(f"✅ {symbol}: {action}")
            success += 1
        else:
            print(f"❌ {symbol}: HTTP {resp.status_code} - {resp.text[:100]}")
            failed += 1
            failed_symbols.append(symbol)
            
    except Exception as e:
        print(f"❌ {symbol}: {e}")
        failed += 1
        failed_symbols.append(symbol)
    
    # Small delay to avoid rate limiting
    time.sleep(0.1)

print()
print("=" * 60)
print(f"✅ Success: {success}")
print(f"❌ Failed: {failed}")

if failed_symbols:
    print(f"\nFailed symbols: {', '.join(failed_symbols)}")

# Get current watchlist count
try:
    resp = requests.get(f"{BASE_URL}/api/v3/watchlist/user", headers=AUTH_HEADER, timeout=10)
    if resp.status_code == 200:
        items = resp.json().get("items", [])
        stocks = [i for i in items if i.get("type") == "stock"]
        cryptos = [i for i in items if i.get("type") == "crypto"]
        print(f"\n📊 Current Watchlist: {len(stocks)} stocks, {len(cryptos)} cryptos")
except:
    pass

print("\n✨ Done!")

#!/usr/bin/env python3
"""
Generate and send a REAL Ghost prediction via API call
"""

import os
import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "940596997")

# Try Railway first, fallback to localhost
GHOST_API_URLS = [
    "https://ghost-protocol-production.up.railway.app",
    "http://localhost:18100"
]

def send_telegram(message: str):
    """Send message via Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

def get_ghost_api_url():
    """Find working Ghost API endpoint"""
    for url in GHOST_API_URLS:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Found Ghost API at: {url}")
                return url
        except:
            continue
    return None

def generate_real_prediction(api_url: str, symbol: str):
    """Generate a REAL prediction using Ghost API"""
    print(f"\n{'='*60}")
    print(f"GENERATING REAL PREDICTION FOR {symbol}")
    print(f"{'='*60}\n")
    
    try:
        # Call Ghost prediction API
        print(f"[1/2] Calling Ghost API: {api_url}/api/predict/run?symbol={symbol}")
        
        response = requests.get(
            f"{api_url}/api/predict/run",
            params={"symbol": symbol},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ API error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
        
        prediction_result = response.json()
        
        if not prediction_result or 'error' in prediction_result:
            print(f"❌ Prediction failed: {prediction_result.get('error', 'Unknown error')}")
            return None
        
        print(f"✅ Prediction complete!")
        
        # Extract data
        direction = prediction_result.get('direction', 'FLAT')
        confidence = prediction_result.get('confidence', 0)
        current_price = prediction_result.get('current_price', 0)
        target_price = prediction_result.get('target_price', current_price)
        feature_count = prediction_result.get('feature_count', 0)
        
        print(f"   Direction: {direction}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Features: {feature_count}")
        
        print(f"\n[2/2] Formatting Telegram message...")
        
        direction_emoji = {
            'UP': '📈 ⬆️',
            'DOWN': '📉 ⬇️',
            'FLAT': '➡️'
        }
        
        # Calculate percentage change
        pct_change = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Get market type
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'SHIB', 'PEPE']
        market = "crypto" if any(symbol.startswith(c) for c in crypto_symbols) else "stock"
        
        message = f"""🔮 *GHOST LIVE PREDICTION*

*Symbol:* {symbol}
*Direction:* {direction_emoji.get(direction, '➡️')} {direction}
*Confidence:* {confidence:.1f}%
*Current Price:* ${current_price:.2f}
*Target (24h):* ${target_price:.2f} ({pct_change:+.1f}%)

━━━━━━━━━━━━━━━━━━━━

📊 *ANALYSIS*

*Features:* {feature_count} indicators
*Provider:* {"Binance Public (FREE)" if market == "crypto" else "Yahoo Finance (FREE)"}
*Time:* {datetime.now().strftime('%H:%M UTC')}

━━━━━━━━━━━━━━━━━━━━

💡 *SIGNAL STRENGTH*

{("🟢 *Strong Signal*" if confidence >= 70 else "🟡 *Moderate Signal*" if confidence >= 55 else "🔴 *Weak Signal*")}

{("High conviction - consider position" if confidence >= 70 else "Monitor closely - wait for confirmation" if confidence >= 55 else "Low conviction - hold off")}

*Direction:* {direction}
{("Bullish bias suggested" if direction == 'UP' else "Bearish bias suggested" if direction == 'DOWN' else "Neutral - wait for clearer signal")}

━━━━━━━━━━━━━━━━━━━━

🤖 Ghost AI | FREE-TIER
Live prediction • $0/month
"""
        
        return {
            'symbol': symbol,
            'message': message,
            'prediction': prediction_result
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Find Ghost API
    api_url = get_ghost_api_url()
    if not api_url:
        print("❌ Ghost API not found. Is Ghost running?")
        exit(1)
    
    # Test with multiple symbols
    test_symbols = ['AAPL', 'MSFT', 'BTC']
    
    for symbol in test_symbols:
        result = generate_real_prediction(api_url, symbol)
        
        if result:
            print(f"\n{'='*60}")
            print(f"SENDING TO TELEGRAM: {symbol}")
            print(f"{'='*60}")
            
            if send_telegram(result['message']):
                print(f"✅ {symbol} prediction sent to Telegram!")
                print("\nCheck your Telegram to see the REAL prediction format!")
                break  # Send only first successful one
            else:
                print(f"❌ Failed to send {symbol} to Telegram")
        else:
            print(f"❌ Failed to generate prediction for {symbol}")
            print(f"Trying next symbol...")
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")

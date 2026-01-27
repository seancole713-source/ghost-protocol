#!/usr/bin/env python3
"""
Generate and send a REAL Ghost prediction using current live data
"""

import os
import sys
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/Users/studio713/ghost-protocol')

# Import Ghost's actual prediction function
from wolf_app import run_prediction

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "940596997")

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

def generate_real_prediction(symbol: str):
    """Generate a REAL prediction using Ghost's production prediction engine"""
    print(f"\n{'='*60}")
    print(f"GENERATING REAL PREDICTION FOR {symbol}")
    print(f"{'='*60}\n")
    
    try:
        # Use Ghost's actual prediction function from wolf_app.py
        print(f"[1/2] Running Ghost prediction engine for {symbol}...")
        
        # Determine market type
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'SHIB', 'PEPE', 'MATIC', 'LINK', 'UNI', 'AVAX']
        market = "crypto" if any(symbol.startswith(c) for c in crypto_symbols) else "stock"
        
        prediction_result = run_prediction(symbol, market=market, horizon="SHORT")
        
        if not prediction_result or 'error' in prediction_result:
            print(f"❌ Prediction failed: {prediction_result.get('error', 'Unknown error')}")
            return None
        
        print(f"✅ Prediction complete!")
        print(f"   Direction: {prediction_result.get('direction', 'N/A')}")
        print(f"   Confidence: {prediction_result.get('confidence', 0):.1f}%")
        
        # Extract data
        direction = prediction_result.get('direction', 'FLAT')
        confidence = prediction_result.get('confidence', 0)
        current_price = prediction_result.get('current_price', 0)
        target_price = prediction_result.get('target_price', current_price)
        feature_count = prediction_result.get('feature_count', 0)
        
        print(f"\n[2/2] Formatting Telegram message...")
        
        direction_emoji = {
            'UP': '📈 ⬆️',
            'DOWN': '📉 ⬇️',
            'FLAT': '➡️'
        }
        
        # Calculate percentage change
        pct_change = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        message = f"""🔮 *GHOST LIVE PREDICTION*

*Symbol:* {symbol}
*Direction:* {direction_emoji.get(direction, '➡️')} {direction}
*Confidence:* {confidence:.1f}%
*Current Price:* ${current_price:.2f}
*Target (24h):* ${target_price:.2f} ({pct_change:+.1f}%)

━━━━━━━━━━━━━━━━━━━━

📊 *ANALYSIS BREAKDOWN*

*Features Extracted:* {feature_count}
*Data Sources:* {"Binance Public (FREE)" if market == "crypto" else "Yahoo Finance (FREE)"}
*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

━━━━━━━━━━━━━━━━━━━━

💡 *SIGNAL STRENGTH*

*Confidence {confidence:.1f}%:*
{"🟢 Strong signal - High conviction" if confidence >= 70 else "🟡 Moderate signal - Monitor closely" if confidence >= 55 else "🔴 Weak signal - Low conviction"}

*Direction {direction}:*
{"Suggest bullish position" if direction == 'UP' else "Suggest bearish position" if direction == 'DOWN' else "Market neutral - wait for clearer signal"}

━━━━━━━━━━━━━━━━━━━━

🤖 Ghost AI | FREE-TIER
Real-time prediction using live market data
Cost: $0/month
"""
        
        return {
            'symbol': symbol,
            'message': message,
            'prediction': prediction_result,
            'current_price': current_price,
            'feature_count': feature_count
        }
        
    except Exception as e:
        print(f"❌ Error generating prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with multiple symbols
    test_symbols = ['AAPL', 'MSFT', 'BTC']
    
    successful = 0
    for symbol in test_symbols:
        result = generate_real_prediction(symbol)
        
        if result:
            print(f"\n{'='*60}")
            print(f"SENDING TO TELEGRAM: {symbol}")
            print(f"{'='*60}")
            
            if send_telegram(result['message']):
                print(f"✅ {symbol} prediction sent to Telegram!")
                successful += 1
                break  # Send only the first successful one
            else:
                print(f"❌ Failed to send {symbol} to Telegram")
        else:
            print(f"❌ Failed to generate prediction for {symbol}")
    
    if successful == 0:
        print("\n❌ No predictions could be generated or sent")
    else:
        print(f"\n✅ Successfully sent {successful} real prediction(s)")


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

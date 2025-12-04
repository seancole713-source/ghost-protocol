// ============================================================================
// GHOST PROTOCOL COCKPIT - BROWSER DIAGNOSTIC SCRIPT
// ============================================================================
// INSTRUCTIONS:
//   1. Open https://ghost-protocol-production.up.railway.app/cockpit
//   2. Open DevTools (F12) → Console tab
//   3. Copy this ENTIRE script
//   4. Paste into Console and press Enter
//   5. Copy all output and send to developer
// ============================================================================

console.clear();
console.log('%c🔍 GHOST PROTOCOL COCKPIT DIAGNOSTIC', 'font-size:20px;font-weight:bold;color:#00ff00');
console.log('=' + '='.repeat(69));

// 1. CHECK FOR SYNTAX ERRORS
console.log('\n%c1️⃣  SYNTAX ERROR CHECK', 'font-size:16px;font-weight:bold;color:#ffaa00');
const errors = window.console.memory ? 'No direct error access' : 'Console API limited';
console.log(`   Console state: ${errors}`);
console.log('   ✅ If you can read this, no fatal syntax errors');

// 2. CHECK JAVASCRIPT FUNCTIONS LOADED
console.log('\n%c2️⃣  FUNCTION AVAILABILITY', 'font-size:16px;font-weight:bold;color:#ffaa00');
const functions = [
    'initializeApp',
    'loadTopMovers', 
    'loadForecast',
    'loadVIPCoins',
    'loadWatchlistByMode',
    'loadHealthScore',
    'updateSystemTime',
    'controlAction'
];

let functionsOK = 0;
functions.forEach(fn => {
    const exists = typeof window[fn] === 'function';
    console.log(`   ${exists ? '✅' : '❌'} ${fn}()`);
    if (exists) functionsOK++;
});

console.log(`\n   📊 ${functionsOK}/${functions.length} functions available`);

// 3. CHECK DOM ELEMENTS
console.log('\n%c3️⃣  DOM ELEMENTS', 'font-size:16px;font-weight:bold;color:#ffaa00');
const elements = [
    'system-time',
    'movers-list',
    'vip-list',
    'forecast-grid',
    'news-list',
    'watchlist-table',
    'health-score-value',
    'btn-start',
    'btn-stop',
    'btn-reset'
];

let elementsOK = 0;
elements.forEach(id => {
    const el = document.getElementById(id);
    console.log(`   ${el ? '✅' : '❌'} #${id}`);
    if (el) elementsOK++;
});

console.log(`\n   📊 ${elementsOK}/${elements.length} elements found`);

// 4. TEST API ENDPOINTS
console.log('\n%c4️⃣  API ENDPOINTS', 'font-size:16px;font-weight:bold;color:#ffaa00');
console.log('   Testing endpoints (this may take 5-10 seconds)...\n');

const endpoints = [
    '/api/v3/cockpit/status',
    '/api/v3/hunter/feed?limit=3',
    '/api/v3/predictions/latest?symbol=BTC',
    '/api/v3/watchlist/enriched',
    '/api/v3/goals/snapshot',
    '/api/v3/news/feed?limit=5'
];

Promise.all(endpoints.map(url => 
    fetch(url)
        .then(r => ({ url, status: r.status, ok: r.ok, data: r.ok ? r.json() : null }))
        .then(async result => {
            const data = result.data ? await result.data : null;
            return { ...result, data };
        })
        .catch(err => ({ url, status: 'ERROR', ok: false, error: err.message }))
)).then(results => {
    results.forEach(r => {
        console.log(`   ${r.ok ? '✅' : '❌'} ${r.url}: ${r.status}`);
        if (r.data) {
            // Show sample data structure
            const keys = Object.keys(r.data).slice(0, 3).join(', ');
            console.log(`      Keys: ${keys}`);
        }
    });
    
    const apisOK = results.filter(r => r.ok).length;
    console.log(`\n   📊 ${apisOK}/${results.length} endpoints working`);
    
    // 5. TEST BUTTON EVENT HANDLERS
    console.log('\n%c5️⃣  BUTTON EVENT HANDLERS', 'font-size:16px;font-weight:bold;color:#ffaa00');
    
    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    const resetBtn = document.getElementById('btn-reset');
    
    if (startBtn && stopBtn && resetBtn) {
        console.log('   ℹ️  Now click each button and check for logs below:');
        console.log('   ℹ️  You should see "[CONTROL] START clicked" etc.\n');
        
        // Add temporary test handlers
        const testHandler = (btn, name) => {
            const originalOnclick = btn.onclick;
            btn.addEventListener('click', function(e) {
                console.log(`   🔘 ${name} button CLICKED (handler fired)`);
            }, { once: false, capture: true });
        };
        
        testHandler(startBtn, 'START');
        testHandler(stopBtn, 'STOP');
        testHandler(resetBtn, 'RESET');
        
        console.log('   ✅ Test handlers attached');
    } else {
        console.log('   ❌ Buttons not found in DOM');
    }
    
    // 6. CHECK CURRENT UI STATE
    console.log('\n%c6️⃣  CURRENT UI STATE', 'font-size:16px;font-weight:bold;color:#ffaa00');
    
    const timer = document.getElementById('system-time');
    const healthScore = document.getElementById('health-score-value');
    const moversList = document.getElementById('movers-list');
    const vipList = document.getElementById('vip-list');
    
    console.log(`   Timer: ${timer ? timer.textContent : 'NOT FOUND'}`);
    console.log(`   Health Score: ${healthScore ? healthScore.textContent : 'NOT FOUND'}`);
    console.log(`   Top Movers rows: ${moversList ? moversList.children.length : 'NOT FOUND'}`);
    console.log(`   VIP Coins rows: ${vipList ? vipList.children.length : 'NOT FOUND'}`);
    
    // 7. SUMMARY
    console.log('\n%c7️⃣  DIAGNOSTIC SUMMARY', 'font-size:16px;font-weight:bold;color:#00ff00');
    console.log('=' + '='.repeat(69));
    
    const totalTests = functionsOK + elementsOK + apisOK;
    const totalPossible = functions.length + elements.length + endpoints.length;
    const percentage = Math.round((totalTests / totalPossible) * 100);
    
    console.log(`\n   📊 Overall Health: ${percentage}% (${totalTests}/${totalPossible} checks passed)`);
    
    if (percentage >= 90) {
        console.log('\n   ✅ COCKPIT IS OPERATIONAL');
        console.log('      - If you still see empty panels, try:');
        console.log('        1. Click START button');
        console.log('        2. Wait 5-10 seconds for data to load');
        console.log('        3. Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)');
    } else if (percentage >= 50) {
        console.log('\n   ⚠️  COCKPIT PARTIALLY OPERATIONAL');
        console.log('      - Some features may not work');
        console.log('      - Copy this output and send to developer');
    } else {
        console.log('\n   ❌ COCKPIT NOT OPERATIONAL');
        console.log('      - Critical initialization failure');
        console.log('      - COPY ALL OUTPUT ABOVE and send to developer');
    }
    
    console.log('\n   🔄 To run diagnostic again: paste this script again');
    console.log('   📋 To manually initialize: type initializeApp() and press Enter');
    console.log('\n' + '='.repeat(70));
    
}).catch(err => {
    console.error('\n❌ Diagnostic failed:', err);
});

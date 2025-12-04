// COCKPIT BROWSER DIAGNOSTIC SCRIPT
// Open https://ghost-protocol-production.up.railway.app/cockpit
// Then paste this into browser console:

console.log('🔍 GHOST PROTOCOL COCKPIT DIAGNOSTIC');
console.log('=' .repeat(70));

// 1. Check if DOMContentLoaded fired
console.log('\n1. DOM STATE:');
console.log('   Document ready state:', document.readyState);
console.log('   Body loaded:', !!document.body);

// 2. Check if main functions exist
console.log('\n2. JAVASCRIPT FUNCTIONS:');
const functions = ['initializeApp', 'loadTopMovers', 'loadForecast', 'loadVIPCoins', 
                   'loadWatchlistByMode', 'loadHealthScore', 'updateSystemTime', 'controlAction'];
functions.forEach(fn => {
    console.log(`   ${typeof window[fn] === 'function' ? '✅' : '❌'} ${fn}()`);
});

// 3. Check DOM elements
console.log('\n3. DOM ELEMENTS:');
const elements = {
    'system-time': 'Timer display',
    'btn-start': 'START button',
    'btn-stop': 'STOP button',
    'btn-reset': 'RESET button',
    'movers-list': 'Top Movers container',
    'vip-list': 'VIP Coins container',
    'forecast-grid': 'Forecast container',
    'news-list': 'News container',
    'watchlist-table': 'Watchlist container',
    'health-score-value': 'Health score'
};

for (const [id, desc] of Object.entries(elements)) {
    const el = document.getElementById(id);
    console.log(`   ${el ? '✅' : '❌'} #${id} (${desc})`);
    if (el && id === 'system-time') {
        console.log(`      Current value: "${el.textContent}"`);
    }
}

// 4. Check if event listeners are attached
console.log('\n4. EVENT LISTENERS:');
const btnStart = document.getElementById('btn-start');
if (btnStart) {
    // Try clicking programmatically
    console.log('   Testing START button...');
    const originalFetch = window.fetch;
    let fetchCalled = false;
    window.fetch = function(...args) {
        fetchCalled = true;
        console.log('   ✅ Fetch called:', args[0]);
        return originalFetch.apply(this, arguments);
    };
    
    btnStart.click();
    
    setTimeout(() => {
        console.log(`   ${fetchCalled ? '✅' : '❌'} Network request triggered`);
        window.fetch = originalFetch;
    }, 100);
}

// 5. Test API endpoints
console.log('\n5. TESTING API ENDPOINTS:');
const endpoints = [
    '/api/v3/cockpit/status',
    '/api/v3/hunter/feed?limit=3',
    '/api/v3/watchlist/enriched'
];

Promise.all(endpoints.map(url => 
    fetch(url)
        .then(r => r.json())
        .then(data => ({url, ok: true, data}))
        .catch(err => ({url, ok: false, error: err.message}))
)).then(results => {
    results.forEach(({url, ok, data, error}) => {
        if (ok) {
            console.log(`   ✅ ${url}`);
            if (data.movers) console.log(`      → ${data.movers.length} movers`);
            if (data.items) console.log(`      → ${data.items.length} items`);
            if (data.active !== undefined) console.log(`      → Active: ${data.active}`);
        } else {
            console.log(`   ❌ ${url}: ${error}`);
        }
    });
    
    console.log('\n' + '='.repeat(70));
    console.log('📊 DIAGNOSTIC COMPLETE');
    console.log('Copy the output above and send to developer');
    console.log('='.repeat(70));
});

// 6. Force initialize if not already done
console.log('\n6. FORCE INITIALIZATION:');
if (typeof initializeApp === 'function') {
    console.log('   Calling initializeApp() manually...');
    try {
        initializeApp();
        console.log('   ✅ Initialization complete');
        console.log('   Check if timer is now animating...');
    } catch (e) {
        console.log('   ❌ Error:', e.message);
    }
} else {
    console.log('   ❌ initializeApp() not found - JS file failed to load');
}

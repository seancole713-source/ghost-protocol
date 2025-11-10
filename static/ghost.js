// ghost.js - Unified fetch and UI utilities for Ghost Trading Bot

// ============================================================================
// THEME SWITCHER - Dark/Light Mode Toggle
// ============================================================================
(function initTheme() {
  // Load saved theme or default to dark
  const savedTheme = localStorage.getItem('ghost-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  if (document.body) {
    document.body.setAttribute('data-theme', savedTheme);
  }
  
  // Wait for DOM to be ready
  const setupToggle = () => {
    // Create theme toggle button
    const themeToggle = document.createElement('button');
    themeToggle.className = 'theme-toggle';
    themeToggle.setAttribute('aria-label', 'Toggle theme');
    themeToggle.setAttribute('title', 'Toggle dark/light mode');
    themeToggle.innerHTML = savedTheme === 'light' ? '🌙' : '☀️';
    
    // Toggle function
    themeToggle.onclick = () => {
      const root = document.documentElement;
      const currentTheme = root.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      root.setAttribute('data-theme', newTheme);
      document.body.setAttribute('data-theme', newTheme);
      localStorage.setItem('ghost-theme', newTheme);
      themeToggle.innerHTML = newTheme === 'light' ? '🌙' : '☀️';
    };
    
    // Insert into navbar (after spacer or at end)
    const navbar = document.querySelector('.navbar');
    if (navbar) {
      const spacer = navbar.querySelector('.spacer');
      if (spacer) {
        spacer.parentNode.insertBefore(themeToggle, spacer.nextSibling);
      } else {
        navbar.appendChild(themeToggle);
      }
    }
  };
  
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupToggle);
  } else {
    setupToggle();
  }
})();

// ============================================================================
// GHOST FETCH - Unified API calls with token support
// ============================================================================
function ghostFetch(url, options = {}) {
  const defaultOptions = {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  };
  const mergedOptions = { ...defaultOptions, ...options };
  try {
    const tok = localStorage.getItem('GHOST_API_TOKEN');
    if (tok) {
      mergedOptions.headers = mergedOptions.headers || {};
      mergedOptions.headers['Authorization'] = `Bearer ${tok}`;
    }
  } catch(_) { /* no-op */ }
  if (mergedOptions.body && typeof mergedOptions.body === 'object') {
    mergedOptions.body = JSON.stringify(mergedOptions.body);
  }
  return fetch(url, mergedOptions)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return response.json();
    })
    .catch(error => {
      console.error('ghostFetch error:', error);
      toast('Error: ' + error.message, 'error');
      throw error;
    });
}

function toast(message, type = 'info') {
  const toastEl = document.createElement('div');
  toastEl.className = `toast toast-${type}`;
  toastEl.textContent = message;
  toastEl.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${type === 'error' ? '#dc2626' : type === 'success' ? '#16a34a' : '#3b82f6'};
    color: white;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    z-index: 1000;
    opacity: 0;
    transition: opacity 0.3s;
  `;
  document.body.appendChild(toastEl);
  setTimeout(() => toastEl.style.opacity = '1', 10);
  setTimeout(() => {
    toastEl.style.opacity = '0';
    setTimeout(() => document.body.removeChild(toastEl), 300);
  }, 3000);
}

// Add toast styles to head
const style = document.createElement('style');
style.textContent = `
  .toast { font-family: system-ui; font-size: 0.9rem; }
`;
document.head.appendChild(style);

// Health banner styles injected here to avoid editing global CSS heavily
(function(){
  const s = document.createElement('style');
  s.textContent = `
    .health { font-family: system-ui; font-size: 0.85rem; padding: 6px 10px; border-radius: 6px; margin-right: 10px; }
    .health-loading { background:#e5e7eb; color:#111827; }
    .health-ok { background:#16a34a; color:white; }
    .health-warn { background:#f59e0b; color:black; }
    .health-bad { background:#dc2626; color:white; }
    .health-bar { display:flex; align-items:center; gap:.5rem; margin: 12px 0; }
    .health-pill { font-family: system-ui; font-size: 0.8rem; padding: 4px 8px; border-radius: 999px; background:#e5e7eb; color:#111827; }
    .hp-ok { background:#16a34a; color:white; }
    .hp-warn { background:#f59e0b; color:black; }
    .hp-bad { background:#dc2626; color:white; }
    .health-title { font-weight:600; margin-right:.25rem; }
  `;
  document.head.appendChild(s);
})();

// Insert a lightweight health badge bar that polls /health
function initHealthBadge(options = {}) {
  try {
    const container = document.createElement('div');
    container.className = 'health-bar';
    // Overall badge
    const overall = document.createElement('span');
    overall.className = 'health health-loading';
  overall.textContent = '';
    // Per-sink pills
    const pricePill = document.createElement('span');
    pricePill.className = 'health-pill';
    pricePill.textContent = 'Price';
    const newsPill = document.createElement('span');
    newsPill.className = 'health-pill';
    newsPill.textContent = 'News';
    const alertPill = document.createElement('span');
    alertPill.className = 'health-pill';
    alertPill.textContent = 'Alerts';
    container.appendChild(overall);
    container.appendChild(pricePill);
    container.appendChild(newsPill);
    container.appendChild(alertPill);
    // Place at top of body
    document.addEventListener('DOMContentLoaded', () => {
      try {
        document.body.insertBefore(container, document.body.firstChild);
      } catch (e) { /* no-op */ }
    });

    const refreshMs = Math.max(5000, options.refreshMs || 15000);
    const update = () => {
      ghostFetch('/health')
        .then((h) => {
          // Compute overall status
          const reasons = Array.isArray(h.degraded_reasons) ? h.degraded_reasons : [];
          const degraded = !!h.degraded;
          const priceDown = reasons.includes('price:unavailable');
          const priceWarn = reasons.includes('price:provider-unavailable') || reasons.includes('price:stale-prev-only');
          const newsMissing = reasons.includes('news:provider-missing');
          const newsRL = reasons.includes('news:rate-limited');
          const alertsOff = reasons.includes('alerts:telegram-disabled');

          // Overall
          overall.classList.remove('health-loading','health-ok','health-warn','health-bad');
          if (priceDown) {
            overall.classList.add('health-bad');
            overall.textContent = 'Health: down (price)';
          } else if (degraded || newsRL || priceWarn) {
            overall.classList.add('health-warn');
            overall.textContent = 'Health: degraded';
          } else {
            overall.classList.add('health-ok');
            overall.textContent = 'Health: ok';
          }

          // Price pill
          pricePill.classList.remove('hp-ok','hp-warn','hp-bad');
          if (priceDown) pricePill.classList.add('hp-bad');
          else if (priceWarn) pricePill.classList.add('hp-warn');
          else pricePill.classList.add('hp-ok');

          // News pill
          newsPill.classList.remove('hp-ok','hp-warn','hp-bad');
          if (newsMissing) newsPill.classList.add('hp-bad');
          else if (newsRL) newsPill.classList.add('hp-warn');
          else newsPill.classList.add('hp-ok');

          // Alerts pill
          alertPill.classList.remove('hp-ok','hp-warn','hp-bad');
          if (alertsOff) alertPill.classList.add('hp-warn'); // not red; optional sink
          else alertPill.classList.add('hp-ok');
        })
        .catch(() => {
          overall.classList.remove('health-loading','health-ok','health-warn');
          overall.classList.add('health-bad');
          overall.textContent = 'Health: unreachable';
          pricePill.classList.remove('hp-ok','hp-warn'); pricePill.classList.add('hp-bad');
          newsPill.classList.remove('hp-ok','hp-warn'); newsPill.classList.add('hp-bad');
          alertPill.classList.remove('hp-ok','hp-warn'); alertPill.classList.add('hp-bad');
        });
    };
    // Start
    update();
    setInterval(update, refreshMs);
  } catch (e) {
    // silent fail in UI-only context
  }
}

// Auto-init badge when this script is loaded
try {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => initHealthBadge());
  } else {
    initHealthBadge();
  }
} catch (e) { /* no-op */ }

// ============================================================================
// KEYBOARD SHORTCUTS - Power User Productivity
// ============================================================================
(function initKeyboardShortcuts() {
  const shortcuts = {
    // Navigation shortcuts
    '1': () => navigateTo('/'),
    '2': () => navigateTo('/engine'),
    '3': () => navigateTo('/bank'),
    '4': () => navigateTo('/security'),
    '5': () => navigateTo('/brain'),
    '6': () => navigateTo('/healthpage'),
    '7': () => navigateTo('/markets'),
    '8': () => navigateTo('/monthly'),
    
    // Action shortcuts
    'r': () => window.location.reload(),
    'h': () => navigateTo('/healthpage'),
    'c': () => navigateTo('/'),
    'm': () => navigateTo('/markets'),
    
    // Search (/)
    '/': (e) => {
      e.preventDefault();
      focusSearch();
    },
    
    // Command palette (Ctrl+K or Cmd+K)
    'k': (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        showCommandPalette();
      }
    },
    
    // Escape key - close modals/palettes
    'Escape': () => {
      closeCommandPalette();
    },
    
    // Help (?)
    '?': (e) => {
      if (e.shiftKey) {
        e.preventDefault();
        showKeyboardHelp();
      }
    }
  };
  
  function navigateTo(path) {
    window.location.href = path;
  }
  
  function focusSearch() {
    const searchInput = document.querySelector('input[type="search"]') || 
                       document.querySelector('input[placeholder*="search" i]');
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    } else {
      toast('No search field found', 'info');
    }
  }
  
  function showCommandPalette() {
    // Remove existing palette if any
    closeCommandPalette();
    
    // Create command palette overlay
    const overlay = document.createElement('div');
    overlay.id = 'command-palette-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding-top: 15vh;
      z-index: 9999;
    `;
    
    const palette = document.createElement('div');
    palette.id = 'command-palette';
    palette.style.cssText = `
      background: var(--card);
      border: 1px solid var(--border-accent);
      border-radius: 12px;
      width: 90%;
      max-width: 600px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    `;
    
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Type a command or navigate...';
    input.style.cssText = `
      width: 100%;
      padding: 16px 20px;
      border: none;
      background: transparent;
      color: var(--text);
      font-size: 1.1rem;
      outline: none;
      border-bottom: 1px solid var(--border);
    `;
    
    const commands = [
      { name: 'Cockpit', key: '/', action: () => navigateTo('/') },
      { name: 'Engine', key: '/engine', action: () => navigateTo('/engine') },
      { name: 'Bank', key: '/bank', action: () => navigateTo('/bank') },
      { name: 'Security', key: '/security', action: () => navigateTo('/security') },
      { name: 'Brain (AI)', key: '/brain', action: () => navigateTo('/brain') },
      { name: 'Health Status', key: '/healthpage', action: () => navigateTo('/healthpage') },
      { name: 'Markets', key: '/markets', action: () => navigateTo('/markets') },
      { name: 'Monthly Report', key: '/monthly', action: () => navigateTo('/monthly') },
      { name: 'Reload Page', key: 'r', action: () => window.location.reload() },
      { name: 'Clear Cache', key: 'clear cache', action: () => clearCaches() }
    ];
    
    const results = document.createElement('div');
    results.style.cssText = `
      max-height: 400px;
      overflow-y: auto;
      padding: 8px;
    `;
    
    function renderResults(filter = '') {
      results.innerHTML = '';
      const filtered = commands.filter(cmd => 
        cmd.name.toLowerCase().includes(filter.toLowerCase()) ||
        cmd.key.toLowerCase().includes(filter.toLowerCase())
      );
      
      filtered.forEach((cmd, idx) => {
        const item = document.createElement('div');
        item.style.cssText = `
          padding: 12px 16px;
          border-radius: 6px;
          cursor: pointer;
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin: 2px 0;
          background: ${idx === 0 ? 'var(--card-alt)' : 'transparent'};
        `;
        item.innerHTML = `
          <span style="font-weight: 500; color: var(--text)">${cmd.name}</span>
          <kbd style="padding: 4px 8px; background: var(--bg); border-radius: 4px; font-size: 0.75rem; color: var(--text-dim)">${cmd.key}</kbd>
        `;
        item.onmouseover = () => {
          results.querySelectorAll('div').forEach(d => d.style.background = 'transparent');
          item.style.background = 'var(--card-alt)';
        };
        item.onclick = () => {
          closeCommandPalette();
          cmd.action();
        };
        results.appendChild(item);
      });
      
      if (filtered.length === 0) {
        results.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-dim)">No commands found</div>';
      }
    }
    
    input.oninput = (e) => renderResults(e.target.value);
    input.onkeydown = (e) => {
      if (e.key === 'Escape') {
        closeCommandPalette();
      } else if (e.key === 'Enter') {
        const firstResult = results.querySelector('div');
        if (firstResult) firstResult.click();
      }
    };
    
    palette.appendChild(input);
    palette.appendChild(results);
    overlay.appendChild(palette);
    document.body.appendChild(overlay);
    
    renderResults();
    input.focus();
    
    overlay.onclick = (e) => {
      if (e.target === overlay) closeCommandPalette();
    };
  }
  
  function closeCommandPalette() {
    const overlay = document.getElementById('command-palette-overlay');
    if (overlay) overlay.remove();
  }
  
  function clearCaches() {
    closeCommandPalette();
    fetch('/api/cache/clear', { method: 'POST' })
      .then(() => toast('All caches cleared', 'success'))
      .catch(() => toast('Failed to clear caches', 'error'));
  }
  
  function showKeyboardHelp() {
    closeCommandPalette();
    
    const overlay = document.createElement('div');
    overlay.id = 'help-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
    `;
    
    const help = document.createElement('div');
    help.style.cssText = `
      background: var(--card);
      border: 1px solid var(--border-accent);
      border-radius: 12px;
      padding: 30px;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
      color: var(--text);
    `;
    
    help.innerHTML = `
      <h2 style="margin-top: 0; display: flex; justify-content: space-between; align-items: center;">
        <span>⌨️ Keyboard Shortcuts</span>
        <button onclick="this.closest('#help-overlay').remove()" style="background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 1.5rem">&times;</button>
      </h2>
      <div style="display: grid; grid-template-columns: auto 1fr; gap: 12px 20px; font-size: 0.95rem">
        <kbd>Ctrl+K</kbd><span>Open command palette</span>
        <kbd>/</kbd><span>Focus search</span>
        <kbd>r</kbd><span>Reload page</span>
        <kbd>?</kbd><span>Show this help</span>
        <kbd>Esc</kbd><span>Close palettes/modals</span>
        <kbd>1-8</kbd><span>Navigate to pages</span>
        <kbd>h</kbd><span>Go to health page</span>
        <kbd>c</kbd><span>Go to cockpit</span>
        <kbd>m</kbd><span>Go to markets</span>
      </div>
      <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 0.85rem">
        Press <kbd>?</kbd> anytime to see this help
      </div>
    `;
    
    overlay.appendChild(help);
    document.body.appendChild(overlay);
    
    overlay.onclick = (e) => {
      if (e.target === overlay) overlay.remove();
    };
  }
  
  // Global keyboard event handler
  document.addEventListener('keydown', (e) => {
    // Ignore if typing in input/textarea
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
      // Still allow Esc and Ctrl+K
      if (e.key !== 'Escape' && !(e.key === 'k' && (e.ctrlKey || e.metaKey))) {
        return;
      }
    }
    
    const key = e.key;
    const handler = shortcuts[key];
    
    if (handler) {
      handler(e);
    }
  });
  
  // Add kbd styling
  const style = document.createElement('style');
  style.textContent = `
    kbd {
      display: inline-block;
      padding: 4px 8px;
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 0.85rem;
      color: var(--text);
      box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
  `;
  document.head.appendChild(style);
})();

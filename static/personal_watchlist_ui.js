// Ghost Protocol V3 - Personal Watchlist UI Module
// Extends cockpit_v3.js with full CRUD operations for personal watchlist

// ============================================================================
// PERSONAL WATCHLIST STATE
// ============================================================================

let personalWatchlistState = {
    items: [],
    showAddForm: false,
    editingSymbol: null,
    activeTab: 'all'  // Default to showing all symbols
};

// ============================================================================
// ENHANCED WATCHLIST LOADING
// ============================================================================

/**
 * Load personal watchlist with enriched prediction data
 * Replaces basic loadWatchlist() from cockpit_v3.js
 */
async function loadPersonalWatchlist() {
    try {
        const response = await fetch('/api/v3/watchlist/user');
        if (!response.ok) {
            console.warn('[PERSONAL WATCHLIST] User endpoint failed, falling back to enriched');
            await loadWatchlistFallback();
            return;
        }
        
        const data = await response.json();
        const items = data.items || [];
        
        // Transform flat API response to expected structure
        personalWatchlistState.items = items.map(item => ({
            symbol: item.symbol,
            asset_type: item.type || 'stock',
            owns_position: item.owns_position || false,
            current_price: item.price || 0,
            change_24h: item.change_pct || 0,
            prediction: {
                direction: item.ghost_direction || 'FLAT',
                confidence: (item.ghost_confidence || 0) / 100,  // Convert 46 -> 0.46
                expected_move: item.change_pct || 0
            }
        }));
        
        // CRITICAL FIX: Populate sharedWatchlistData for Major Caps and XRP VIP
        // Convert to format expected by Major Caps (same as market watchlist)
        if (typeof sharedWatchlistData !== 'undefined') {
            sharedWatchlistData = items.map(item => ({
                symbol: item.symbol,
                price: item.price || 0,
                change_pct: item.change_pct || 0,
                ghost_confidence: item.ghost_confidence || 0,
                ghost_direction: item.ghost_direction || 'FLAT',
                type: item.type || 'stock'
            }));
            console.log('[PERSONAL WATCHLIST] Populated sharedWatchlistData for Major Caps:', sharedWatchlistData.length, 'items');
        }
        
        renderPersonalWatchlist(getFilteredWatchlistItems());
        
        console.log(`[PERSONAL WATCHLIST] Loaded ${personalWatchlistState.items.length} symbols`);
    } catch (error) {
        console.error('[PERSONAL WATCHLIST] Load error:', error);
        await loadWatchlistFallback();
    }
}

/**
 * Fallback to existing enriched endpoint if personal watchlist not available
 */
async function loadWatchlistFallback() {
    try {
        const response = await fetch('/api/v3/watchlist/enriched');
        if (!response.ok) throw new Error('Fallback failed');
        
        const data = await response.json();
        const items = data.items || [];
        
        // Convert to personal watchlist format
        personalWatchlistState.items = items.map(item => ({
            symbol: item.symbol,
            asset_type: item.type || 'stock',
            owns_position: false,
            current_price: item.price,
            prediction: {
                direction: item.direction || 'FLAT',
                confidence: item.ghost_score ? item.ghost_score / 100 : 0,
                expected_move: item.change || 0
            }
        }));
        
        // CRITICAL FIX: Populate sharedWatchlistData for Major Caps (fallback path)
        if (typeof sharedWatchlistData !== 'undefined') {
            sharedWatchlistData = items.map(item => ({
                symbol: item.symbol,
                price: item.price || 0,
                change_pct: item.change_pct || 0,
                ghost_confidence: item.ghost_confidence || 0,
                ghost_direction: item.ghost_direction || 'FLAT',
                type: item.type || 'stock'
            }));
            console.log('[PERSONAL WATCHLIST] Fallback: Populated sharedWatchlistData');
        }
        
        renderPersonalWatchlist(getFilteredWatchlistItems());
    } catch (error) {
        console.error('[PERSONAL WATCHLIST] Fallback error:', error);
        renderPersonalWatchlist([]);
    }
}

/**
 * Get filtered watchlist items based on active tab
 */
function getFilteredWatchlistItems() {
    const tab = personalWatchlistState.activeTab;
    
    if (tab === 'all') {
        return personalWatchlistState.items;
    }
    
    // Filter by asset_type: 'stocks' tab => 'stock', 'crypto' tab => 'crypto'
    const assetType = tab === 'stocks' ? 'stock' : 'crypto';
    return personalWatchlistState.items.filter(item => item.asset_type === assetType);
}

/**
 * Update active tab (called by cockpit_v3.js tab handler)
 * This function is exposed globally so tab clicks can filter the watchlist
 */
function updateWatchlistTab(tabName) {
    personalWatchlistState.activeTab = tabName;
    renderPersonalWatchlist(getFilteredWatchlistItems());
}

// ============================================================================
// WATCHLIST RENDERING
// ============================================================================

/**
 * Render personal watchlist with add/remove controls
 */
function renderPersonalWatchlist(items) {
    const container = document.getElementById('watchlist-table');
    
    if (!items || items.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 30px; color: var(--text-secondary);">
                <p style="font-size: 16px; margin-bottom: 20px;">📋 Your watchlist is empty</p>
                <button onclick="showAddSymbolForm()" class="btn-primary" style="padding: 10px 20px;">
                    ➕ Add Symbol
                </button>
            </div>
        `;
        return;
    }
    
    // Add symbol button at top
    const addButtonHtml = `
        <div style="margin-bottom: 15px; text-align: right;">
            <button onclick="showAddSymbolForm()" class="btn-secondary" style="padding: 6px 12px; font-size: 14px;">
                ➕ Add Symbol
            </button>
        </div>
    `;
    
    // Render items
    const itemsHtml = items.slice(0, 20).map(item => renderWatchlistItem(item)).join('');
    
    container.innerHTML = addButtonHtml + '<div class="watchlist-items">' + itemsHtml + '</div>';
}

/**
 * Render single watchlist item row
 */
function renderWatchlistItem(item) {
    const prediction = item.prediction || {};
    const direction = prediction.direction || 'FLAT';
    const confidence = prediction.confidence || 0;
    const expectedMove = prediction.expected_move || 0;
    const price = item.current_price || 0;
    const change24h = item.change_24h || 0;
    
    // Direction emoji and color
    const dirEmoji = direction === 'UP' ? '🟢↑' : direction === 'DOWN' ? '🔴↓' : '⚪→';
    const dirClass = direction === 'UP' ? 'positive' : direction === 'DOWN' ? 'negative' : 'neutral';
    
    // 24h change display with color
    const change24hDisplay = change24h !== 0 ? 
        `<span class="${change24h >= 0 ? 'positive' : 'negative'}">${change24h >= 0 ? '+' : ''}${change24h.toFixed(2)}%</span>` : 
        '<span>--</span>';
    
    // Ownership badge
    const ownershipBadge = item.owns_position ? 
        '<span style="background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;">OWN</span>' : 
        '';
    
    // Asset type badge
    const assetTypeBadge = item.asset_type === 'crypto' ? 
        '<span style="background: #f39c12; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;">CRYPTO</span>' : 
        '<span style="background: #3498db; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;">STOCK</span>';
    
    return `
        <div class="watchlist-row" style="gap: 15px;">
            <!-- Symbol Info -->
            <div style="flex: 1; min-width: 120px;">
                <div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">
                    ${item.symbol}
                    ${ownershipBadge}
                </div>
                <div style="font-size: 12px; color: var(--text-secondary);">
                    ${assetTypeBadge}
                    ${price > 0 ? `$${price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '--'}
                    <span style="margin-left: 8px;">24h: ${change24hDisplay}</span>
                </div>
            </div>
            
            <!-- Prediction Info -->
            <div style="flex: 1; min-width: 150px;">
                <div style="font-size: 14px; margin-bottom: 4px;">
                    <span style="margin-right: 8px;">${dirEmoji}</span>
                    <span class="${dirClass}">${direction}</span>
                </div>
                <div style="font-size: 12px; color: var(--text-secondary);">
                    ${confidence > 0 ? `${(confidence * 100).toFixed(0)}% conf` : '--'}
                    ${expectedMove !== 0 ? ` • ${expectedMove > 0 ? '+' : ''}${expectedMove.toFixed(1)}%` : ''}
                </div>
            </div>
            
            <!-- Actions -->
            <div style="display: flex; gap: 8px; align-items: center;">
                <button 
                    onclick="toggleOwnership('${item.symbol}', '${item.asset_type}', ${!item.owns_position})"
                    class="btn-icon"
                    title="${item.owns_position ? 'Mark as NOT owned' : 'Mark as owned'}"
                    style="padding: 6px 10px; font-size: 12px;">
                    ${item.owns_position ? '✅' : '➕'}
                </button>
                <button 
                    onclick="viewSymbolHistory('${item.symbol}')"
                    class="btn-icon"
                    title="View prediction history"
                    style="padding: 6px 10px; font-size: 12px;">
                    📊
                </button>
                <button 
                    onclick="removeSymbolFromWatchlist('${item.symbol}', '${item.asset_type}')"
                    class="btn-icon btn-danger"
                    title="Remove from watchlist"
                    style="padding: 6px 10px; font-size: 12px; color: #dc3545;">
                    ✖
                </button>
            </div>
        </div>
    `;
}

// ============================================================================
// ADD SYMBOL FORM
// ============================================================================

/**
 * Show add symbol form modal
 */
function showAddSymbolForm() {
    const modalHtml = `
        <div id="add-symbol-modal" style="
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 10000;
            display: flex; align-items: center; justify-content: center;">
            
            <div style="
                background: var(--bg-panel); padding: 30px; border-radius: 12px; border: 1px solid var(--border-subtle);
                max-width: 500px; width: 90%; box-shadow: 0 10px 40px rgba(0,0,0,0.5);">
                
                <h2 style="margin: 0 0 20px 0; font-size: 20px;">➕ Add Symbol to Watchlist</h2>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 6px; font-weight: 600;">Symbol</label>
                    <input 
                        type="text" 
                        id="add-symbol-input" 
                        placeholder="e.g., AAPL, BTC"
                        style="width: 100%; padding: 10px; font-size: 14px; background: var(--bg-dark); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: 6px;"
                        maxlength="20">
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 6px; font-weight: 600;">Asset Type</label>
                    <select 
                        id="add-asset-type" 
                        style="width: 100%; padding: 10px; font-size: 14px; background: var(--bg-dark); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: 6px; cursor: pointer;">
                        <option value="stock">Stock</option>
                        <option value="crypto">Crypto</option>
                    </select>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: flex; align-items: center; cursor: pointer;">
                        <input type="checkbox" id="add-owns-position" style="margin-right: 8px;">
                        <span>I currently own this asset</span>
                    </label>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 6px; font-weight: 600;">Alert Threshold (%)</label>
                    <input 
                        type="number" 
                        id="add-alert-threshold" 
                        value="5.0"
                        min="0.1"
                        max="50"
                        step="0.5"
                        style="width: 100%; padding: 10px; font-size: 14px; background: var(--bg-dark); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: 6px;">
                    <small style="color: var(--text-secondary);">Alert when price moves ±this %</small>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 6px; font-weight: 600;">Notes (optional)</label>
                    <textarea 
                        id="add-notes" 
                        placeholder="e.g., Watching for breakout above $150"
                        style="width: 100%; padding: 10px; font-size: 14px; background: var(--bg-dark); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: 6px; resize: vertical;"
                        maxlength="500"
                        rows="3"></textarea>
                </div>
                
                <div style="display: flex; gap: 10px; justify-content: flex-end;">
                    <button onclick="closeAddSymbolForm()" class="btn btn-secondary" style="padding: 10px 20px;">
                        Cancel
                    </button>
                    <button onclick="submitAddSymbol()" class="btn btn-primary" style="padding: 10px 20px;">
                        ➕ Add Symbol
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    document.getElementById('add-symbol-input').focus();
}

/**
 * Close add symbol form modal
 */
function closeAddSymbolForm() {
    const modal = document.getElementById('add-symbol-modal');
    if (modal) {
        modal.remove();
    }
}

/**
 * Submit add symbol form
 */
async function submitAddSymbol() {
    const symbol = document.getElementById('add-symbol-input').value.trim().toUpperCase();
    const assetType = document.getElementById('add-asset-type').value;
    const ownsPosition = document.getElementById('add-owns-position').checked;
    const alertThreshold = parseFloat(document.getElementById('add-alert-threshold').value);
    const notes = document.getElementById('add-notes').value.trim();
    
    if (!symbol) {
        alert('Please enter a symbol');
        return;
    }
    
    if (symbol.length > 20) {
        alert('Symbol must be 20 characters or less');
        return;
    }
    
    try {
        const response = await fetch('/api/v3/watchlist/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                asset_type: assetType,
                owns_position: ownsPosition,
                notes,
                alert_threshold_pct: alertThreshold,
                priority: 1
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to add symbol');
        }
        
        const result = await response.json();
        console.log('[PERSONAL WATCHLIST] Symbol added:', result);
        
        // Close modal and reload watchlist
        closeAddSymbolForm();
        await loadPersonalWatchlist();
        
        showNotification(`✅ ${symbol} added to watchlist`, 'success');
    } catch (error) {
        console.error('[PERSONAL WATCHLIST] Add symbol error:', error);
        alert(`Failed to add symbol: ${error.message}`);
    }
}

// ============================================================================
// WATCHLIST ACTIONS
// ============================================================================

/**
 * Remove symbol from watchlist
 */
async function removeSymbolFromWatchlist(symbol, assetType) {
    if (!confirm(`Remove ${symbol} from watchlist?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/v3/watchlist/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                asset_type: assetType
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to remove symbol');
        }
        
        console.log(`[PERSONAL WATCHLIST] Symbol removed: ${symbol}`);
        
        // Reload watchlist
        await loadPersonalWatchlist();
        
        showNotification(`🗑️ ${symbol} removed from watchlist`, 'info');
    } catch (error) {
        console.error('[PERSONAL WATCHLIST] Remove symbol error:', error);
        alert(`Failed to remove symbol: ${error.message}`);
    }
}

/**
 * Toggle ownership flag for symbol
 */
async function toggleOwnership(symbol, assetType, ownsPosition) {
    try {
        const response = await fetch('/api/v3/watchlist/update-position', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                asset_type: assetType,
                owns_position: ownsPosition
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to update ownership');
        }
        
        console.log(`[PERSONAL WATCHLIST] Ownership updated: ${symbol} = ${ownsPosition}`);
        
        // Reload watchlist to reflect changes
        await loadPersonalWatchlist();
        
        const message = ownsPosition ? 
            `✅ Marked ${symbol} as owned` : 
            `⚠️ Marked ${symbol} as not owned`;
        showNotification(message, 'success');
    } catch (error) {
        console.error('[PERSONAL WATCHLIST] Toggle ownership error:', error);
        alert(`Failed to update ownership: ${error.message}`);
    }
}

/**
 * View prediction history for symbol
 */
async function viewSymbolHistory(symbol) {
    try {
        const response = await fetch(`/api/v3/watchlist/history/${symbol}?limit=20`);
        if (!response.ok) throw new Error('Failed to load history');
        
        const data = await response.json();
        const history = data.history || [];
        
        // Show history modal
        showHistoryModal(symbol, history);
    } catch (error) {
        console.error('[PERSONAL WATCHLIST] View history error:', error);
        alert(`Failed to load history: ${error.message}`);
    }
}

/**
 * Show prediction history modal
 */
function showHistoryModal(symbol, history) {
    const historyHtml = history.length > 0 ? 
        history.map(item => `
            <div style="padding: 12px; border-bottom: 1px solid var(--border-subtle); background: var(--bg-dark); border-radius: 6px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="font-weight: 600;">${item.direction}</span>
                    <span style="font-size: 12px; color: var(--text-secondary);">
                        ${new Date(item.generated_at).toLocaleString()}
                    </span>
                </div>
                <div style="font-size: 13px; color: var(--text-secondary);">
                    Confidence: ${(item.confidence * 100).toFixed(0)}% • 
                    Expected: ${item.expected_move_pct > 0 ? '+' : ''}${item.expected_move_pct.toFixed(1)}% • 
                    Reason: ${item.reason}
                    ${item.alert_sent ? ' • 📨 Alert sent' : ''}
                </div>
            </div>
        `).join('') : 
        '<p style="text-align: center; padding: 20px; color: var(--text-secondary);">No prediction history yet</p>';
    
    const modalHtml = `
        <div id="history-modal" style="
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 10000;
            display: flex; align-items: center; justify-content: center;">
            
            <div style="
                background: var(--bg-panel); padding: 30px; border-radius: 12px; border: 1px solid var(--border-subtle);
                max-width: 700px; width: 90%; max-height: 80vh; overflow-y: auto;
                box-shadow: 0 10px 40px rgba(0,0,0,0.5);">
                
                <h2 style="margin: 0 0 20px 0; font-size: 20px;">📊 Prediction History: ${symbol}</h2>
                
                <div style="margin-bottom: 20px;">
                    ${historyHtml}
                </div>
                
                <div style="text-align: right;">
                    <button onclick="closeHistoryModal()" class="btn-secondary" style="padding: 10px 20px;">
                        Close
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

/**
 * Close history modal
 */
function closeHistoryModal() {
    const modal = document.getElementById('history-modal');
    if (modal) {
        modal.remove();
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Show toast notification
 */
function showNotification(message, type = 'info') {
    const bgColor = type === 'success' ? '#28a745' : 
                    type === 'error' ? '#dc3545' : 
                    type === 'warning' ? '#ffc107' : '#17a2b8';
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 10001;
        background: ${bgColor}; color: white; padding: 15px 20px;
        border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        font-size: 14px; max-width: 400px; animation: slideInRight 0.3s;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================================================
// INITIALIZATION HOOK
// ============================================================================

/**
 * Personal watchlist module initialization
 * Works alongside cockpit_v3.js dual-mode watchlist system
 */
function initPersonalWatchlist() {
    // DO NOT override loadWatchlist - let cockpit_v3.js handle mode switching
    // This module provides loadPersonalWatchlist() which is called by cockpit_v3.js
    
    console.log('[PERSONAL WATCHLIST] UI module initialized and ready');
}

// Auto-initialize if DOM already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait for cockpit_v3.js to initialize first
        setTimeout(initPersonalWatchlist, 500);
    });
} else {
    setTimeout(initPersonalWatchlist, 500);
}


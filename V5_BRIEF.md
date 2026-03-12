# GHOST PROTOCOL v5 — COMPLETE AGENT BRIEF

## WHAT THIS IS

Ghost Protocol is a stock/crypto prediction system. It runs background tasks that analyze markets, generate trade picks, send them to Telegram, track paper trades, and score outcomes. The current cockpit (v4) has the right tab structure but the data pipes are not connected — most tabs are empty. This brief tells you what to build, what it should look like, and where the data comes from.

## DESIGN REFERENCES

**Robinhood (primary layout):** Market ticker strip across the top showing S&P 500, DOW, NASDAQ, EUR/USD, Crude Oil, VIX — all in one horizontal bar, always visible on every tab. Left sidebar for navigation icons. Main center panel for content. Right sidebar for positions and watchlist. Everything visible on one screen, no scrolling needed. Dark theme. Green/red color coding for up/down.

**Yahoo Finance (for tables):** Clean data table with columns: Symbol, Name, mini sparkline chart, Price, Change, Change %, Volume. Column headers are sortable. Filterable tabs above the table (Most Active, Top Gainers, Top Losers). Right sidebar shows Trending tickers, Top gainers, Top losers as compact lists.

**Houston Play Studios dashboard (for philosophy):** Each page mirrors one data source. The display matches reality. If the Google Calendar says 4 bookings, the dashboard says 4 bookings. One number per metric, shown once, verifiable. Tab-based navigation: Home, Bookings, Customers, Financials, Analytics.

## THE CORE RULE

Every tab mirrors one backend system. The display is a window into the machine. If a backend task is running, its tab shows live data. If a task says "never" on the Health tab heartbeat, the corresponding tab should honestly say "Not running yet" instead of showing fake or empty data. Sean needs to look at the cockpit and immediately see what is working and what is not. No guessing, no fixing loops.

---

## LAYOUT STRUCTURE

### MARKET TICKER BAR (permanent, top of every tab)

A horizontal strip that stays visible on all tabs, like Robinhood top bar. Shows major indices and assets with price, change amount, change percent, and mini sparkline. Green for up, red for down.

Example content: S&P 500 6703.08 -72.72 (-1.07%) | DOW 46881 -536.05 (-1.13%) | NASDAQ 22422 -294.13 (-1.29%) | BTC 70447 -828.05 (-1.16%) | VIX 25.80 +1.57

Ghost already has price feed data. The Telegram health check mentions "Price Feeds — 1/2 feeds responding." Pipe that data into this bar.

### LEFT SIDEBAR (icon navigation, like Robinhood)

A slim vertical sidebar with icons instead of horizontal tabs. This frees the full top bar for the market ticker and keeps nav accessible on every tab. Each icon highlights when active. Hover shows the label.

Icons in order:
- Ghost logo at top
- - Picks (chart icon)
  - - Stocks (trending up icon)
    - - Crypto (coin icon)
      - - History (clock icon)
        - - Health (heart icon)
          - - AI Brain (brain icon)
            - - News (newspaper icon)
              - - Financials (dollar icon)
               
                - ### MAIN CONTENT AREA (center, changes per tab)
               
                - Full width minus the two sidebars. This is where each tab content lives.
               
                - ### RIGHT SIDEBAR (context panel, like Robinhood Positions panel)
               
                - A slim right sidebar showing context relevant to the current tab. Active positions on the Picks tab. Top movers on the Stocks/Crypto tabs. Quick stats on Health tab.
                - 
                ---

                ## TAB-BY-TAB SPECIFICATION

                ### TAB 1: PICKS
                **Mirrors backend:** prediction-cycle task + notification-loop task

                This tab shows exactly what the Telegram sends. The prediction cycle generates picks, the notification loop sends them to Telegram. This tab displays the same output.

                **Main content:** Today's picks in Telegram card format. Each pick is a card showing: direction (UP green or DOWN red), symbol, entry price, target price with profit percent, stop loss price, deadline date, and the "$100 in yields $X back" line. Show status: PENDING, WON, or LOST.

                Today's Telegram sent 5 picks: T (DOWN at $27.16, target $26.35, stop $27.70, by Mar 16), NET (DOWN at $213.00, target $206.61, stop $217.26, by Mar 19), XPO (UP at $193.90, target $199.72, stop $190.02, by Mar 16), PANW (DOWN at $164.93, target $159.98, stop $168.23, by Mar 19), DDOG (UP at $127.49, target $131.31, stop $124.94, by Mar 16).

                Below the cards show a "Recent Picks" table with the last 7 days of picks in rows.

                **Right sidebar:** Active paper trades with live P&L. Symbol, direction, entry, current price, P&L in green/red. Like Robinhood Positions panel.

                **Header line:** "GHOST PICKS — Thursday, March 12, 2026 | 5 picks today | 53.3% accuracy | 249/467 correct"

                **Data source:** Whatever function builds the Telegram GHOST PICKS message, call the same function. If 5 picks were sent this morning, 5 cards appear here. The current v4 shows 0 picks while Telegram sent 5. This is the number one fix — connect to the actual pick generation endpoint.

                ### TAB 2: STOCKS
                **Mirrors backend:** stock watchlist + stock price feeds

                **Main content:** Yahoo Finance style table with columns: Symbol, Name, Price, Change, Change %, Ghost Direction (UP/DOWN arrow, color coded), Ghost Confidence %, mini sparkline chart. Show all stocks Ghost watches: NET, PANW, XPO, DDOG, T, FTNT, BMBL. Columns should be sortable.

                Filter buttons above: All | Active Picks | Watching

                **Right sidebar:** Top Stock Movers — biggest percent movers from the watchlist, compact list format.

                **Data source:** The old v3 cockpit displayed all these stocks with live prices, directions, and Ghost confidence scores. Those endpoints still exist at the API. The current v4 Stocks tab shows "No items" because it is not connected. Connect it to the existing watchlist/predictions endpoints.

                ### TAB 3: CRYPTO
                **Mirrors backend:** crypto watchlist + crypto price feeds

                Same layout as Stocks but only crypto assets: CHZ, LINK, XRP, ETH. Same table format, same columns, same filter buttons.

                **Right sidebar:** Top Crypto Movers.

                **Data source:** Same endpoints as Stocks, filtered for crypto. The old v3 showed these with live prices and confidence. Connect the endpoints.

                ### TAB 4: HISTORY
                **Mirrors backend:** outcome-reconciler task

                The outcome reconciler resolves trades and determines win/loss. This tab shows its full output.

                **Summary bar at top:** Total trades | Wins (green) | Losses (red) | Win Rate % | Total P&L ($)

                **Filter buttons:** All | Stocks | Crypto | Wins | Losses (v4 already has these and they work, keep them)

                **Table columns:** Symbol | Direction | Entry Price | Exit Price | P&L ($) | Result (WIN green / LOSS red) | Date

                **Critical data fix:** The current v4 History tab shows only 9 trades (all CHZ and DDOG, all with dashes for Exit, $0.00 for P&L, dashes for Date). But the Health tab says 249W / 218L = 467 total resolved. Pull ALL 467 resolved trades with their actual exit prices, P&L amounts, and resolution dates. The Telegram sends messages like "XPO — You lost. Put in $100 got back $97.60" — those exit prices and P&L values exist somewhere in the backend. Pull them into this table.

                ### TAB 5: HEALTH
                **Mirrors backend:** all background tasks (system overview)

                v4 already built this well. Keep the existing layout with these sections:

                **Top line:** "53.3% accuracy | 249/467 correct | Status: DECLINING | System: 77/100 | 7 issues"

                **Accuracy Breakdown:** Four cards showing 24 Hour accuracy, 7 Day accuracy, 30 Day accuracy, and Overall Record (W/L).

                **Heartbeat grid:** Grid of cards showing every background task name and its last pulse time. Green dot for alive, gray dot for never pulsed. Currently 5 tasks are alive (prediction cycle, news analysis, notification loop, doctor cron, outcome reconciler) and 11 have never pulsed (accuracy tracker, alert worker, autosave worker, full scanner, guardian oracle, money game, online calibrator, open close scheduler, premarket scanner, self improvement, vip scanner).

                **Telegram Health Check mirror:** Add a section that displays the exact same format as the Telegram health check so Sean can compare them side by side. Format: checkmark or warning icon, subsystem name, status detail. Example: "API Server — HTTP 200" and "Predictions — 18 preds, newest 37m ago" and "Price Feeds — 1/2 feeds responding" and "Accuracy Tracker — 249/467 correct (53.3%), 394 pending, 505 skipped".

                **Issues section:** List of WARN, INFO, ERROR items with severity color coding. Keep as-is from v4.

                ### TAB 6: AI BRAIN (NEW)
                **Mirrors backend:** self-improvement task + intelligence hub subsystems

                Ghost has an intelligence hub with 10+ subsystems and a self-improvement task. This tab shows what the AI is thinking and learning.

                **Sections to build:**
                - Edge Symbols: The 11 edge symbols mentioned in the Telegram health check. Show what they are and why Ghost has edge on them.
                - - Confidence Map: All watched symbols ranked by Ghost confidence level, highest to lowest. Shows where Ghost has strong opinions versus no signal.
                  - - Skip Analysis: 505 of 972 predictions are skip-tagged. Show which symbols get skipped most and why. What pattern causes skips?
                    - - Low Accuracy Breakdown: DDOG (23%), XPO (29%), CHZ (33%) — show what Ghost predicted versus what actually happened for these symbols.
                      - - Self-Improvement Log: Timeline of what adjustments the self-improvement task has made.
                        - - Intelligence Hub Subsystems: List all subsystems and their current state (loaded/not loaded).
                         
                          - **If data not available:** If these endpoints don't exist yet, build them. The intelligence hub loads 10/9 subsystems per the Telegram health check, so the data is there.
                         
                          - ### TAB 7: NEWS
                          - **Mirrors backend:** news-analysis task
                         
                          - v4 already has a working news feed with BULLISH/BEARISH/NEUTRAL sentiment tags, timestamps, and clickable links. Give it its own full-width tab with better layout.
                         
                          - **Layout:** Full-width article rows. Each row shows: headline text, source name or icon, sentiment tag (BULLISH green, BEARISH red, NEUTRAL gray), timestamp (Xm ago), and relevant symbol tags (which stocks or crypto the article affects).
                         
                          - **Filter buttons:** All | Stocks-related | Crypto-related | Macro/Market
                         
                          - **Data source:** Already connected in v4. Just needs more space, symbol relevance tags, and filtering.
                         
                          - ### TAB 8: FINANCIALS (NEW)
                          - **Mirrors backend:** money-game task + paper trading P&L data
                         
                          - **Sections to build:**
                          - - Paper Trading P&L Chart: Line chart showing cumulative P&L over time, like Robinhood portfolio value chart with 1D/1W/1M/1Y/ALL time toggles.
                            - - Performance by Symbol: Table showing each symbol win rate, total P&L, number of trades, average win size, average loss size.
                              - - Best and Worst: Best performing symbol, worst performing symbol, biggest single win, biggest single loss.
                                - - Risk Metrics: Overall win rate, profit factor, average risk/reward ratio.
                                 
                                  - **If money-game task has never pulsed:** Show "Financial analysis not running yet — money-game task has not started" instead of empty or fake data. Be honest about what is and is not working.
                                  - 
                                  ---

                                  ## CRITICAL DATA PIPE FIXES

                                  These are the connections that need to be made. The v4 front end is built but empty on most tabs:

                                  1. **Picks tab must connect to the Telegram pick generator.** Whatever function or endpoint creates the GHOST PICKS Telegram message, the Picks tab must call the same function. The Telegram sent 5 picks today (T, NET, XPO, PANW, DDOG) but the cockpit shows 0. This is the number one priority fix.
                                 
                                  2. 2. **Stocks and Crypto watchlists must connect to existing prediction and watchlist data.** The old v3 cockpit displayed NET, PANW, XPO, DDOG, T, FTNT, BMBL (stocks) and CHZ, LINK, XRP, ETH (crypto) with live prices and Ghost confidence scores. Those API endpoints still exist. Connect them to the new Stocks and Crypto tab tables.
                                    
                                     3. 3. **History table must pull the full resolved trade log.** Health tab says 249W / 218L = 467 total resolved. History tab shows only 9. Pull all 467 trades with actual exit prices, P&L dollar amounts, and resolution dates. Not just the most recent page — all of them.
                                       
                                        4. 4. **Market ticker bar must connect to price feed data.** Ghost has price feeds (the Telegram health check says "Price Feeds — 1/2 feeds responding"). Pipe S&P 500, DOW, NASDAQ, BTC into the top ticker bar.
                                          
                                           5. 5. **One accuracy number everywhere.** Use 53.3% (249/467) consistently on the Picks header, the Health tab, and the Financials tab. Same number, same source, zero contradictions.
                                             
                                              6. ---
                                             
                                              7. ## WHAT THE TELEGRAM SENDS (source of truth the display must match)
                                             
                                              8. ### Daily Picks format:
                                              9. GHOST PICKS — Thu March 12, 2026
                                              10. 1) RED T is going DOWN (star) — Get in at $27.16, Get out at $26.35 (you make 3.0%), Run away at $27.70 (you lose get out), Done by Mon Mar 16, Put $100 in yields Get $103.00 back
                                                  2) 2) RED NET is going DOWN (star) — Get in at $213.00, Get out at $206.61 (you make 3.0%), Run away at $217.26, Done by Thu Mar 19, $100 in yields $103.00 back
                                                     3) 3) GREEN XPO is going UP (star) — Get in at $193.90, Get out at $199.72 (you make 3.0%), Run away at $190.02, Done by Mon Mar 16, $100 in yields $103.00 back
                                                        4) 4) RED PANW is going DOWN (star) — same format
                                                           5) 5) GREEN DDOG is going UP (star) — same format
                                                             
                                                              6) GREEN = going UP (buy now sell later). RED = going DOWN (sell now buy back later). Star = this one wins a lot.
                                                             
                                                              7) ### Trade Resolution format:
                                                              8) Ghost is watching — X symbol — You lost. Put in $100, Got back $97.60. It happens. Move on.
                                                              9) Ghost is watching — X symbol — You won. Put in $100, Got back $103.00.
                                                             
                                                              10) ### Health Check format:
                                                              11) Ghost Health Check
                                                              12) API Server — HTTP 200
                                                              13) Predictions — 10 preds, newest 1.4h ago
                                                              14) Edge Symbols — 11 edge symbols
                                                              15) Price Feeds — 1/2 feeds responding
                                                              16) Intelligence Hub — 10/9 subsystems loaded
                                                              17) Core Imports — 7/7 modules OK
                                                              18) Accuracy Tracker — 241/439 correct (55%), 305 pending, 494 skipped
                                                              19) Telegram Config — configured
                                                              20) RESULT: PASS (8/8 passed)
                                                             
                                                              21) Every one of these should have a visible, verifiable counterpart on the cockpit. Picks appear on the Picks tab. Resolutions appear on the History tab. Health check appears on the Health tab. If they do not match, the display is wrong.
                                                             
                                                              22) ---
                                                             
                                                              23) ## CURRENT v4 STATE (what you are starting from)
                                                             
                                                              24) The v4 agent built 5 tabs: Picks, Stocks, Crypto, History, Health. The structure is correct. Here is what works and what does not:
                                                             
                                                              25) - **Picks:** Structure good. DATA EMPTY. Shows 0 picks even though Telegram sent 5 today. Not connected to pick generation endpoint.
                                                                  - - **Stocks:** Structure good. DATA EMPTY. Watchlist shows "No items", active trades empty. Not connected to watchlist/prediction endpoints.
                                                                    - - **Crypto:** Structure good. DATA EMPTY. Same problem as Stocks tab.
                                                                      - - **History:** Structure good. DATA PARTIAL. Shows only 9 trades instead of 467. All exits show dashes, all P&L shows $0.00, all dates show dashes. Filter buttons work (All, Stocks, Crypto, Wins, Losses).
                                                                        - - **Health:** WORKS WELL. Accuracy breakdown accurate (0% 24h, 53.3% 7d, 53.3% 30d, 249W/218L record). Heartbeat grid shows all tasks with correct pulse times. Issues list shows 7 real issues with severity tags.
                                                                          - - **News:** Works on Stocks and Crypto tabs. Sentiment tags (BULLISH/BEARISH/NEUTRAL), timestamps, clickable links all functional.
                                                                           
                                                                            - You are adding 3 new tabs (AI Brain, News standalone, Financials), switching from horizontal tabs to a left sidebar with icons, adding the market ticker bar across the top, and adding right sidebars. Most importantly you are connecting the data pipes so the empty tabs show real data.
                                                                           
                                                                            - ---

                                                                            ## PANELS DELETED FROM v3 (do not bring back)

                                                                            These were removed from the old v3 cockpit for good reasons:
                                                                            - Latest CHZ Prediction panel (was duplicated in 3 other places)
                                                                            - - Ghost Forecast panel (duplicate of Latest Prediction, had inverted range bug)
                                                                              - - Trade Journal panel (empty, nobody used it)
                                                                                - - VIP Watch / VIP Sniper Coins panel (zero data, 5 presale meme coins with no prices or analysis)
                                                                                  - - Ghost Performance Score panel (fabricated numbers, claimed 100/100 A system health while errors were active, claimed 72% accuracy while real accuracy was 53%)
                                                                                    - - Top Movers panel (redundant with Watchlist, merged into tab views)
                                                                                      - - Prediction Accuracy chart panel (said "IMPROVING" when 24h accuracy was 0%, misleading)
                                                                                       
                                                                                        - Do not recreate any of these. Their useful data is now in the proper tabs (picks in Picks tab, accuracy in Health tab, watchlist in Stocks/Crypto tabs, history in History tab).
                                                                                       
                                                                                        - ---

                                                                                        ## SUMMARY

                                                                                        Build a Robinhood-style layout with left sidebar nav, top market ticker bar, center content, right context sidebar. Eight tabs that each mirror one backend system. Connect the data pipes so picks, watchlists, history, and health all show real data matching what the Telegram sends. Be honest — if something is not running, say so. The display is a mirror of the backend, separated by tabs.

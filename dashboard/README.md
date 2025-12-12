# Ghost Protocol Dashboard

Real-time monitoring dashboard for Ghost Protocol autonomous trading system.

## Features

- **Live WebSocket Updates**: Real-time trade execution monitoring
- **Performance Metrics**: P&L tracking, win rate, drawdown analysis
- **Trade History**: Scrollable list of recent trades with details
- **Phase 5 Status**: Autonomous execution engine monitoring
- **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open browser
open http://localhost:3000
```

## Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

Create `.env.local`:

```env
BACKEND_URL=https://ghost-protocol-production.up.railway.app
```

## Technology Stack

- **Framework**: Next.js 14 + React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Real-time**: WebSocket (native)

## Dashboard Sections

### 1. Stats Grid
- Execution Cycles (total Phase 5 runs)
- Trades Today (24h trade count)
- Total P&L (cumulative profit/loss)
- Win Rate (% of profitable trades)

### 2. P&L Chart
- Real-time line chart showing P&L over time
- Updates automatically with new trades
- Green for profit, red for loss

### 3. Recent Trades
- Last 10 trades displayed
- Symbol, side (BUY/SELL), quantity, price
- Color-coded by direction
- Timestamps for each trade

### 4. Performance Metrics
- Total trades executed
- Winning trades count
- Losing trades count

## WebSocket Integration

Dashboard connects to `/ws/trades` endpoint:

```typescript
const ws = new WebSocket('wss://ghost-protocol-production.up.railway.app/ws/trades')

ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'trade_update') {
    // Update UI with new trade data
  }
}
```

## API Endpoints Used

- `GET /api/v3/phase5/status` - Phase 5 execution status
- `GET /api/v3/trade/dashboard` - Dashboard summary data
- `WS /ws/trades` - Real-time trade updates

## Troubleshooting

**Dashboard not connecting?**
- Check backend is running: `curl https://ghost-protocol-production.up.railway.app/api/health`
- Verify WebSocket endpoint: `wscat -c wss://ghost-protocol-production.up.railway.app/ws/trades`

**No trades showing?**
- Phase 5 may be waiting for 60%+ confidence predictions
- Check execution status: `curl https://ghost-protocol-production.up.railway.app/api/v3/phase5/status`
- Test trade injection: `curl -X POST https://ghost-protocol-production.up.railway.app/api/v3/test/inject-trade`

**Styles not loading?**
- Run `npm install` to ensure Tailwind is installed
- Restart dev server: `npm run dev`

## Next Steps

1. Deploy to Vercel: `vercel deploy`
2. Add authentication (optional)
3. Add more charts (equity curve, drawdown, strategy breakdown)
4. Add trade filtering (by symbol, date range, strategy)
5. Add export functionality (CSV download)

## Support

For issues or questions:
- Check `ALERTS_SETUP_GUIDE.md` for backend configuration
- Review `COCKPIT_V3_IMPLEMENTATION_COMPLETE.md` for Phase 5 details
- Test backend endpoints before debugging dashboard

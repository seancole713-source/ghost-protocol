'use client'

import { useEffect, useState } from 'react'
import { Activity, TrendingUp, TrendingDown, Target, AlertCircle, BarChart3 } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface Phase5Status {
  enabled: boolean
  total_cycles: number
  trades_today: number
  last_cycle_time: number
}

interface TradeData {
  symbol: string
  side: string
  quantity: number
  price: number
  timestamp: string
}

interface PerformanceMetrics {
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  total_pnl: number
  daily_pnl: number
}

export default function Dashboard() {
  const [phase5, setPhase5] = useState<Phase5Status | null>(null)
  const [trades, setTrades] = useState<TradeData[]>([])
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [pnlHistory, setPnlHistory] = useState<{time: string, pnl: number}[]>([])

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Phase 5 status
        const phase5Res = await fetch('https://ghost-protocol-production.up.railway.app/api/v3/phase5/status')
        const phase5Data = await phase5Res.json()
        setPhase5(phase5Data.phase5)

        // Trade dashboard
        const dashboardRes = await fetch('https://ghost-protocol-production.up.railway.app/api/v3/trade/dashboard')
        const dashboardData = await dashboardRes.json()
        setMetrics(dashboardData.performance)
        setTrades(dashboardData.recent_trades || [])
      } catch (error) {
        console.error('Failed to fetch data:', error)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30s

    return () => clearInterval(interval)
  }, [])

  // WebSocket connection for real-time updates
  useEffect(() => {
    let ws: WebSocket | null = null

    const connect = () => {
      ws = new WebSocket('wss://ghost-protocol-production.up.railway.app/ws/trades')

      ws.onopen = () => {
        console.log('WebSocket connected')
        setWsConnected(true)
        
        // Send ping every 30s
        const pingInterval = setInterval(() => {
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send('ping')
          }
        }, 30000)

        ws.onclose = () => {
          clearInterval(pingInterval)
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.type === 'trade_update') {
            setTrades(prev => [data.data, ...prev].slice(0, 10))
            setMetrics(data.metrics)
            
            // Update P&L history
            setPnlHistory(prev => [...prev, {
              time: new Date().toLocaleTimeString(),
              pnl: data.metrics.total_pnl
            }].slice(-20))
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setWsConnected(false)
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...')
        setWsConnected(false)
        setTimeout(connect, 5000) // Reconnect after 5s
      }
    }

    connect()

    return () => {
      ws?.close()
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-2">Ghost Protocol</h1>
            <p className="text-gray-400">Autonomous Trading Dashboard</p>
          </div>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${wsConnected ? 'bg-green-900/30' : 'bg-red-900/30'}`}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-sm">{wsConnected ? 'Live' : 'Disconnected'}</span>
            </div>
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${phase5?.enabled ? 'bg-blue-900/30' : 'bg-gray-700'}`}>
              <Activity className="w-4 h-4" />
              <span className="text-sm">Phase 5: {phase5?.enabled ? 'Active' : 'Disabled'}</span>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Execution Cycles"
            value={phase5?.total_cycles || 0}
            icon={<Activity className="w-6 h-6" />}
            color="blue"
          />
          <StatCard
            title="Trades Today"
            value={phase5?.trades_today || 0}
            icon={<BarChart3 className="w-6 h-6" />}
            color="purple"
          />
          <StatCard
            title="Total P&L"
            value={`$${metrics?.total_pnl?.toFixed(2) || '0.00'}`}
            icon={metrics && metrics.total_pnl >= 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
            color={metrics && metrics.total_pnl >= 0 ? 'green' : 'red'}
          />
          <StatCard
            title="Win Rate"
            value={`${metrics?.win_rate?.toFixed(1) || '0.0'}%`}
            icon={<Target className="w-6 h-6" />}
            color="yellow"
          />
        </div>

        {/* P&L Chart */}
        {pnlHistory.length > 0 && (
          <div className="bg-gray-800/50 backdrop-blur rounded-lg p-6 mb-8">
            <h2 className="text-xl font-bold mb-4">P&L History</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={pnlHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                  labelStyle={{ color: '#9CA3AF' }}
                />
                <Line type="monotone" dataKey="pnl" stroke="#10B981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Recent Trades */}
        <div className="bg-gray-800/50 backdrop-blur rounded-lg p-6">
          <h2 className="text-xl font-bold mb-4">Recent Trades</h2>
          {trades.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <AlertCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No trades yet. Waiting for 60%+ confidence predictions...</p>
            </div>
          ) : (
            <div className="space-y-3">
              {trades.map((trade, i) => (
                <div key={i} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg hover:bg-gray-700/50 transition">
                  <div className="flex items-center gap-4">
                    <div className={`w-2 h-2 rounded-full ${trade.side === 'BUY' ? 'bg-green-400' : 'bg-red-400'}`} />
                    <div>
                      <div className="font-bold">{trade.symbol}</div>
                      <div className="text-sm text-gray-400">
                        {trade.side} {trade.quantity} @ ${trade.price.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400">
                      {new Date(trade.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Performance Metrics */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <MetricCard title="Total Trades" value={metrics.total_trades} />
            <MetricCard title="Winning Trades" value={metrics.winning_trades} color="green" />
            <MetricCard title="Losing Trades" value={metrics.losing_trades} color="red" />
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ title, value, icon, color }: { title: string, value: string | number, icon: React.ReactNode, color: string }) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/30',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/30',
    green: 'from-green-500/20 to-green-600/20 border-green-500/30',
    red: 'from-red-500/20 to-red-600/20 border-red-500/30',
    yellow: 'from-yellow-500/20 to-yellow-600/20 border-yellow-500/30'
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} backdrop-blur border rounded-lg p-6`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">{title}</span>
        {icon}
      </div>
      <div className="text-3xl font-bold">{value}</div>
    </div>
  )
}

function MetricCard({ title, value, color }: { title: string, value: number, color?: string }) {
  return (
    <div className="bg-gray-800/50 backdrop-blur rounded-lg p-6">
      <div className="text-gray-400 text-sm mb-1">{title}</div>
      <div className={`text-2xl font-bold ${color === 'green' ? 'text-green-400' : color === 'red' ? 'text-red-400' : ''}`}>
        {value}
      </div>
    </div>
  )
}

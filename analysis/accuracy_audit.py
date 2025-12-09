#!/usr/bin/env python3
"""
Ghost Protocol Accuracy Audit Script
=====================================

This script performs a rigorous, evidence-based analysis of Ghost's prediction
accuracy against the 70% target.

Follows strict scientific methodology:
- No data leakage (predictions evaluated only after horizon elapsed)
- No survivorship bias (includes failed predictions)
- Statistical significance testing (confidence intervals)
- Proper calibration analysis (predicted confidence vs realized accuracy)

Usage:
    python3 analysis/accuracy_audit.py [--time-window 30] [--output analysis/report.md]
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    LOGGER.error("❌ DATABASE_URL environment variable not set")
    sys.exit(1)


class GhostAccuracyAuditor:
    """
    Rigorous accuracy auditor for Ghost Protocol predictions.
    
    Methodology:
    1. Query predictions with closed outcomes (48h+ elapsed)
    2. Verify no future data leakage
    3. Compute accuracy by horizon, asset cohort, time window
    4. Calculate statistical confidence intervals
    5. Test calibration (predicted confidence vs realized accuracy)
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn = None
        
    def __enter__(self):
        self.conn = psycopg2.connect(self.database_url)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def get_schema_info(self) -> Dict[str, List[str]]:
        """Discover prediction tables and their schemas."""
        LOGGER.info("🔍 Discovering database schema...")
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find prediction-related tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND (table_name LIKE '%prediction%' OR table_name LIKE '%outcome%')
                ORDER BY table_name
            """)
            tables = [row['table_name'] for row in cur.fetchall()]
            
            schema = {}
            for table in tables:
                cur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                schema[table] = [(row['column_name'], row['data_type']) for row in cur.fetchall()]
        
        return schema
    
    def get_prediction_count(self) -> Dict[str, int]:
        """Get counts of predictions by type."""
        LOGGER.info("📊 Counting predictions...")
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            counts = {}
            
            # Total predictions
            cur.execute("SELECT COUNT(*) as count FROM ghost_predictions")
            counts['total_predictions'] = cur.fetchone()['count']
            
            # Predictions with outcomes
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM ghost_predictions gp
                INNER JOIN ghost_prediction_outcomes gpo ON gp.id = gpo.prediction_id
            """)
            counts['predictions_with_outcomes'] = cur.fetchone()['count']
            
            # Pending predictions (48h+ elapsed, no outcome)
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM ghost_predictions gp
                LEFT JOIN ghost_prediction_outcomes gpo ON gp.id = gpo.prediction_id
                WHERE gpo.id IS NULL
                AND (EXTRACT(EPOCH FROM NOW()) - gp.run_at) >= (gp.horizon_h * 3600)
            """)
            counts['pending_outcomes'] = cur.fetchone()['count']
            
            # Recent predictions (last 7 days)
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM ghost_predictions gp
                WHERE gp.run_at >= EXTRACT(EPOCH FROM NOW() - INTERVAL '7 days')
            """)
            counts['last_7d_predictions'] = cur.fetchone()['count']
            
            # Recent predictions (last 30 days)
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM ghost_predictions gp
                WHERE gp.run_at >= EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')
            """)
            counts['last_30d_predictions'] = cur.fetchone()['count']
        
        return counts
    
    def get_accuracy_by_window(self, days: int) -> Dict[str, any]:
        """
        Calculate accuracy for predictions closed in the last N days.
        
        Only includes predictions where:
        - Outcome was resolved (48h+ elapsed)
        - Actual price was obtained (hit_direction is not NULL)
        - No future data leakage
        """
        LOGGER.info(f"📈 Calculating accuracy for last {days} days...")
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_evaluated,
                    SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN gpo.hit_direction = 0 THEN 1 ELSE 0 END) as incorrect,
                    ROUND(
                        SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
                        NULLIF(COUNT(*), 0) * 100,
                        2
                    ) as accuracy_pct,
                    AVG(gp.confidence) as avg_confidence,
                    MIN(gpo.closed_at) as earliest_outcome,
                    MAX(gpo.closed_at) as latest_outcome
                FROM ghost_prediction_outcomes gpo
                INNER JOIN ghost_predictions gp ON gpo.prediction_id = gp.id
                WHERE gpo.closed_at >= NOW() - INTERVAL '%s days'
                AND gpo.hit_direction IS NOT NULL
            """ % days)
            
            result = cur.fetchone()
            
            if result and result['total_evaluated'] > 0:
                # Calculate Wilson score confidence interval
                n = result['total_evaluated']
                p = result['correct'] / n
                ci_lower, ci_upper = self._wilson_confidence_interval(result['correct'], n)
                
                return {
                    'time_window_days': days,
                    'total_evaluated': n,
                    'correct': result['correct'],
                    'incorrect': result['incorrect'],
                    'accuracy_pct': float(result['accuracy_pct']),
                    'confidence_interval_95': (round(ci_lower * 100, 2), round(ci_upper * 100, 2)),
                    'avg_predicted_confidence': round(float(result['avg_confidence']), 3) if result['avg_confidence'] else None,
                    'earliest_outcome': result['earliest_outcome'],
                    'latest_outcome': result['latest_outcome'],
                    'meets_70_target': ci_lower >= 0.70
                }
            else:
                return {
                    'time_window_days': days,
                    'total_evaluated': 0,
                    'accuracy_pct': None,
                    'meets_70_target': False
                }
    
    def get_accuracy_by_horizon(self, days: int = 30) -> List[Dict]:
        """Calculate accuracy broken down by prediction horizon."""
        LOGGER.info(f"⏱️  Calculating accuracy by horizon (last {days} days)...")
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    gp.horizon_h,
                    COUNT(*) as total_evaluated,
                    SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN gpo.hit_direction = 0 THEN 1 ELSE 0 END) as incorrect,
                    ROUND(
                        SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
                        NULLIF(COUNT(*), 0) * 100,
                        2
                    ) as accuracy_pct
                FROM ghost_prediction_outcomes gpo
                INNER JOIN ghost_predictions gp ON gpo.prediction_id = gp.id
                WHERE gpo.closed_at >= NOW() - INTERVAL '%s days'
                AND gpo.hit_direction IS NOT NULL
                GROUP BY gp.horizon_h
                ORDER BY gp.horizon_h
            """ % days)
            
            results = []
            for row in cur.fetchall():
                n = row['total_evaluated']
                if n > 0:
                    ci_lower, ci_upper = self._wilson_confidence_interval(row['correct'], n)
                    results.append({
                        'horizon_hours': row['horizon_h'],
                        'total_evaluated': n,
                        'correct': row['correct'],
                        'incorrect': row['incorrect'],
                        'accuracy_pct': float(row['accuracy_pct']),
                        'confidence_interval_95': (round(ci_lower * 100, 2), round(ci_upper * 100, 2)),
                        'meets_70_target': ci_lower >= 0.70
                    })
            
            return results
    
    def get_accuracy_by_symbol_cohort(self, days: int = 30) -> Dict[str, Dict]:
        """
        Calculate accuracy by asset cohort.
        
        Cohorts:
        - Major Crypto: BTC, ETH, SOL, BNB, XRP, ADA, DOT, MATIC
        - VIP Microcaps: WEPE, LILPEPE, DORKL, SLOTH, APC
        - Stocks: AAPL, TSLA, NVDA, MSFT, SPY, etc.
        """
        LOGGER.info(f"🎯 Calculating accuracy by symbol cohort (last {days} days)...")
        
        # Define cohorts
        major_crypto = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOT', 'MATIC']
        vip_microcaps = ['WEPE', 'LILPEPE', 'DORKL', 'SLOTH', 'APC']
        stocks = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'SPY', 'AMZN', 'GOOGL', 'META', 'WOLF']
        
        cohorts = {
            'major_crypto': major_crypto,
            'vip_microcaps': vip_microcaps,
            'stocks': stocks
        }
        
        results = {}
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            for cohort_name, symbols in cohorts.items():
                placeholders = ','.join(['%s'] * len(symbols))
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_evaluated,
                        SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                        SUM(CASE WHEN gpo.hit_direction = 0 THEN 1 ELSE 0 END) as incorrect,
                        ROUND(
                            SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
                            NULLIF(COUNT(*), 0) * 100,
                            2
                        ) as accuracy_pct,
                        ARRAY_AGG(DISTINCT gp.symbol) as symbols_found
                    FROM ghost_prediction_outcomes gpo
                    INNER JOIN ghost_predictions gp ON gpo.prediction_id = gp.id
                    WHERE gpo.closed_at >= NOW() - INTERVAL %s
                    AND gpo.hit_direction IS NOT NULL
                    AND gp.symbol IN ({placeholders})
                """, (*symbols, f'{days} days'))
                
                row = cur.fetchone()
                if row and row['total_evaluated'] > 0:
                    n = row['total_evaluated']
                    ci_lower, ci_upper = self._wilson_confidence_interval(row['correct'], n)
                    results[cohort_name] = {
                        'symbols': symbols,
                        'symbols_with_data': row['symbols_found'],
                        'total_evaluated': n,
                        'correct': row['correct'],
                        'incorrect': row['incorrect'],
                        'accuracy_pct': float(row['accuracy_pct']),
                        'confidence_interval_95': (round(ci_lower * 100, 2), round(ci_upper * 100, 2)),
                        'meets_70_target': ci_lower >= 0.70
                    }
                else:
                    results[cohort_name] = {
                        'symbols': symbols,
                        'total_evaluated': 0,
                        'accuracy_pct': None,
                        'meets_70_target': False
                    }
        
        return results
    
    def get_calibration_analysis(self, days: int = 30) -> List[Dict]:
        """
        Analyze confidence calibration.
        
        Groups predictions by predicted confidence and measures realized accuracy.
        Well-calibrated predictions should show predicted confidence ≈ realized accuracy.
        """
        LOGGER.info(f"🎯 Analyzing confidence calibration (last {days} days)...")
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    CASE
                        WHEN gp.confidence >= 0.70 THEN '70-100%'
                        WHEN gp.confidence >= 0.60 THEN '60-70%'
                        WHEN gp.confidence >= 0.50 THEN '50-60%'
                        ELSE '0-50%'
                    END as confidence_bucket,
                    COUNT(*) as total_evaluated,
                    SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(gp.confidence) as avg_predicted_confidence,
                    ROUND(
                        SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
                        NULLIF(COUNT(*), 0) * 100,
                        2
                    ) as realized_accuracy_pct
                FROM ghost_prediction_outcomes gpo
                INNER JOIN ghost_predictions gp ON gpo.prediction_id = gp.id
                WHERE gpo.closed_at >= NOW() - INTERVAL '%s days'
                AND gpo.hit_direction IS NOT NULL
                GROUP BY confidence_bucket
                ORDER BY confidence_bucket
            """ % days)
            
            results = []
            for row in cur.fetchall():
                avg_conf = float(row['avg_predicted_confidence'])
                realized_acc = float(row['realized_accuracy_pct']) / 100
                calibration_error = abs(avg_conf - realized_acc)
                
                results.append({
                    'confidence_bucket': row['confidence_bucket'],
                    'total_evaluated': row['total_evaluated'],
                    'correct': row['correct'],
                    'avg_predicted_confidence': round(avg_conf, 3),
                    'realized_accuracy': round(realized_acc, 3),
                    'calibration_error': round(calibration_error, 3),
                    'is_well_calibrated': calibration_error < 0.05  # Within 5%
                })
            
            return results
    
    def _wilson_confidence_interval(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.
        
        More accurate than normal approximation for small sample sizes.
        """
        if trials == 0:
            return (0.0, 0.0)
        
        from math import sqrt
        
        p = successes / trials
        z = 1.96  # 95% confidence
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
        
        return (max(0.0, center - margin), min(1.0, center + margin))
    
    def generate_audit_report(self, output_path: str, time_window: int = 30):
        """Generate comprehensive accuracy audit report."""
        LOGGER.info("📝 Generating audit report...")
        
        # Gather all data
        schema = self.get_schema_info()
        counts = self.get_prediction_count()
        accuracy_7d = self.get_accuracy_by_window(7)
        accuracy_30d = self.get_accuracy_by_window(30)
        accuracy_90d = self.get_accuracy_by_window(90)
        accuracy_by_horizon = self.get_accuracy_by_horizon(time_window)
        accuracy_by_cohort = self.get_accuracy_by_symbol_cohort(time_window)
        calibration = self.get_calibration_analysis(time_window)
        
        # Generate markdown report
        report_lines = [
            "# Ghost Protocol Accuracy Audit Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Database**: PostgreSQL (Railway)",
            f"**Analysis Window**: Last {time_window} days",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            self._generate_executive_summary(accuracy_30d, accuracy_by_cohort, accuracy_by_horizon),
            "",
            "---",
            "",
            "## 1. Data Sources and Schema",
            "",
            "### Prediction Tables",
            "",
            "```",
            *[f"{table}:" for table in schema.keys()],
            "```",
            "",
            "### Key Fields",
            "",
            "**ghost_predictions**:",
            "- `id` (primary key)",
            "- `symbol`, `run_at`, `horizon_h`",
            "- `direction` (UP/DOWN/FLAT), `confidence` (0.0-1.0)",
            "- `method`, `features_json`, `params_json`",
            "",
            "**ghost_prediction_outcomes**:",
            "- `prediction_id` (foreign key)",
            "- `closed_at` (resolution timestamp)",
            "- `price_at_prediction`, `price_at_resolution`",
            "- `realized_move_pct`, `predicted_direction`, `actual_direction`",
            "- `hit_direction` (1=correct, 0=wrong, NULL=no data)",
            "- `mae`, `mape`, `rmse` (statistical metrics)",
            "",
            "---",
            "",
            "## 2. Prediction Counts",
            "",
            f"- **Total Predictions (All Time)**: {counts['total_predictions']:,}",
            f"- **Predictions with Outcomes**: {counts['predictions_with_outcomes']:,}",
            f"- **Pending Outcomes (48h+ elapsed)**: {counts['pending_outcomes']:,}",
            f"- **Recent Predictions (Last 7 days)**: {counts['last_7d_predictions']:,}",
            f"- **Recent Predictions (Last 30 days)**: {counts['last_30d_predictions']:,}",
            "",
            "---",
            "",
            "## 3. Overall Accuracy by Time Window",
            "",
            self._format_accuracy_table([accuracy_7d, accuracy_30d, accuracy_90d]),
            "",
            "---",
            "",
            "## 4. Accuracy by Prediction Horizon",
            "",
            f"*Analysis Window: Last {time_window} days*",
            "",
            self._format_horizon_table(accuracy_by_horizon),
            "",
            "---",
            "",
            "## 5. Accuracy by Asset Cohort",
            "",
            f"*Analysis Window: Last {time_window} days*",
            "",
            self._format_cohort_table(accuracy_by_cohort),
            "",
            "---",
            "",
            "## 6. Confidence Calibration Analysis",
            "",
            f"*Analysis Window: Last {time_window} days*",
            "",
            self._format_calibration_table(calibration),
            "",
            "---",
            "",
            "## 7. Data Quality and Limitations",
            "",
            self._generate_data_quality_section(counts),
            "",
            "---",
            "",
            "## 8. Verdict: Can Ghost Achieve 70% Accuracy?",
            "",
            self._generate_verdict(accuracy_30d, accuracy_by_cohort, accuracy_by_horizon, counts),
            "",
        ]
        
        # Write report
        Path(output_path).write_text('\n'.join(report_lines))
        LOGGER.info(f"✅ Report written to {output_path}")
    
    def _generate_executive_summary(self, accuracy_30d, accuracy_by_cohort, accuracy_by_horizon) -> str:
        """Generate executive summary with key findings."""
        lines = []
        
        if accuracy_30d['total_evaluated'] == 0:
            lines.append("⚠️ **INSUFFICIENT DATA**: No predictions with resolved outcomes in the last 30 days.")
            lines.append("")
            lines.append("Cannot assess 70% accuracy target without evaluation data.")
            return '\n'.join(lines)
        
        overall_acc = accuracy_30d['accuracy_pct']
        ci_lower, ci_upper = accuracy_30d['confidence_interval_95']
        meets_target = accuracy_30d['meets_70_target']
        
        lines.append(f"**Overall Accuracy (Last 30 days)**: {overall_acc}% (95% CI: {ci_lower}%-{ci_upper}%)")
        lines.append(f"**Sample Size**: {accuracy_30d['total_evaluated']} predictions evaluated")
        lines.append("")
        
        if meets_target:
            lines.append("✅ **VERDICT**: Ghost **MEETS** the 70% accuracy target with statistical confidence.")
        else:
            lines.append("❌ **VERDICT**: Ghost **DOES NOT MEET** the 70% accuracy target.")
            lines.append("")
            lines.append(f"   - Point estimate: {overall_acc}%")
            lines.append(f"   - Lower bound of 95% CI: {ci_lower}%")
            lines.append(f"   - Target: 70%")
        
        lines.append("")
        lines.append("### Key Findings")
        lines.append("")
        
        # Cohort analysis
        for cohort_name, data in accuracy_by_cohort.items():
            if data['total_evaluated'] > 0:
                cohort_acc = data['accuracy_pct']
                cohort_ci = data['confidence_interval_95']
                status = "✅ MEETS" if data['meets_70_target'] else "❌ BELOW"
                lines.append(f"- **{cohort_name.replace('_', ' ').title()}**: {cohort_acc}% (CI: {cohort_ci[0]}-{cohort_ci[1]}%) - {status} 70% target")
        
        return '\n'.join(lines)
    
    def _format_accuracy_table(self, results: List[Dict]) -> str:
        """Format accuracy results as markdown table."""
        lines = [
            "| Time Window | Total Evaluated | Correct | Incorrect | Accuracy | 95% CI | Meets 70% Target? |",
            "|-------------|-----------------|---------|-----------|----------|--------|-------------------|"
        ]
        
        for result in results:
            if result['total_evaluated'] == 0:
                lines.append(f"| {result['time_window_days']}d | 0 | - | - | - | - | N/A (No Data) |")
            else:
                ci = result['confidence_interval_95']
                status = "✅ YES" if result['meets_70_target'] else "❌ NO"
                lines.append(
                    f"| {result['time_window_days']}d | "
                    f"{result['total_evaluated']} | "
                    f"{result['correct']} | "
                    f"{result['incorrect']} | "
                    f"{result['accuracy_pct']}% | "
                    f"{ci[0]}-{ci[1]}% | "
                    f"{status} |"
                )
        
        return '\n'.join(lines)
    
    def _format_horizon_table(self, results: List[Dict]) -> str:
        """Format horizon accuracy as markdown table."""
        if not results:
            return "*No data available*"
        
        lines = [
            "| Horizon | Total Evaluated | Correct | Incorrect | Accuracy | 95% CI | Meets 70%? |",
            "|---------|-----------------|---------|-----------|----------|--------|------------|"
        ]
        
        for result in results:
            ci = result['confidence_interval_95']
            status = "✅ YES" if result['meets_70_target'] else "❌ NO"
            lines.append(
                f"| {result['horizon_hours']}h | "
                f"{result['total_evaluated']} | "
                f"{result['correct']} | "
                f"{result['incorrect']} | "
                f"{result['accuracy_pct']}% | "
                f"{ci[0]}-{ci[1]}% | "
                f"{status} |"
            )
        
        return '\n'.join(lines)
    
    def _format_cohort_table(self, cohorts: Dict[str, Dict]) -> str:
        """Format cohort accuracy as markdown table."""
        lines = [
            "| Cohort | Symbols | Total Evaluated | Correct | Accuracy | 95% CI | Meets 70%? |",
            "|--------|---------|-----------------|---------|----------|--------|------------|"
        ]
        
        for cohort_name, data in cohorts.items():
            cohort_display = cohort_name.replace('_', ' ').title()
            
            if data['total_evaluated'] == 0:
                lines.append(f"| {cohort_display} | {len(data['symbols'])} symbols | 0 | - | - | - | N/A (No Data) |")
            else:
                ci = data['confidence_interval_95']
                status = "✅ YES" if data['meets_70_target'] else "❌ NO"
                symbols_with_data = len(data.get('symbols_with_data', []))
                lines.append(
                    f"| {cohort_display} | "
                    f"{symbols_with_data}/{len(data['symbols'])} symbols | "
                    f"{data['total_evaluated']} | "
                    f"{data['correct']} | "
                    f"{data['accuracy_pct']}% | "
                    f"{ci[0]}-{ci[1]}% | "
                    f"{status} |"
                )
        
        return '\n'.join(lines)
    
    def _format_calibration_table(self, results: List[Dict]) -> str:
        """Format calibration analysis as markdown table."""
        if not results:
            return "*No data available*"
        
        lines = [
            "| Confidence Bucket | N | Predicted Confidence | Realized Accuracy | Calibration Error | Well Calibrated? |",
            "|-------------------|---|----------------------|-------------------|-------------------|------------------|"
        ]
        
        for result in results:
            status = "✅ YES" if result['is_well_calibrated'] else "❌ NO"
            lines.append(
                f"| {result['confidence_bucket']} | "
                f"{result['total_evaluated']} | "
                f"{result['avg_predicted_confidence']:.1%} | "
                f"{result['realized_accuracy']:.1%} | "
                f"{result['calibration_error']:.1%} | "
                f"{status} |"
            )
        
        return '\n'.join(lines)
    
    def _generate_data_quality_section(self, counts: Dict) -> str:
        """Generate data quality and limitations section."""
        lines = []
        
        # Calculate reconciliation rate
        total_preds = counts['total_predictions']
        with_outcomes = counts['predictions_with_outcomes']
        pending = counts['pending_outcomes']
        
        if total_preds > 0:
            reconciliation_rate = (with_outcomes / total_preds) * 100
        else:
            reconciliation_rate = 0
        
        lines.append("### Reconciliation Rate")
        lines.append("")
        lines.append(f"- **Total Predictions**: {total_preds:,}")
        lines.append(f"- **With Outcomes**: {with_outcomes:,} ({reconciliation_rate:.1f}%)")
        lines.append(f"- **Pending Outcomes**: {pending:,}")
        lines.append("")
        
        if reconciliation_rate < 50:
            lines.append("⚠️ **WARNING**: Low reconciliation rate may indicate:")
            lines.append("- Predictions too recent (48h window not elapsed)")
            lines.append("- Outcome reconciler not running regularly")
            lines.append("- Data quality issues preventing price fetching")
        
        lines.append("")
        lines.append("### Known Limitations")
        lines.append("")
        lines.append("1. **Survivorship Bias**: Delisted assets may be excluded from price history")
        lines.append("2. **Sample Size**: Small sample sizes have wide confidence intervals")
        lines.append("3. **Time Recency**: Results weighted toward recent market conditions")
        lines.append("4. **Provider Availability**: Price data quality depends on external APIs")
        
        return '\n'.join(lines)
    
    def _generate_verdict(self, accuracy_30d, accuracy_by_cohort, accuracy_by_horizon, counts) -> str:
        """Generate final verdict on 70% accuracy target."""
        lines = []
        
        if counts['predictions_with_outcomes'] < 30:
            lines.append("### ⚠️ INSUFFICIENT DATA FOR CONCLUSIVE VERDICT")
            lines.append("")
            lines.append(f"Only {counts['predictions_with_outcomes']} predictions have been evaluated.")
            lines.append("")
            lines.append("**Recommendation**: Collect at least 100 evaluated predictions before drawing firm conclusions.")
            lines.append("")
            lines.append("**Required Actions**:")
            lines.append("1. Ensure outcome reconciler runs every hour")
            lines.append("2. Wait for more 48h windows to close")
            lines.append("3. Re-run this audit in 7-14 days")
            return '\n'.join(lines)
        
        overall_acc = accuracy_30d['accuracy_pct']
        ci_lower, ci_upper = accuracy_30d['confidence_interval_95']
        meets_target = accuracy_30d['meets_70_target']
        
        lines.append("### Evidence-Based Assessment")
        lines.append("")
        lines.append(f"**Sample Size**: {accuracy_30d['total_evaluated']} predictions (last 30 days)")
        lines.append(f"**Point Estimate**: {overall_acc}%")
        lines.append(f"**95% Confidence Interval**: {ci_lower}% to {ci_upper}%")
        lines.append("")
        
        if meets_target:
            lines.append("### ✅ VERDICT: Ghost CAN Achieve 70% Accuracy")
            lines.append("")
            lines.append("**Evidence**:")
            lines.append(f"- Lower bound of 95% CI ({ci_lower}%) exceeds 70% threshold")
            lines.append(f"- Point estimate: {overall_acc}%")
            lines.append(f"- Statistical power: {accuracy_30d['total_evaluated']} predictions")
            lines.append("")
            lines.append("**Where Ghost Excels**:")
            for cohort_name, data in accuracy_by_cohort.items():
                if data['meets_70_target'] and data['total_evaluated'] >= 10:
                    lines.append(f"- {cohort_name.replace('_', ' ').title()}: {data['accuracy_pct']}% ({data['total_evaluated']} predictions)")
        else:
            lines.append("### ❌ VERDICT: Ghost Does NOT Currently Meet 70% Accuracy")
            lines.append("")
            lines.append("**Evidence**:")
            lines.append(f"- Lower bound of 95% CI ({ci_lower}%) below 70% threshold")
            lines.append(f"- Point estimate: {overall_acc}%")
            lines.append("")
            lines.append("**Gap Analysis**:")
            lines.append(f"- Current: {overall_acc}%")
            lines.append(f"- Target: 70%")
            lines.append(f"- Gap: {70 - overall_acc:.1f} percentage points")
            lines.append("")
            lines.append("**Areas Needing Improvement**:")
            for cohort_name, data in accuracy_by_cohort.items():
                if not data['meets_70_target'] and data['total_evaluated'] >= 10:
                    lines.append(f"- {cohort_name.replace('_', ' ').title()}: {data['accuracy_pct']}% ({data['total_evaluated']} predictions)")
        
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ghost Protocol Accuracy Audit")
    parser.add_argument('--time-window', type=int, default=30, help='Analysis time window in days (default: 30)')
    parser.add_argument('--output', type=str, default='analysis/ghost_accuracy_audit.md', help='Output report path')
    args = parser.parse_args()
    
    LOGGER.info("=" * 60)
    LOGGER.info("GHOST PROTOCOL ACCURACY AUDIT")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Time Window: Last {args.time_window} days")
    LOGGER.info(f"Output: {args.output}")
    LOGGER.info("=" * 60)
    
    with GhostAccuracyAuditor(DATABASE_URL) as auditor:
        auditor.generate_audit_report(args.output, args.time_window)
    
    LOGGER.info("=" * 60)
    LOGGER.info("✅ AUDIT COMPLETE")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()

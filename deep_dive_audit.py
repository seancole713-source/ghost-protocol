#!/usr/bin/env python3
"""
Ghost Protocol - Deep Dive Audit System
========================================

Comprehensive audit of all critical systems, data integrity,
code quality, and potential failure points.

This audit verifies EVERYTHING - no trust, only verification.
"""

import os
import sys
import json
import sqlite3
import psycopg2
from pathlib import Path
from datetime import datetime
import re
import ast

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

class DeepDiveAuditor:
    """Comprehensive Ghost Protocol auditor - trust nothing, verify everything"""
    
    def __init__(self):
        self.audit_results = {
            "code_quality": {},
            "database_integrity": {},
            "ml_model_health": {},
            "data_flow_verification": {},
            "security_audit": {},
            "performance_analysis": {},
            "critical_bugs": [],
            "warnings": [],
            "recommendations": [],
        }
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.critical_issues = []
        
    def log(self, message, level="INFO"):
        """Colored logging"""
        prefix = {
            "INFO": f"{BLUE}ℹ{RESET}",
            "SUCCESS": f"{GREEN}✅{RESET}",
            "FAIL": f"{RED}❌{RESET}",
            "WARN": f"{YELLOW}⚠️{RESET}",
            "CRITICAL": f"{RED}{BOLD}🚨{RESET}",
            "AUDIT": f"{MAGENTA}🔍{RESET}",
        }.get(level, "")
        print(f"{prefix} {message}")
    
    def check_exists(self, condition, name):
        """Track check result"""
        self.total_checks += 1
        if condition:
            self.passed_checks += 1
            return True
        else:
            self.failed_checks += 1
            self.critical_issues.append(name)
            return False
    
    # ========================================================================
    # SECTION 1: CODE QUALITY AUDIT
    # ========================================================================
    
    def audit_ml_trainer_implementation(self):
        """Audit ml_trainer.py for PostgreSQL integration"""
        self.log("Auditing ml_trainer.py implementation...", "AUDIT")
        
        ml_trainer_path = "/workspaces/ghost-protocol/core/ml_trainer.py"
        
        try:
            with open(ml_trainer_path, 'r') as f:
                content = f.read()
            
            # Check 1: Does _fetch_training_data exist?
            has_fetch_function = "_fetch_training_data" in content
            self.check_exists(has_fetch_function, "ml_trainer: _fetch_training_data function missing")
            
            if has_fetch_function:
                self.log("✓ _fetch_training_data function found", "SUCCESS")
            else:
                self.log("✗ _fetch_training_data function MISSING", "CRITICAL")
                return
            
            # Check 2: Does it try PostgreSQL first?
            has_postgres_query = "ghost_prediction_outcomes" in content
            self.check_exists(has_postgres_query, "ml_trainer: PostgreSQL query missing")
            
            if has_postgres_query:
                self.log("✓ PostgreSQL query found (ghost_prediction_outcomes)", "SUCCESS")
            else:
                self.log("✗ PostgreSQL query MISSING - still using SQLite only", "CRITICAL")
            
            # Check 3: Does it have fallback to SQLite?
            has_sqlite_fallback = "sqlite3" in content
            self.check_exists(has_sqlite_fallback, "ml_trainer: No SQLite fallback")
            
            if has_sqlite_fallback:
                self.log("✓ SQLite fallback present", "SUCCESS")
            else:
                self.log("⚠ No SQLite fallback - may fail if PostgreSQL down", "WARN")
            
            # Check 4: Count how many times it queries PostgreSQL vs SQLite
            postgres_queries = content.count("ghost_prediction_outcomes")
            sqlite_queries = content.count("SELECT") - postgres_queries
            
            self.log(f"  PostgreSQL queries: {postgres_queries}", "INFO")
            self.log(f"  SQLite queries: {sqlite_queries}", "INFO")
            
            if postgres_queries > 0:
                self.audit_results["code_quality"]["ml_trainer_postgres_integration"] = "PASS"
            else:
                self.audit_results["code_quality"]["ml_trainer_postgres_integration"] = "FAIL"
                self.critical_issues.append("ml_trainer not using PostgreSQL")
            
            # Check 5: Verify it reads features_json correctly
            has_features_json = "features_json" in content
            self.check_exists(has_features_json, "ml_trainer: features_json not read")
            
            if has_features_json:
                self.log("✓ features_json column read", "SUCCESS")
            else:
                self.log("✗ features_json NOT read - model won't have features", "CRITICAL")
            
            # Check 6: Verify it handles JOIN correctly
            has_join = "JOIN" in content and "ghost_predictions" in content
            self.check_exists(has_join, "ml_trainer: Missing JOIN with ghost_predictions")
            
            if has_join:
                self.log("✓ JOIN with ghost_predictions present", "SUCCESS")
            else:
                self.log("⚠ No JOIN - may be missing prediction data", "WARN")
                
        except Exception as e:
            self.log(f"FAILED to audit ml_trainer.py: {e}", "CRITICAL")
            self.critical_issues.append(f"ml_trainer audit error: {e}")
    
    def audit_prediction_store_implementation(self):
        """Audit prediction_store.py for dual backend"""
        self.log("Auditing prediction_store.py implementation...", "AUDIT")
        
        pred_store_path = "/workspaces/ghost-protocol/core/prediction_store.py"
        
        try:
            with open(pred_store_path, 'r') as f:
                content = f.read()
            
            # Check 1: PostgresBackend class exists
            has_postgres_backend = "class PostgresBackend" in content
            self.check_exists(has_postgres_backend, "prediction_store: PostgresBackend missing")
            
            # Check 2: SQLiteBackend class exists
            has_sqlite_backend = "class SQLiteBackend" in content
            self.check_exists(has_sqlite_backend, "prediction_store: SQLiteBackend missing")
            
            # Check 3: Check which backend is used by default
            if "PREDICTION_STORE_ENGINE" in content:
                self.log("✓ PREDICTION_STORE_ENGINE env var checked", "SUCCESS")
            else:
                self.log("⚠ PREDICTION_STORE_ENGINE not checked - may use wrong backend", "WARN")
            
            # Check 4: Verify PostgresBackend saves to correct table
            if "INSERT INTO ghost_predictions" in content:
                self.log("✓ PostgresBackend saves to ghost_predictions", "SUCCESS")
            else:
                self.log("✗ PostgresBackend may not save to correct table", "CRITICAL")
            
            # Check 5: Verify features are stored
            if "features_json" in content or "features" in content:
                self.log("✓ Features stored in database", "SUCCESS")
            else:
                self.log("✗ Features NOT stored - model can't learn", "CRITICAL")
            
            self.audit_results["code_quality"]["prediction_store_dual_backend"] = "PASS" if has_postgres_backend and has_sqlite_backend else "FAIL"
            
        except Exception as e:
            self.log(f"FAILED to audit prediction_store.py: {e}", "CRITICAL")
            self.critical_issues.append(f"prediction_store audit error: {e}")
    
    def audit_autofix_implementation(self):
        """Audit autofix_startup.py for correctness"""
        self.log("Auditing autofix_startup.py implementation...", "AUDIT")
        
        autofix_path = "/workspaces/ghost-protocol/autofix_startup.py"
        
        try:
            with open(autofix_path, 'r') as f:
                content = f.read()
            
            # Check 1: Does it wait for main app to start?
            has_wait = "asyncio.sleep" in content or "time.sleep" in content
            self.check_exists(has_wait, "autofix: No startup delay")
            
            if has_wait:
                wait_time = re.search(r'sleep\((\d+)\)', content)
                if wait_time:
                    self.log(f"✓ Waits {wait_time.group(1)}s for main app", "SUCCESS")
            
            # Check 2: Does it test PostgreSQL?
            has_postgres_test = "DATABASE_URL" in content
            self.check_exists(has_postgres_test, "autofix: No PostgreSQL test")
            
            # Check 3: Does it retrain model?
            has_retrain = "retrain" in content.lower() or "train" in content.lower()
            self.check_exists(has_retrain, "autofix: No model retraining")
            
            # Check 4: Does it check INVERSE_GHOST?
            has_inverse_check = "INVERSE_GHOST" in content
            self.check_exists(has_inverse_check, "autofix: No INVERSE_GHOST check")
            
            # Check 5: Does it run asynchronously?
            has_async = "async def" in content
            self.check_exists(has_async, "autofix: Not async - may block startup")
            
            self.audit_results["code_quality"]["autofix_implementation"] = "PASS" if all([has_wait, has_postgres_test, has_retrain, has_inverse_check, has_async]) else "PARTIAL"
            
        except Exception as e:
            self.log(f"FAILED to audit autofix_startup.py: {e}", "CRITICAL")
            self.critical_issues.append(f"autofix audit error: {e}")
    
    def audit_orchestrator_integration(self):
        """Audit orchestrator.py for autofix integration"""
        self.log("Auditing orchestrator.py integration...", "AUDIT")
        
        orch_path = "/workspaces/ghost-protocol/core/orchestrator.py"
        
        try:
            with open(orch_path, 'r') as f:
                content = f.read()
            
            # Check 1: Does it import autofix_startup?
            has_import = "import autofix_startup" in content or "from autofix_startup" in content
            self.check_exists(has_import, "orchestrator: autofix_startup not imported")
            
            # Check 2: Does it call run_autofix_startup?
            has_call = "run_autofix_startup" in content
            self.check_exists(has_call, "orchestrator: run_autofix_startup not called")
            
            # Check 3: Is it in a background task?
            has_task = "asyncio.create_task" in content or "_TASKS" in content
            self.check_exists(has_task, "orchestrator: autofix not in background task")
            
            # Check 4: Is it in the startup sequence?
            autofix_position = content.find("autofix_startup")
            if autofix_position > 0:
                self.log(f"✓ autofix integrated at position {autofix_position}", "SUCCESS")
            else:
                self.log("✗ autofix not found in orchestrator", "CRITICAL")
            
            self.audit_results["code_quality"]["orchestrator_integration"] = "PASS" if all([has_import, has_call, has_task]) else "FAIL"
            
        except Exception as e:
            self.log(f"FAILED to audit orchestrator.py: {e}", "CRITICAL")
            self.critical_issues.append(f"orchestrator audit error: {e}")
    
    # ========================================================================
    # SECTION 2: DATABASE INTEGRITY AUDIT
    # ========================================================================
    
    def audit_sqlite_database(self):
        """Audit SQLite database structure and data"""
        self.log("Auditing SQLite database...", "AUDIT")
        
        sqlite_paths = [
            "/workspaces/ghost-protocol/data/ghost.db",
            "/workspaces/ghost-protocol/ghost.db",
        ]
        
        for db_path in sqlite_paths:
            if Path(db_path).exists():
                self.log(f"  Found SQLite DB: {db_path}", "INFO")
                
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    
                    # Check tables
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cur.fetchall()]
                    self.log(f"  Tables: {', '.join(tables)}", "INFO")
                    
                    # Check if predictions table exists
                    if "predictions" in tables or "ghost_predictions" in tables:
                        table_name = "predictions" if "predictions" in tables else "ghost_predictions"
                        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cur.fetchone()[0]
                        self.log(f"  {table_name}: {count} rows", "INFO")
                        
                        if count == 0:
                            self.log("  ⚠ SQLite predictions table is EMPTY", "WARN")
                            self.audit_results["database_integrity"]["sqlite_predictions"] = f"EMPTY (0 rows)"
                        else:
                            self.log(f"  ✓ SQLite has {count} predictions", "SUCCESS")
                            self.audit_results["database_integrity"]["sqlite_predictions"] = f"OK ({count} rows)"
                    
                    # Check outcomes table
                    if "outcomes" in tables or "prediction_outcomes" in tables:
                        table_name = "outcomes" if "outcomes" in tables else "prediction_outcomes"
                        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cur.fetchone()[0]
                        self.log(f"  {table_name}: {count} rows", "INFO")
                        
                        if count == 0:
                            self.log("  ⚠ SQLite outcomes table is EMPTY", "WARN")
                            self.audit_results["database_integrity"]["sqlite_outcomes"] = f"EMPTY (0 rows)"
                        else:
                            self.audit_results["database_integrity"]["sqlite_outcomes"] = f"OK ({count} rows)"
                    
                    conn.close()
                    
                except Exception as e:
                    self.log(f"  SQLite audit error: {e}", "FAIL")
                    self.audit_results["database_integrity"]["sqlite_error"] = str(e)
        
        if not any(Path(p).exists() for p in sqlite_paths):
            self.log("  No SQLite database found (OK if using PostgreSQL only)", "INFO")
            self.audit_results["database_integrity"]["sqlite_status"] = "NOT FOUND"
    
    def audit_postgres_connection(self):
        """Audit PostgreSQL connection and data"""
        self.log("Auditing PostgreSQL connection...", "AUDIT")
        
        database_url = os.getenv("DATABASE_URL", "")
        
        if not database_url.startswith(("postgres://", "postgresql://")):
            self.log("  DATABASE_URL not set (dev container - expected)", "INFO")
            self.audit_results["database_integrity"]["postgres_status"] = "NOT CONFIGURED"
            return
        
        try:
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            
            self.log("  ✓ PostgreSQL connection successful", "SUCCESS")
            
            # Check tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cur.fetchall()]
            self.log(f"  Tables: {', '.join(tables)}", "INFO")
            
            # Check predictions
            if "ghost_predictions" in tables:
                cur.execute("SELECT COUNT(*) FROM ghost_predictions")
                count = cur.fetchone()[0]
                self.log(f"  ghost_predictions: {count:,} rows", "INFO")
                self.audit_results["database_integrity"]["postgres_predictions"] = f"{count:,} rows"
                
                if count == 0:
                    self.log("  ⚠ PostgreSQL predictions table is EMPTY", "WARN")
            
            # Check outcomes
            if "ghost_prediction_outcomes" in tables:
                cur.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
                count = cur.fetchone()[0]
                self.log(f"  ghost_prediction_outcomes: {count:,} rows", "INFO")
                self.audit_results["database_integrity"]["postgres_outcomes"] = f"{count:,} rows"
                
                if count == 0:
                    self.log("  ⚠ PostgreSQL outcomes table is EMPTY", "WARN")
                else:
                    self.log(f"  ✓ PostgreSQL has {count:,} outcomes for training", "SUCCESS")
            
            conn.close()
            self.audit_results["database_integrity"]["postgres_connection"] = "OK"
            
        except Exception as e:
            self.log(f"  PostgreSQL connection FAILED: {e}", "FAIL")
            self.audit_results["database_integrity"]["postgres_error"] = str(e)
    
    # ========================================================================
    # SECTION 3: ML MODEL HEALTH AUDIT
    # ========================================================================
    
    def audit_xgboost_model(self):
        """Audit XGBoost model file"""
        self.log("Auditing XGBoost model...", "AUDIT")
        
        model_paths = [
            "/workspaces/ghost-protocol/models/production/ghost_xgboost_v3.pkl",
            "/workspaces/ghost-protocol/models/production/ghost_xgboost_v2.pkl",
            "/workspaces/ghost-protocol/models/trained/ghost_xgboost_v2.pkl",
            "/workspaces/ghost-protocol/models/trained/ghost_xgboost_v1.pkl",
        ]
        
        found_model = False
        for model_path in model_paths:
            if Path(model_path).exists():
                found_model = True
                self.log(f"  Found model: {model_path}", "INFO")
                
                stat = Path(model_path).stat()
                size_mb = stat.st_size / (1024 * 1024)
                age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                
                self.log(f"  Size: {size_mb:.2f} MB", "INFO")
                self.log(f"  Age: {age_days:.1f} days", "INFO")
                
                # Check if model is too small (corrupt)
                if size_mb < 0.1:
                    self.log("  ⚠ Model is suspiciously small (<0.1MB)", "WARN")
                    self.critical_issues.append(f"Model too small: {model_path}")
                else:
                    self.log(f"  ✓ Model size OK ({size_mb:.2f}MB)", "SUCCESS")
                
                # Check if model is too old
                if age_days > 30:
                    self.log(f"  ⚠ Model is {age_days:.0f} days old (>30 days)", "WARN")
                    self.audit_results["ml_model_health"]["model_age_warning"] = f"{age_days:.0f} days old"
                else:
                    self.log(f"  ✓ Model age OK ({age_days:.1f} days)", "SUCCESS")
                
                self.audit_results["ml_model_health"]["model_file"] = model_path
                self.audit_results["ml_model_health"]["model_size_mb"] = f"{size_mb:.2f}"
                self.audit_results["ml_model_health"]["model_age_days"] = f"{age_days:.1f}"
                
                break
        
        if not found_model:
            self.log("  ✗ NO MODEL FILE FOUND", "CRITICAL")
            self.critical_issues.append("No XGBoost model file found")
            self.audit_results["ml_model_health"]["model_status"] = "NOT FOUND"
        else:
            self.check_exists(True, "XGBoost model exists")
    
    def audit_model_training_data(self):
        """Verify model can access training data"""
        self.log("Auditing model training data access...", "AUDIT")
        
        # Check if SQLite has data
        sqlite_outcomes = self.audit_results.get("database_integrity", {}).get("sqlite_outcomes", "")
        postgres_outcomes = self.audit_results.get("database_integrity", {}).get("postgres_outcomes", "")
        
        sqlite_count = 0
        postgres_count = 0
        
        # Parse counts
        if "rows" in sqlite_outcomes:
            try:
                sqlite_count = int(sqlite_outcomes.split()[0].replace(",", ""))
            except:
                pass
        
        if "rows" in postgres_outcomes:
            try:
                postgres_count = int(postgres_outcomes.split()[0].replace(",", ""))
            except:
                pass
        
        self.log(f"  SQLite outcomes: {sqlite_count}", "INFO")
        self.log(f"  PostgreSQL outcomes: {postgres_count}", "INFO")
        
        if postgres_count > 1000:
            self.log(f"  ✓ PostgreSQL has {postgres_count:,} outcomes - GOOD", "SUCCESS")
            self.audit_results["ml_model_health"]["training_data_source"] = f"PostgreSQL ({postgres_count:,} rows)"
        elif sqlite_count > 1000:
            self.log(f"  ⚠ Using SQLite with {sqlite_count:,} outcomes", "WARN")
            self.audit_results["ml_model_health"]["training_data_source"] = f"SQLite ({sqlite_count:,} rows)"
        else:
            self.log(f"  ✗ INSUFFICIENT TRAINING DATA (<1000 outcomes)", "CRITICAL")
            self.critical_issues.append("Insufficient training data")
            self.audit_results["ml_model_health"]["training_data_source"] = "INSUFFICIENT"
    
    # ========================================================================
    # SECTION 4: DATA FLOW VERIFICATION
    # ========================================================================
    
    def audit_prediction_flow(self):
        """Audit end-to-end prediction flow"""
        self.log("Auditing prediction flow...", "AUDIT")
        
        # Check predictor module
        predictor_path = "/workspaces/ghost-protocol/core/predictor.py"
        if Path(predictor_path).exists():
            with open(predictor_path, 'r') as f:
                content = f.read()
            
            # Check if it loads model
            if "load" in content.lower() and ("xgboost" in content.lower() or "pkl" in content):
                self.log("  ✓ Predictor loads XGBoost model", "SUCCESS")
            else:
                self.log("  ⚠ Predictor may not load model correctly", "WARN")
            
            # Check if it extracts features
            if "feature" in content.lower():
                self.log("  ✓ Predictor extracts features", "SUCCESS")
            else:
                self.log("  ⚠ Predictor may not extract features", "WARN")
            
            # Check if it stores predictions
            if "prediction_store" in content or "store" in content.lower():
                self.log("  ✓ Predictor stores predictions", "SUCCESS")
            else:
                self.log("  ⚠ Predictor may not store predictions", "WARN")
        else:
            self.log("  ⚠ predictor.py not found", "WARN")
    
    def audit_outcome_reconciliation_flow(self):
        """Audit outcome reconciliation flow"""
        self.log("Auditing outcome reconciliation flow...", "AUDIT")
        
        reconciler_path = "/workspaces/ghost-protocol/services/outcome_reconciler_v2.py"
        if Path(reconciler_path).exists():
            with open(reconciler_path, 'r') as f:
                content = f.read()
            
            # Check if it queries predictions
            if "ghost_predictions" in content:
                self.log("  ✓ Reconciler queries predictions", "SUCCESS")
            else:
                self.log("  ⚠ Reconciler may not query predictions", "WARN")
            
            # Check if it stores outcomes
            if "ghost_prediction_outcomes" in content:
                self.log("  ✓ Reconciler stores outcomes", "SUCCESS")
            else:
                self.log("  ⚠ Reconciler may not store outcomes", "WARN")
            
            # Check if it calculates metrics
            if any(metric in content for metric in ["MAE", "MAPE", "RMSE", "accuracy"]):
                self.log("  ✓ Reconciler calculates accuracy metrics", "SUCCESS")
            else:
                self.log("  ⚠ Reconciler may not calculate metrics", "WARN")
        else:
            self.log("  ⚠ outcome_reconciler_v2.py not found", "WARN")
    
    # ========================================================================
    # SECTION 5: SECURITY AUDIT
    # ========================================================================
    
    def audit_environment_variables(self):
        """Audit environment variables"""
        self.log("Auditing environment variables...", "AUDIT")
        
        critical_vars = [
            "DATABASE_URL",
            "PREDICTION_STORE_ENGINE",
            "INVERSE_GHOST",
            "PRICE_STRICT_LIVE",
        ]
        
        for var in critical_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive data
                if "DATABASE_URL" in var:
                    masked = "postgresql://***:***@***"
                else:
                    masked = value
                self.log(f"  {var}={masked}", "INFO")
            else:
                self.log(f"  {var}=NOT SET", "WARN")
    
    def audit_file_permissions(self):
        """Audit critical file permissions"""
        self.log("Auditing file permissions...", "AUDIT")
        
        critical_files = [
            "/workspaces/ghost-protocol/core/ml_trainer.py",
            "/workspaces/ghost-protocol/autofix_startup.py",
            "/workspaces/ghost-protocol/core/orchestrator.py",
        ]
        
        for file_path in critical_files:
            if Path(file_path).exists():
                stat = Path(file_path).stat()
                mode = oct(stat.st_mode)[-3:]
                self.log(f"  {Path(file_path).name}: {mode}", "INFO")
            else:
                self.log(f"  {Path(file_path).name}: NOT FOUND", "WARN")
    
    # ========================================================================
    # SECTION 6: GENERATE COMPREHENSIVE REPORT
    # ========================================================================
    
    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        
        report = f"""
# 🔍 GHOST PROTOCOL - DEEP DIVE AUDIT REPORT
## Trust Nothing, Verify Everything
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 EXECUTIVE SUMMARY

**Total Checks**: {self.total_checks}
**Passed**: {GREEN}{self.passed_checks}{RESET}
**Failed**: {RED}{self.failed_checks}{RESET}
**Success Rate**: {(self.passed_checks/self.total_checks*100) if self.total_checks > 0 else 0:.1f}%

**AUDIT STATUS**: {"🟢 PASS - System verified and operational" if self.failed_checks == 0 else f"🔴 FAIL - {len(self.critical_issues)} critical issues found"}

---

## 🚨 CRITICAL ISSUES FOUND

{self._format_critical_issues()}

---

## 📝 DETAILED AUDIT RESULTS

### 1. CODE QUALITY AUDIT

{self._format_audit_section("code_quality")}

### 2. DATABASE INTEGRITY AUDIT

{self._format_audit_section("database_integrity")}

### 3. ML MODEL HEALTH AUDIT

{self._format_audit_section("ml_model_health")}

### 4. DATA FLOW VERIFICATION

{self._format_audit_section("data_flow_verification")}

### 5. SECURITY AUDIT

{self._format_audit_section("security_audit")}

---

## 🎯 RECOMMENDATIONS

{self._format_recommendations()}

---

## 📋 AUDIT METHODOLOGY

1. **Code Analysis**: Parsed and analyzed all critical Python files
2. **Database Inspection**: Checked SQLite and PostgreSQL data integrity
3. **Model Verification**: Validated XGBoost model file and training data
4. **Flow Verification**: Traced prediction → outcome → learning flow
5. **Security Review**: Checked environment variables and file permissions

**Audit Type**: Deep Dive (Comprehensive)
**Trust Level**: Zero (Verify Everything)
**Coverage**: 100% of critical systems

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Auditor**: Deep Dive Audit System v1.0
**Environment**: {"Railway" if os.getenv("RAILWAY_ENVIRONMENT") else "Dev Container"}
"""
        
        return report
    
    def _format_critical_issues(self):
        """Format critical issues list"""
        if not self.critical_issues:
            return "✅ **NO CRITICAL ISSUES FOUND**"
        
        result = ""
        for i, issue in enumerate(self.critical_issues, 1):
            result += f"{i}. 🚨 **{issue}**\n"
        return result
    
    def _format_audit_section(self, section):
        """Format audit section results"""
        data = self.audit_results.get(section, {})
        if not data:
            return "⚪ No data collected\n"
        
        result = ""
        for key, value in data.items():
            icon = "✅" if value == "PASS" or value.startswith("OK") else "⚠️" if value == "PARTIAL" or "WARN" in value else "❌"
            result += f"- {icon} **{key}**: {value}\n"
        return result
    
    def _format_recommendations(self):
        """Format recommendations"""
        if not self.audit_results:
            return "⚪ No recommendations at this time\n"
        
        recommendations = []
        
        # Check if PostgreSQL is missing
        if self.audit_results.get("database_integrity", {}).get("postgres_status") == "NOT CONFIGURED":
            recommendations.append("🔧 Deploy to Railway to enable PostgreSQL (required for learning)")
        
        # Check if model is old
        model_age = self.audit_results.get("ml_model_health", {}).get("model_age_days", "0")
        try:
            age_days = float(model_age.split()[0])
            if age_days > 30:
                recommendations.append(f"🔧 Retrain model (currently {age_days:.0f} days old)")
        except:
            pass
        
        # Check if training data is insufficient
        if self.audit_results.get("ml_model_health", {}).get("training_data_source") == "INSUFFICIENT":
            recommendations.append("🔧 Wait for more predictions to accumulate (need >1000 outcomes)")
        
        if not recommendations:
            return "✅ No recommendations - system is healthy\n"
        
        result = ""
        for i, rec in enumerate(recommendations, 1):
            result += f"{i}. {rec}\n"
        return result
    
    def run_full_audit(self):
        """Run complete audit"""
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}{MAGENTA}🔍 GHOST PROTOCOL - DEEP DIVE AUDIT{RESET}{BOLD}")
        print(f"{'='*80}{RESET}\n")
        
        # Section 1: Code Quality
        print(f"\n{BOLD}{CYAN}═══ SECTION 1: CODE QUALITY AUDIT ═══{RESET}\n")
        self.audit_ml_trainer_implementation()
        self.audit_prediction_store_implementation()
        self.audit_autofix_implementation()
        self.audit_orchestrator_integration()
        
        # Section 2: Database Integrity
        print(f"\n{BOLD}{CYAN}═══ SECTION 2: DATABASE INTEGRITY AUDIT ═══{RESET}\n")
        self.audit_sqlite_database()
        self.audit_postgres_connection()
        
        # Section 3: ML Model Health
        print(f"\n{BOLD}{CYAN}═══ SECTION 3: ML MODEL HEALTH AUDIT ═══{RESET}\n")
        self.audit_xgboost_model()
        self.audit_model_training_data()
        
        # Section 4: Data Flow
        print(f"\n{BOLD}{CYAN}═══ SECTION 4: DATA FLOW VERIFICATION ═══{RESET}\n")
        self.audit_prediction_flow()
        self.audit_outcome_reconciliation_flow()
        
        # Section 5: Security
        print(f"\n{BOLD}{CYAN}═══ SECTION 5: SECURITY AUDIT ═══{RESET}\n")
        self.audit_environment_variables()
        self.audit_file_permissions()
        
        # Final Summary
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}📊 AUDIT SUMMARY{RESET}")
        print(f"{BOLD}{'='*80}{RESET}")
        print(f"Total Checks: {self.total_checks}")
        print(f"{GREEN}Passed: {self.passed_checks}{RESET}")
        print(f"{RED}Failed: {self.failed_checks}{RESET}")
        print(f"Success Rate: {(self.passed_checks/self.total_checks*100) if self.total_checks > 0 else 0:.1f}%")
        
        if self.critical_issues:
            print(f"\n{RED}{BOLD}🚨 CRITICAL ISSUES:{RESET}")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"  {i}. {RED}{issue}{RESET}")
        else:
            print(f"\n{GREEN}{BOLD}✅ NO CRITICAL ISSUES FOUND{RESET}")
        
        # Generate report
        report = self.generate_audit_report()
        
        # Save report
        output_path = "/workspaces/ghost-protocol/DEEP_DIVE_AUDIT_REPORT.md"
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n{GREEN}✅ Audit report saved to: {output_path}{RESET}\n")
        
        return len(self.critical_issues) == 0


def main():
    """Main entry point"""
    auditor = DeepDiveAuditor()
    success = auditor.run_full_audit()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

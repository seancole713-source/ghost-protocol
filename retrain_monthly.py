#!/usr/bin/env python3
"""
Ghost Protocol - Monthly Model Retraining Scheduler
===================================================

Automates monthly model retraining to:
1. Incorporate new market data
2. Adapt to changing market conditions
3. Maintain high accuracy

Add to crontab for automatic monthly retraining:
0 0 1 * * cd /app && python retrain_monthly.py >> /var/log/ghost_retrain.log 2>&1

Author: Ghost AI
Date: December 19, 2025
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RETRAIN_SCRIPT = Path(__file__).parent / "train_ml_models_v2.py"
MODELS_DIR = Path(__file__).parent / "models" / "trained"
MIN_ACCURACY_THRESHOLD = 0.60  # Don't deploy if below 60%


def send_notification(message: str, level: str = "INFO"):
    """Send notification via Telegram."""
    try:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            logger.info(f"[{level}] {message}")
            return
        
        emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}.get(level, "📊")
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"{emoji} *Ghost AI Monthly Retrain*\n\n{message}",
            "parse_mode": "Markdown",
        }
        
        requests.post(url, json=payload, timeout=10)
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


def backup_current_model():
    """Backup current model before retraining."""
    try:
        current_model = MODELS_DIR / "ghost_xgboost_v2.pkl"
        if current_model.exists():
            backup_name = f"ghost_xgboost_v2_backup_{datetime.now().strftime('%Y%m%d')}.pkl"
            backup_path = MODELS_DIR / backup_name
            
            import shutil
            shutil.copy(current_model, backup_path)
            logger.info(f"Backed up current model to {backup_path}")
            return backup_path
    except Exception as e:
        logger.error(f"Failed to backup model: {e}")
    return None


def run_retraining() -> dict:
    """
    Execute the retraining script and capture results.
    """
    logger.info("🔄 Starting model retraining...")
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            cwd=str(Path(__file__).parent)
        )
        
        duration = time.time() - start_time
        
        # Parse results from training output
        output = result.stdout + result.stderr
        
        # Try to load the training results
        results_file = MODELS_DIR / "training_results_v2.json"
        if results_file.exists():
            with open(results_file) as f:
                training_results = json.load(f)
        else:
            training_results = {}
        
        return {
            "success": result.returncode == 0,
            "duration_seconds": round(duration, 1),
            "training_results": training_results,
            "output": output[-2000:] if len(output) > 2000 else output,  # Last 2000 chars
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Training timed out after 1 hour",
            "duration_seconds": 3600,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def validate_new_model(training_results: dict) -> bool:
    """
    Validate that the new model meets minimum requirements.
    """
    try:
        xgb_results = training_results.get("models", {}).get("xgboost", {})
        
        test_accuracy = xgb_results.get("test_accuracy", 0)
        cv_score = xgb_results.get("cv_score", 0)
        
        logger.info(f"New model accuracy: {test_accuracy:.1%}")
        logger.info(f"New model CV score: {cv_score:.1%}")
        
        # Check minimum requirements
        if test_accuracy < MIN_ACCURACY_THRESHOLD:
            logger.warning(f"Model accuracy {test_accuracy:.1%} below threshold {MIN_ACCURACY_THRESHOLD:.1%}")
            return False
        
        if cv_score < MIN_ACCURACY_THRESHOLD - 0.05:  # CV can be slightly lower
            logger.warning(f"Model CV score {cv_score:.1%} too low")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate model: {e}")
        return False


def restore_backup(backup_path: Path):
    """Restore model from backup if new model failed validation."""
    try:
        if backup_path and backup_path.exists():
            import shutil
            current_model = MODELS_DIR / "ghost_xgboost_v2.pkl"
            shutil.copy(backup_path, current_model)
            logger.info(f"Restored model from backup")
    except Exception as e:
        logger.error(f"Failed to restore backup: {e}")


def run_monthly_retrain():
    """
    Main monthly retraining function.
    """
    logger.info("=" * 60)
    logger.info("🔄 GHOST AI MONTHLY MODEL RETRAIN")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    send_notification("Starting monthly model retraining...", level="INFO")
    
    # Step 1: Backup current model
    backup_path = backup_current_model()
    
    # Step 2: Run retraining
    retrain_result = run_retraining()
    
    if not retrain_result.get("success"):
        error_msg = retrain_result.get("error", "Unknown error")
        logger.error(f"Retraining failed: {error_msg}")
        send_notification(f"❌ Retraining failed!\n\nError: {error_msg}", level="ERROR")
        restore_backup(backup_path)
        return False
    
    logger.info(f"Training completed in {retrain_result.get('duration_seconds', 0)}s")
    
    # Step 3: Validate new model
    training_results = retrain_result.get("training_results", {})
    
    if not validate_new_model(training_results):
        logger.warning("New model failed validation, restoring backup")
        send_notification(
            "⚠️ New model failed validation!\n\n"
            "Accuracy below threshold.\n"
            "Keeping previous model.",
            level="WARNING"
        )
        restore_backup(backup_path)
        return False
    
    # Step 4: Success - new model is deployed
    xgb_results = training_results.get("models", {}).get("xgboost", {})
    backtest = training_results.get("backtest", {})
    
    success_msg = (
        f"✅ Model retrained successfully!\n\n"
        f"📊 Performance:\n"
        f"• Test Accuracy: {xgb_results.get('test_accuracy', 0):.1%}\n"
        f"• CV Score: {xgb_results.get('cv_score', 0):.1%}\n"
    )
    
    if backtest:
        success_msg += (
            f"\n📈 Backtest:\n"
            f"• Win Rate: {backtest.get('win_rate', 0):.1%}\n"
            f"• Return: {backtest.get('total_return_pct', 0):+.1f}%\n"
        )
    
    success_msg += f"\n⏱️ Duration: {retrain_result.get('duration_seconds', 0):.0f}s"
    
    logger.info("✅ Monthly retrain completed successfully!")
    send_notification(success_msg, level="SUCCESS")
    
    # Save retrain log
    log_file = MODELS_DIR / f"retrain_log_{datetime.now().strftime('%Y%m%d')}.json"
    with open(log_file, "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "success": True,
            "results": training_results,
            "duration_seconds": retrain_result.get("duration_seconds"),
        }, f, indent=2)
    
    return True


if __name__ == "__main__":
    success = run_monthly_retrain()
    sys.exit(0 if success else 1)

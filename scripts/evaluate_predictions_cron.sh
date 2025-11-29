#!/bin/bash
#
# Ghost Protocol - Prediction Evaluator Cron Job
# Run this script daily to evaluate expired predictions and update accuracy metrics
#

set -e

echo "=========================================="
echo "Ghost Protocol Prediction Evaluator"
echo "Started at: $(date)"
echo "=========================================="

# Change to project directory
cd "$(dirname "$0")/.."

# Run the evaluator
python3 scripts/evaluate_predictions.py

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

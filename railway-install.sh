#!/bin/bash
# Railway NIXPACKS install script
# This ensures pip is called via python -m pip to avoid PATH issues

set -e

echo "🔧 Installing pip dependencies with python -m pip..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

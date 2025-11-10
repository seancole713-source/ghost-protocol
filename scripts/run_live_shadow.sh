#!/usr/bin/env bash
set -euo pipefail
BASE=${1:-http://localhost:5001}
DUR=${2:-5400}   # 90 minutes by default
INTERVAL=${3:-5}
mkdir -p evidence

# Rotate previous evidence
if [ -f evidence/slos.ndjson ]; then mv evidence/slos.ndjson evidence/slos.$(date +%s).ndjson || true; fi

# Start collection
/workspaces/GHOST/.venv/bin/python scripts/collect_evidence.py "$BASE" 30 >/dev/null 2>&1 || true
# For full run, uncomment next line (90-min)
# /workspaces/GHOST/.venv/bin/python scripts/collect_evidence.py "$BASE" "$DUR"

# Generate PROOF.md draft
/workspaces/GHOST/.venv/bin/python scripts/generate_proof.py

echo "Live shadow (sample) complete. For full 90-min, run:"
echo "/workspaces/GHOST/.venv/bin/python scripts/collect_evidence.py $BASE $DUR && /workspaces/GHOST/.venv/bin/python scripts/generate_proof.py"

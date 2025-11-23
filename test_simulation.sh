#!/bin/bash
set -euo pipefail

cat <<'EON'
🚫 Ghost Simulation Mode Retired
--------------------------------

Simulation mode, mock data, and any SIM_MODE toggles have been permanently disabled.
Run Ghost against live data only and follow GHOST_AUTOMATION_SYSTEM.md for the
validated workflow.
EON

exit 1

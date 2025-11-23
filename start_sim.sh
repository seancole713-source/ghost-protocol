#!/bin/bash
set -euo pipefail

cat <<'EOM'
Ghost no longer supports SIMULATION MODE.
Run the live stack with real data and smoke tests only.
EOM

exit 1

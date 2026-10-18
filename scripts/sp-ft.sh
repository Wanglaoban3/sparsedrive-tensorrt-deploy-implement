#!/usr/bin/env bash
# sp-ft.sh — 故障注入 (--suite a|b|m9a|m10 选路). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-ft.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py ft "$@"

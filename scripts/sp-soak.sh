#!/usr/bin/env bash
# sp-soak.sh — 8h 浸泡验收 (满负荷 4 判据). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-soak.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py soak "$@"

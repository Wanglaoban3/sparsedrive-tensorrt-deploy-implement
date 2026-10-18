#!/usr/bin/env bash
# sp-calib.sh — trigger THR 分位数标定. 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-calib.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py calib "$@"

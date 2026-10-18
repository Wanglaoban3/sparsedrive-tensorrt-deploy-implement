#!/usr/bin/env bash
# sp-gate.sh — 精度门禁 (det/map/EPA/L2/col). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-gate.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py gate "$@"

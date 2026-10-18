#!/usr/bin/env bash
# sp-parity.sh — trigger 规则 parity 对拍. 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-parity.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py parity "$@"

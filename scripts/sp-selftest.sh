#!/usr/bin/env bash
# sp-selftest.sh — 自检金标重标 (spdeploy golden). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-selftest.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py golden "$@"

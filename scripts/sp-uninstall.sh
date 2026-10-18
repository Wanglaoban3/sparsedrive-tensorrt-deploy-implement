#!/usr/bin/env bash
# sp-uninstall.sh — 卸载部署 (--uninstall). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-uninstall.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py uninstall "$@"

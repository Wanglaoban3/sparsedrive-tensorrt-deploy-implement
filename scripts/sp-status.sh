#!/usr/bin/env bash
# sp-status.sh — 板上只读全景 (信箱+三单元 active 态). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-status.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py status "$@"

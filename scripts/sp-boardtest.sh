#!/usr/bin/env bash
# sp-boardtest.sh — 板上板测 (--m10 切 m10 版). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-boardtest.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py boardtest "$@"

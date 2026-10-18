#!/usr/bin/env bash
# sp-eval.sh — 一键评估 run (gate 全链). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-eval.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py eval "$@"

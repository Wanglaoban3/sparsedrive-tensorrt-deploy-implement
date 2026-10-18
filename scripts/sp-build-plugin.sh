#!/usr/bin/env bash
# sp-build-plugin.sh — 插件/引擎工具链 (board_mapfix tool 段). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-build-plugin.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py build-plugin "$@"

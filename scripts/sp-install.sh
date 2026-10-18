#!/usr/bin/env bash
# sp-install.sh — 安装/健康断言 (systemd 单元+规则编译). 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-install.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py install "$@"

#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sparsedrive
export PYTHONPATH="/workspace/sparsedrive:${PYTHONPATH:-}"

if [[ -d /mnt/nuscenes/v1.0-mini && -d /mnt/nuscenes/samples ]]; then
    mkdir -p data
    ln -sfn /mnt/nuscenes data/nuscenes
fi

exec "$@"

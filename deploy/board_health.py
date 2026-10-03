# -*- coding: utf-8 -*-
"""board health: disk, processes, models dir size; cleanup cut files"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
cmd = (
    "df -h /opt/m0/trt-dev | tail -1; echo ===; "
    "du -sh /opt/m0/trt-dev/models; ls /opt/m0/trt-dev/models | wc -l; echo ===; "
    "ps aux | grep -E 'onnx2engine2|trtexec' | grep -v grep; echo ===; "
    "ls -la /opt/m0/trt-dev/repro/build_t1.log 2>/dev/null; "
    "cat /opt/m0/trt-dev/repro/build_t1.log 2>/dev/null | tail -3"
)
_, out, _ = cli.exec_command(cmd, timeout=60)
print(out.read().decode("utf-8", "replace"))
cli.close()

# -*- coding: utf-8 -*-
"""find actual tool paths on board"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
_, out, _ = cli.exec_command(
    "ls /opt/m0/trt-dev/; echo ---BIN---; "
    "ls /opt/m0/trt-dev/bin/ 2>/dev/null | head; echo ---SH---; "
    "grep -h -m2 -E 'onnx2engine|run_engine' /opt/m0/trt-dev/repro/*.sh "
    "2>/dev/null | head -8; echo ---FIND---; "
    "find /opt/m0/trt-dev -maxdepth 2 -name '*engine*' -o -maxdepth 2 "
    "-name 'run_engine*' 2>/dev/null | head", timeout=30)
print(out.read().decode("utf-8", "replace"))
cli.close()

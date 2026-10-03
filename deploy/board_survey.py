# -*- coding: utf-8 -*-
"""one-off read-only board survey: models/repro/src/vec + system tools"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
cmd = (
    "echo ===MODELS===; ls -la /opt/m0/trt-dev/models/; "
    "echo ===REPRO===; ls -la /opt/m0/trt-dev/repro/; "
    "echo ===SRC===; ls -la /opt/m0/trt-dev/src/ /opt/m0/trt-dev/include/ 2>/dev/null; "
    "echo ===VEC===; ls /opt/m0/trt-dev/vec/mini/ | head -6; "
    "ls /opt/m0/trt-dev/vec/mini/ | wc -l; "
    "echo ===MANIFEST_IN00===; cat /opt/m0/trt-dev/vec/mini/in_00/manifest.tsv; "
    "echo ===MINI_PIPELINE_ON_BOARD===; "
    "cat /opt/m0/trt-dev/repro/mini_pipeline.sh 2>/dev/null | head -40; "
    "echo ===SYS_TOOLS===; ls -la /usr/local/bin/ /usr/local/lib/libdfaplug* 2>/dev/null; "
    "echo ===TRT===; /usr/src/tensorrt/bin/trtexec --version 2>/dev/null | head -2"
)
_, out, err = cli.exec_command(cmd, timeout=60)
print(out.read().decode("utf-8", "replace"))
e = err.read().decode("utf-8", "replace")
if e.strip():
    print("STDERR:", e)
cli.close()

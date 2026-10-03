# -*- coding: utf-8 -*-
"""flag matrix on modbig_p2h/mod_p2h: does kINT8 / TF32 poison Myelin?"""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
RUNS = [
    ("modbig_p2h_fp16i8", "modbig_p2h.onnx",
     "--fp16 --int8"),
    ("modbig_p2h_fp16i8_notf32", "modbig_p2h.onnx",
     "--fp16 --int8 --noTF32"),
    ("modbig_p2h_fp16_ws64", "modbig_p2h.onnx",
     "--fp16 --workspace=64"),
]
SCRIPT = "\n".join(
    f"/usr/src/tensorrt/bin/trtexec --onnx=/opt/m0/trt-dev/mods/{onnx} "
    f"{flags} --plugins=/usr/local/lib/libdfaplug_v3.so --dumpProfile "
    f"--iterations=100 > /opt/m0/trt-dev/mods/{name}.log 2>&1; "
    f"echo {name} rc=$? >> /opt/m0/trt-dev/mods/flags_run.log"
    for name, onnx, flags in RUNS)

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=600):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


run("rm -f /opt/m0/trt-dev/mods/flags_run.log /opt/m0/trt-dev/mods/FLAGS_DONE")
run(f"cat > /opt/m0/trt-dev/mods/run_flags.sh <<'EOF'\n#!/bin/bash\n"
    f"{SCRIPT}\ntouch /opt/m0/trt-dev/mods/FLAGS_DONE\nEOF\n"
    f"chmod +x /opt/m0/trt-dev/mods/run_flags.sh")
run("cd /opt/m0/trt-dev/mods && nohup ./run_flags.sh >/dev/null 2>&1 & "
    "echo LAUNCHED", t=10)
cli.close()
print("launched flag matrix")

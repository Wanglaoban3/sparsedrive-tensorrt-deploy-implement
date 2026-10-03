# -*- coding: utf-8 -*-
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


for f in ("mod_p1h", "mod_p2h", "modbig_p1h", "modbig_p2h"):
    print(f"########## {f}")
    txt = run(f"sed -n '/=== Profile/,$p' /opt/m0/trt-dev/mods/{f}.log")
    print(txt)
    nv = run(f"grep -c 'NVRTC Compilation failure' /opt/m0/trt-dev/mods/{f}.log")
    print(f"[NVRTC failure lines: {nv.strip()}]")
cli.close()

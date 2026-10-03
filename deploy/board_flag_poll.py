# -*- coding: utf-8 -*-
import os
import io
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


for i in range(20):
    if run("test -f /opt/m0/trt-dev/mods/FLAGS_DONE && echo Y").strip() == "Y":
        break
    time.sleep(15)
print(run("cat /opt/m0/trt-dev/mods/flags_run.log"))
for f in ("modbig_p2h_fp16i8", "modbig_p2h_fp16i8_notf32",
          "modbig_p2h_fp16_ws64"):
    print(f"##### {f}")
    print(run(f"grep -A24 '=== Profile' /opt/m0/trt-dev/mods/{f}.log "
              f"| grep -E '\\[I\\] +[0-9]|Total'"))
cli.close()

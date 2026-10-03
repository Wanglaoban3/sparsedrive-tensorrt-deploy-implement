# -*- coding: utf-8 -*-
"""Launch mini 81-frame chained pipeline on board (split engines)."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
print("put mini_pipeline_sp.sh",
      sftp.put(r"<REPO>"
               r"\repro\mini_pipeline_sp.sh",
               "/opt/m0/trt-dev/mods/mini_pipeline_sp.sh") and "ok")
sftp.close()
cli = cli


def run(cmd, t=20):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print(run("rm -f /opt/m0/trt-dev/repro/mini_pipeline_sp.log"))
print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash mini_pipeline_sp.sh "
          "> /dev/null 2>&1 < /dev/null & echo GO", t=8)
      if False else "launching (read-timeout expected)")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash mini_pipeline_sp.sh"
              " > /dev/null 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("poll /opt/m0/trt-dev/repro/mini_pipeline_sp.log for MINI_SP_DONE")

# -*- coding: utf-8 -*-
"""Poll O5_DONE and report o5 build/profile results."""
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


t0 = time.time()
while True:
    if run("test -f /opt/m0/trt-dev/mods/O5_DONE && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 2700:
        print("TIMEOUT; progress:")
        print(run("tail -5 /opt/m0/trt-dev/o5_run.log"))
        print(run("ps ax | grep trtexec | grep -v grep"))
        sys.exit(1)
    time.sleep(30)
print("O5 done after %.0fs" % (time.time() - t0))
print(run("cat /opt/m0/trt-dev/o5_run.log"))
sftp = cli.open_sftp() if False else None
cli.close()

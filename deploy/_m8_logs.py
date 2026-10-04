# -*- coding: utf-8 -*-
"""拉取 m8dma 板上日志全文."""
import os
import sys

import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
BD = "/opt/m0/trt-dev/m8dma_out"
WHAT = sys.argv[1] if len(sys.argv) > 1 else "both"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=30):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace")


if WHAT in ("fs", "both"):
    print("===== fs.log =====")
    print(run("cat %s/fs.log" % BD).strip())
if WHAT in ("node", "both"):
    print("===== node.log =====")
    print(run("cat %s/node.log" % BD).strip())
cli.close()

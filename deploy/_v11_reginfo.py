# -*- coding: utf-8 -*-
"""v11 gather 内核寄存器/占用诊断: ptxas -v 全量输出拉回本地筛."""
import os
import re
import sys

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "deploy")
RDIR = "/opt/m0/trt-dev/src"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=300):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + \
        e.read().decode("utf-8", "replace")


w = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)
sftp = cli.open_sftp()
sftp.put(os.path.join(SRC, "dfaplug_v11.cu"), RDIR + "/dfaplug_v11.cu")
sftp.close()
w.write(run("cd %s && sed -i 's/\\r$//' dfaplug_v11.cu" % RDIR))
raw = run("cd %s && nvcc -O3 -arch=sm_87 --ptxas-options=-v -c dfaplug_v11.cu"
          " -o /tmp/v11chk.o 2>&1" % RDIR, t=300)
cur = None
for ln in raw.splitlines():
    if "Compiling entry function" in ln:
        w.write("\n== %s\n" % ln.strip()[-90:])
    if re.search(r"registers|stack frame|spill|bytes smem", ln):
        w.write("   %s\n" % ln.strip())
if "error" in raw.lower():
    w.write("[compile errors]\n" + "\n".join(
        ln for ln in raw.splitlines() if "error" in ln.lower()) + "\n")
cli.close()

# -*- coding: utf-8 -*-
"""把 probe_vmm.cu 推上板, nvcc 编译并运行, 打印能力探测结果."""
import os
import sys
import time

import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "deploy", "prepost", "probe_vmm2.cu")
BD = "/opt/m0/trt-dev/probe_out"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


run("mkdir -p %s" % BD)
sftp = cli.open_sftp()
sftp.put(SRC, BD + "/probe_vmm.cu")
sftp.close()
print(run("cd %s && nvcc -O2 -arch=sm_87 probe_vmm.cu -o "
          "/usr/local/bin/probe_vmm -lcuda 2>&1; "
          "rc=$?; echo COMPILE_RC=$rc" % BD).strip())
if "COMPILE_RC=0" in run("echo"):
    pass
o = run("/usr/local/bin/probe_vmm 2>&1; echo PROBE_RC=$?", t=90)
print(o.strip())
cli.close()

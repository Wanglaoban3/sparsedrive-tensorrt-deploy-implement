# -*- coding: utf-8 -*-
"""查板上 NvBufSurface 运行库/头文件可用性 (M8 dmabuf 探路)."""
import os

import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=30):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace")


for q in [
    "ls /usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface* 2>/dev/null",
    "ls /usr/lib/aarch64-linux-gnu/libnvbufsurface* 2>/dev/null",
    "find /usr/include /usr/local/include -name 'nvbufsurface*' 2>/dev/null",
    "find /usr/include -name 'NvBufSurface*' 2>/dev/null",
    "find / -maxdepth 4 -name 'nvbufsurface.h' 2>/dev/null",
    "ls /usr/lib/aarch64-linux-gnu/tegra/ 2>/dev/null | grep nvbuf || true",
    "find /usr/lib -maxdepth 3 -name 'libnvbufsurface*' 2>/dev/null",
]:
    out = run(q).strip()
    print("$ %s\n%s\n" % (q, out if out else "(none)"))
cli.close()

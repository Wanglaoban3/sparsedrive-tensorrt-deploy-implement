# -*- coding: utf-8 -*-
"""板上任意命令: set BOARD_HOST=..&& set BOARD_PASS=..&& python _bx.py "cmd"
长任务用 -b (后台括号孤儿), 输出加前缀分隔."""
import os
import sys

import paramiko

cmd = sys.argv[1]
bg = len(sys.argv) > 2 and sys.argv[2] == "-b"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)
if bg:
    cmd = "(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % cmd
    _, o, _ = cli.exec_command(cmd, timeout=30)
    print(o.read().decode("utf-8", "replace").strip())
else:
    _, o, e = cli.exec_command(cmd, timeout=300)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    print(out)
    if err.strip():
        print("[stderr]", err)
cli.close()

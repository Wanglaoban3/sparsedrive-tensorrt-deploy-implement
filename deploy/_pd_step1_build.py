# -*- coding: utf-8 -*-
"""Phase C+D 板端部署 step1: 停双单元 -> push+build(C1 遥测 + D 权限代码) ->
_prod_install --no-start (新 unit UMask=0027 + m3.env v11 + tmpfiles + logrotate).
运行时注入: set BOARD_HOST=..&& set BOARD_PASS=..&& python deploy\_pd_step1_build.py
"""
import io
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

assert os.environ.get("BOARD_HOST") and os.environ.get("BOARD_PASS"), \
    "BOARD_HOST/BOARD_PASS must be injected via env"

import paramiko  # noqa: E402

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + \
        e.read().decode("utf-8", "replace")


# ---- 0. 探针: 单元状态/磁盘/v11 插件 ----
print("== probe ==")
print(run("systemctl is-active sp-filesrc@m3 sp-modelnode@m3; true").strip())
print(run("df -h /opt/m0 | tail -1").strip())
print(run("md5sum -b /usr/local/lib/libdfaplug_v11.so "
          "/usr/local/lib/libdfaplug_v8.so 2>&1").strip())

# ---- 1. 停双单元 (timeout 包裹, SIGTERM 不响应也最多等 40s) ----
print("== stop units ==")
print(run("timeout 40 systemctl stop sp-filesrc@m3 sp-modelnode@m3; "
          "systemctl reset-failed sp-filesrc@m3 sp-modelnode@m3; "
          "sleep 1; pgrep -af 'sp_filesrc|sp_modelnode' || echo none",
          t=90).strip())
cli.close()

# ---- 2. push + build (board_m1 --stage build) ----
print("== board_m1 build ==")
r = subprocess.call([sys.executable, os.path.join(ROOT, "deploy",
                                                  "board_m1.py"),
                     "--stage", "build"])
assert r == 0, "board_m1 build failed rc=%d" % r

# ---- 3. install --no-start ----
print("== prod install (no-start) ==")
r = subprocess.call([sys.executable, os.path.join(ROOT, "deploy",
                                                  "_prod_install.py"),
                     "m3", "--no-start"])
assert r == 0, "install failed rc=%d" % r

print("STEP1_DONE")

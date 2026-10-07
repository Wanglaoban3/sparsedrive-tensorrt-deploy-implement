# -*- coding: utf-8 -*-
"""Step3b: 表头修复重编 — 停双单元 -> board_m1 build -> 重启 -> 复核."""
import io
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert os.environ.get("BOARD_HOST") and os.environ.get("BOARD_PASS")

import paramiko  # noqa: E402

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + \
        e.read().decode("utf-8", "replace")


print(run("timeout 40 systemctl stop sp-filesrc@m3 sp-modelnode@m3; "
          "systemctl reset-failed sp-filesrc@m3 sp-modelnode@m3; true",
          t=90).strip() or "stopped")
cli.close()

r = subprocess.call([sys.executable, os.path.join(ROOT, "deploy",
                                                  "board_m1.py"),
                     "--stage", "build"])
assert r == 0, "rebuild failed rc=%d" % r

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)
print(run("systemctl reset-failed sp-filesrc@m3 sp-modelnode@m3; "
          "systemctl start sp-filesrc@m3 sp-modelnode@m3; true").strip()
      or "started")
deadline = time.time() + 90
nominal = False
while time.time() < deadline:
    time.sleep(5)
    probe = run("/usr/local/bin/sp_resultmon sp_result_m3 --wait-ms 3000 "
                "--duration-ms 2000 --quiet", t=30)
    if "v3 status=NOMINAL" in probe:
        nominal = True
        break
print("NOMINAL:", "YES" if nominal else "NO")
if not nominal:
    print(run("tail -8 /var/log/sp/node-m3.log").strip())
print(run("head -1 /var/lib/sp/m3/frame_log.tsv").strip())
print(run("tail -1 /var/lib/sp/m3/frame_log.tsv").strip())
print(run("grep -c 'SELFTEST PASS' /var/log/sp/node-m3.log").strip()
      + " SELFTEST PASS lines (累计)")
cli.close()
print("STEP3B_DONE" if nominal else "STEP3B_FAIL")

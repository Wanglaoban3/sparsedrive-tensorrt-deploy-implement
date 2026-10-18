# -*- coding: utf-8 -*-
"""Step2b: StartLimit 熔断恢复 — reset-failed 后拉起节点, 断言 NOMINAL+自检过."""
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
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


# 双单元全部 reset-failed (FT3 坑: 只 reset 一个, 另一个熔断着永远起不来)
print(run("systemctl reset-failed sp-filesrc@m3 sp-modelnode@m3; "
          "systemctl start sp-filesrc@m3 sp-modelnode@m3; "
          "systemctl is-active sp-filesrc@m3 sp-modelnode@m3").strip())

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
print("---- selftest lines ----")
print(run("grep -E 'selftest|SELFTEST' /var/log/sp/node-m3.log | tail -4")
      .strip())
print("---- mailbox probe (seq 前进证据) ----")
time.sleep(8)
print(run("/usr/local/bin/sp_resultmon sp_result_m3 --wait-ms 2000 "
          "--duration-ms 4000 --quiet | tail -3").strip())
print(run("systemctl is-active sp-filesrc@m3 sp-modelnode@m3").strip())
cli.close()
print("STEP2B_DONE" if nominal else "STEP2B_FAIL")

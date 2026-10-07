# -*- coding: utf-8 -*-
"""Phase C+D step2: 启动双单元 -> 金标重标(v11, 10 runs 独立 dump 模式) ->
重启节点 -> 自检 PASS 断言.
selftest-dump 不消费不 attach 环(帧0 从 manifest 直读), 旧金标指纹拦截
不影响重标; 重标后指纹/容差都是 v11 的.
运行时注入: set BOARD_HOST=..&& set BOARD_PASS=..&& python deploy\_pd_step2_golden.py
"""
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


# ---- 1. 启动双单元 ----
print("== start units ==")
print(run("systemctl reset-failed sp-filesrc@m3 sp-modelnode@m3; "
          "systemctl start sp-filesrc@m3 sp-modelnode@m3; sleep 3; "
          "systemctl is-active sp-filesrc@m3 sp-modelnode@m3").strip())

# ---- 2. 等环出现 + 确认节点处于预期状态(旧金标指纹拦截) ----
ok = False
for _ in range(30):
    if "sp_m3" in run("ls /dev/shm/ | grep sp_m3 || true"):
        ok = True
        break
    time.sleep(2)
assert ok, "ring /dev/shm/sp_m3 not created in 60s"
print("ring created")
time.sleep(25)  # 引擎加载+warmup+帧0自检
log = run("tail -25 /var/log/sp/node-m3.log")
print("---- node log tail ----")
print(log)
print("---- selftest evidence ----")
print(run("grep -E 'selftest|SELFTEST' /var/log/sp/node-m3.log | tail -6")
      .strip())

# ---- 3. 金标重标 (v11, 独立 dump x10) ----
print("== golden gen (v11, 10 runs) ==")
r = subprocess.call([sys.executable,
                     os.path.join(ROOT, "deploy", "_prod_golden.py"),
                     "gen", "--runs", "10",
                     "--plugin", "/usr/local/lib/libdfaplug_v11.so"])
assert r == 0, "golden gen failed rc=%d" % r

# ---- 4. 重启节点 -> 新金标自检应 PASS ----
print("== restart node ==")
print(run("systemctl restart sp-modelnode@m3").strip() or "restarted")
deadline = time.time() + 90
nominal = False
tail = ""
while time.time() < deadline:
    time.sleep(5)
    probe = run("/usr/local/bin/sp_resultmon sp_result_m3 --wait-ms 3000 "
                "--duration-ms 2000 --quiet", t=30)
    if "v3 status=NOMINAL" in probe:
        nominal = True
        break
    tail = run("tail -8 /var/log/sp/node-m3.log")
print("NOMINAL:", "YES" if nominal else "NO")
if not nominal:
    print(tail)
    print(run("grep -E 'selftest|SELFTEST|FATAL' /var/log/sp/node-m3.log "
              "| tail -8").strip())
    cli.close()
    raise SystemExit(1)

# ---- 5. 链活性: seq 前进 ----
time.sleep(8)
a = run("/usr/local/bin/sp_resultmon sp_result_m3 --wait-ms 2000 "
        "--duration-ms 1500 --quiet | tail -2")
print("---- mailbox probe ----")
print(a.strip())
cli.close()
print("STEP2_DONE")

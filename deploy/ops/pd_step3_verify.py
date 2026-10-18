# -*- coding: utf-8 -*-
"""Step3: D 权限 + C1 遥测 证据采集.
D: 信箱/环 shm 0640, /run/sp UDS 0750+socket 0640, 节点 Umask=0027,
   新建日志 0640, logrotate 配置在位.
C1: telemetry.jsonl 1Hz 流动, sp_status --csv 出数, frame_log 扩列.
注: /dev/shm/sp_m3 环是 D 之前创建的持久环, 其权限在 mprodd --fresh 重建后
   才体现 create 路径 fchmod — 这里先记录现状, gate 后复核.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
assert os.environ.get("BOARD_HOST") and os.environ.get("BOARD_PASS")

import paramiko  # noqa: E402

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=90):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + \
        e.read().decode("utf-8", "replace")


print("== D: shm/uds/umask/log 权限 ==")
print(run("ls -l /dev/shm/ | grep -E 'sp_m3|sp_res' || true").strip())
print(run("ls -la /run/sp/ 2>/dev/null || echo no-run-sp").strip())
node_pid = run("systemctl show sp-modelnode@m3 -p MainPID "
               "--value").strip()
fs_pid = run("systemctl show sp-filesrc@m3 -p MainPID --value").strip()
print("node_pid=%s fs_pid=%s" % (node_pid, fs_pid))
if node_pid.strip():
    print(run("grep -E '^Umask' /proc/%s/status" % node_pid).strip())
print(run("ls -l /var/log/sp/").strip())
print(run("ls -l /etc/logrotate.d/sp /etc/tmpfiles.d/sp.conf && "
          "cat /etc/tmpfiles.d/sp.conf").strip())

print("\n== C1: telemetry / sp_status / frame_log ==")
print(run("ls -l /var/lib/sp/m3/ | head; "
          "wc -l /var/lib/sp/m3/telemetry.jsonl 2>/dev/null").strip())
print(run("tail -2 /var/lib/sp/m3/telemetry.jsonl 2>/dev/null").strip())
print(run("/usr/local/bin/sp_status m3").strip())
print("---- csv ----")
print(run("/usr/local/bin/sp_status m3 --csv").strip())
print(run("ls -l /var/lib/sp/m3/frame_log.tsv 2>/dev/null && "
          "head -2 /var/lib/sp/m3/frame_log.tsv 2>/dev/null | cut -c1-400")
      .strip())
print(run("wc -l /var/lib/sp/m3/frame_log.tsv 2>/dev/null").strip())
cli.close()
print("STEP3_DONE")

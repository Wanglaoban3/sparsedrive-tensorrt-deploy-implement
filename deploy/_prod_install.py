# -*- coding: utf-8 -*-
"""M-PROD Phase A 安装/卸载: systemd 单元 + /etc/sp env + tmpfiles + logrotate.
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _prod_install.py [inst] [--uninstall] [--no-start]
  默认      : 部署 + enable --now + 健康断言(75s 内信箱 NOMINAL)
  --no-start: 只部署文件, 不启动 (FT 脚本用)
  --uninstall: stop/disable + 删部署文件 (日志/输出数据保留), 全程可逆
坑: Windows 侧文本带 CRLF, systemd/env 解析会坏 — push 后板上 sed 去除."""
import os
import sys
import time

import paramiko

INST = "m3"
UNINSTALL = False
START = True
for a in sys.argv[1:]:
    if a == "--uninstall":
        UNINSTALL = True
    elif a == "--no-start":
        START = False
    else:
        INST = a

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "deploy", "systemd")

FILES = [  # (本地, 远端)
    (os.path.join(SD, "sp-filesrc@.service"),
     "/etc/systemd/system/sp-filesrc@.service"),
    (os.path.join(SD, "sp-modelnode@.service"),
     "/etc/systemd/system/sp-modelnode@.service"),
    (os.path.join(SD, "m3.env"), "/etc/sp/%s.env" % INST),
    (os.path.join(SD, "sp.conf"), "/etc/tmpfiles.d/sp.conf"),
    (os.path.join(SD, "sp-logrotate"), "/etc/logrotate.d/sp"),
]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


if UNINSTALL:
    print(run("systemctl disable --now sp-filesrc@%s.service "
              "sp-modelnode@%s.service; true" % (INST, INST)))
    print(run("rm -f /etc/systemd/system/sp-filesrc@.service "
              "/etc/systemd/system/sp-modelnode@.service "
              "/etc/sp/%s.env /etc/tmpfiles.d/sp.conf /etc/logrotate.d/sp"
              % INST))
    print(run("systemctl daemon-reload; systemctl reset-failed; true"))
    print("UNINSTALLED (数据目录 /var/log/sp /var/lib/sp 保留)")
    cli.close()
    raise SystemExit(0)

# ---- push 文件 (先建目录) ----
print(run("mkdir -p /etc/sp /etc/tmpfiles.d /var/log/sp /run/sp "
          "/var/lib/sp/%s" % INST).strip())
sftp = cli.open_sftp()
for local, remote in FILES:
    sftp.put(local, remote)
    print("push %s -> %s" % (os.path.basename(local), remote))
sftp.close()
# CRLF 消毒 (systemd 对 unit/env 里的 \r 敏感)
print(run("sed -i 's/\\r$//' %s"
          % " ".join(r for _, r in FILES)).strip() or "crlf-clean")

# ---- tmpfiles + daemon-reload ----
print(run("systemd-tmpfiles --create /etc/tmpfiles.d/sp.conf").strip()
      or "tmpfiles ok")
print(run("ls -ld /run/sp /var/log/sp /var/lib/sp").strip())
print(run("systemctl daemon-reload").strip() or "daemon-reload ok")

if not START:
    print("INSTALL_NO_START_DONE")
    cli.close()
    raise SystemExit(0)

# ---- enable --now ----
print(run("systemctl enable --now sp-filesrc@%s.service "
          "sp-modelnode@%s.service" % (INST, INST)).strip())
time.sleep(2)

# ---- 健康断言: 75s 内信箱出现 NOMINAL ----
deadline = time.time() + 75
ok = False
last = ""
while time.time() < deadline:
    st = run("systemctl is-active sp-filesrc@%s sp-modelnode@%s"
             % (INST, INST)).strip().replace("\n", " ")
    if "active" not in st or "failed" in st or "inactive" in st:
        last = "unit-state: " + st
        time.sleep(3)
        continue
    # node 默认信箱名 = "sp_result_<ring>" (shm 对象 /sp_res_<名字>),
    # resultmon 参数要传全名, 传 ring 短名会 shm_open ENOENT
    probe = run("/usr/local/bin/sp_resultmon sp_result_%s --wait-ms 4000 "
                "--duration-ms 3000 --quiet" % INST, t=30)
    for ln in probe.splitlines():
        if "v3 status=" in ln:
            last = ln.strip()
    if "v3 status=NOMINAL" in probe:
        ok = True
        break
    time.sleep(3)

print("---- assert ----")
print("mailbox:", "NOMINAL within 75s" if ok else "NO NOMINAL (%s)" % last)
print(run("grep -c FATAL /var/log/sp/node-%s.log || true" % INST).strip()
      + " FATAL lines in node log")
if not ok:
    print(run("systemctl status sp-filesrc@%s sp-modelnode@%s --no-pager "
              "-n 12" % (INST, INST)))
    print(run("tail -30 /var/log/sp/node-%s.log" % INST))
    cli.close()
    raise SystemExit(1)
print("INSTALL_HEALTHY")
cli.close()

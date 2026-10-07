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
    # M9a: 触发器 (只读信箱, 崩溃不影响关键链, spec §6)
    (os.path.join(SD, "sp-trigger@.service"),
     "/etc/systemd/system/sp-trigger@.service"),
    (os.path.join(SD, "trigger.env"), "/etc/sp/trigger.env"),
    (os.path.join(SD, "thr.conf"), "/etc/sp/thr.conf"),
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
              "sp-modelnode@%s.service sp-trigger@%s.service; true"
              % (INST, INST, INST)))
    print(run("rm -f /etc/systemd/system/sp-filesrc@.service "
              "/etc/systemd/system/sp-modelnode@.service "
              "/etc/systemd/system/sp-trigger@.service "
              "/etc/sp/%s.env /etc/sp/trigger.env /etc/sp/thr.conf "
              "/etc/tmpfiles.d/sp.conf /etc/logrotate.d/sp "
              "/usr/local/bin/sp_trigger; rm -rf "
              "/usr/local/share/sp/rules" % INST))
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

# ---- M9a 规则插件 + 触发器宿主: 源码 push (自足全量) + 板上编译 ----
# 终审 I2: 编译必须只吃本次 push 的源 (曾编板上陈旧 sp_trigger.cpp);
# 产物先写临时名再 mv (运行中进程占着旧 inode, 直接覆盖 = ETXTBSY);
# 编译失败必须熔断安装 (曾只退出远端 shell, install 照常走完).
import glob  # noqa: E402
RULES_DIR = "/usr/local/share/sp/rules"
PREPOST_BD = "/opt/m0/trt-dev/prepost"
print(run("mkdir -p %s/rules" % PREPOST_BD).strip())
sftp = cli.open_sftp()
PP = os.path.join(ROOT, "deploy", "prepost")
for f in ["sp_result.h", "sp_rule.h", "sp_ruleload.h", "sp_rule_util.h",
          "sp_quota.h", "sp_egoring.h", "sp_trigger.cpp",
          "sp_rule_template.c"]:
    sftp.put(os.path.join(PP, f), "%s/%s" % (PREPOST_BD, f))
for f in glob.glob(os.path.join(PP, "rules", "*.c")):
    sftp.put(f, "%s/rules/%s" % (PREPOST_BD, os.path.basename(f)))
sftp.close()

# 规则 .so: 逐个编到 .tmp 再原子替换; 模板 demo 不进生产目录
# (老 install 曾把 sp_rule_template.so 装进 RULES_DIR, 生产加载着
# demo_speed_high 凑成 15 rules — 清掉)
rules_c = glob.glob(os.path.join(PP, "rules", "*.c"))
out = run("rm -f %s/.tmp_*.so %s/sp_rule_template.so; "
          "cd %s/rules && fail=0; for f in *.c; do "
          "g++ -O2 -shared -fPIC -I%s $f -o %s/.tmp_${f%%.c}.so || "
          "{ fail=1; break; }; "
          "mv -f %s/.tmp_${f%%.c}.so %s/${f%%.c}.so || { fail=1; break; }; "
          "done; [ $fail -eq 0 ] && echo RULES_BUILD_OK" %
          (RULES_DIR, RULES_DIR,
           PREPOST_BD, PREPOST_BD, RULES_DIR, RULES_DIR, RULES_DIR))
print(out.strip())
assert "RULES_BUILD_OK" in out, "rule build failed (see above)"
nso = int(run("ls %s/*.so 2>/dev/null | wc -l" % RULES_DIR).strip() or 0)
assert nso == len(rules_c), "rules %d on board != %d sources" % (nso,
                                                                 len(rules_c))
print("rules: %d .so == %d .c" % (nso, len(rules_c)))

# 触发器宿主二进制 (M9a; board_m1 build 不覆盖它)
out = run("cd %s && g++ -O2 -std=c++14 -Wall -pthread sp_trigger.cpp "
          "-o /tmp/sp_trigger.build -lrt -ldl && "
          "mv -f /tmp/sp_trigger.build /usr/local/bin/sp_trigger && "
          "echo TRIGGER_BUILD_OK" % PREPOST_BD)
print(out.strip())
assert "TRIGGER_BUILD_OK" in out, "sp_trigger build failed (see above)"

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
          "sp-modelnode@%s.service sp-trigger@%s.service"
          % (INST, INST, INST)).strip())
time.sleep(2)

# ---- 健康断言: 75s 内信箱出现 NOMINAL + trigger ACTIVE ----
deadline = time.time() + 75
ok = False
last = ""
while time.time() < deadline:
    st = run("systemctl is-active sp-filesrc@%s sp-modelnode@%s "
             "sp-trigger@%s" % (INST, INST, INST)).strip().replace("\n", " ")
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

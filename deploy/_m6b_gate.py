# -*- coding: utf-8 -*-
"""M6b 门禁: 节点 --mp --serial 12 帧, sp_resultmon 逐帧捕获信箱 v2
final_plan, 本地 numpy 按 plan_cls/plan_reg dump 复算同 cmd 解码对拍.
用法: python _m6b_gate.py"""
import os
import re
import sys
import time

import numpy as np
import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", "m6b")
BD = "/opt/m0/trt-dev/m6b_out"
CMD = 2  # 与节点 --cmd 一致 (2=直行)

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


print("kill:", run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; "
                   "pkill -9 -f sp_resultmon; sleep 1; "
                   "pgrep -af 'sp_filesrc|sp_modelnode|sp_resultmon' "
                   "|| echo none").strip())
# 信箱 shm 一并清掉, 让 mon --wait-ms 等本次节点创建 (旧信箱会秒附)
print(run("rm -rf /dev/shm/sp_m3 /dev/shm/sp_res_sp_result_m3 %s; "
          "mkdir -p %s" % (BD, BD)).strip())

# stdbuf -oL: mon 输出走文件重定向是块缓冲, pkill -9 会把 <4KB 缓冲全丢
# (实测 0 字节 log); 行缓冲 + 先 SIGTERM 优雅退出双保险
mon = ("stdbuf -oL /usr/local/bin/sp_resultmon sp_result_m3 --interval-ms 10 "
       "--duration-ms 180000 "
       "--wait-ms 60000 > %s/mon.log 2>&1 < /dev/null &" % BD)
run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % mon)
fs = ("/usr/local/bin/sp_filesrc m3 1600 900 5 "
      "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
      "--fresh --wait-cons 1 > %s/fs.log 2>&1 < /dev/null &" % BD)
run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
nd = ("/usr/local/bin/sp_modelnode m3 /opt/m0/trt-dev/models/e_bb2.engine "
      "/usr/local/lib/libdfaplug_v8.so "
      "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s "
      "--hd /opt/m0/trt-dev/models/e_hd.engine "
      "--mp /opt/m0/trt-dev/models/e_mp.engine "
      "--warmup 2 --frames 12 --serial --graph --cmd %d "
      "--img-from /opt/m0/trt-dev/vec/mini "
      "> %s/node.log 2>&1 < /dev/null; "
      "echo rc=$? > %s/RC" % (BD, CMD, BD, BD))
run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
o = ""
ok = False
for i in range(150):
    time.sleep(2)
    o = run("cat %s/RC 2>/dev/null" % BD)
    if o.strip().startswith("rc="):
        ok = True
        break
print("rc:", o.strip() if ok else "TIMEOUT")
print(run("tail -4 %s/node.log" % BD))
time.sleep(2)
run("pkill -f sp_filesrc; pkill -f sp_modelnode; "
    "kill -TERM -f sp_resultmon; sleep 1.5; "
    "pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; "
    "pkill -9 -f sp_resultmon; true")
monlog = run("cat %s/mon.log" % BD)
mt = monlog.splitlines()
print("mon: %d lines, tail: %s" % (len(mt), mt[-2:] if mt else "(empty)"))

os.makedirs(DST, exist_ok=True)
sftp = cli.open_sftp()
n = 0
for k in range(12):
    for nm in ("plan_cls", "plan_reg"):
        try:
            sftp.get("%s/outm_%02d/%s.bin" % (BD, k, nm),
                     os.path.join(DST, "outm_%02d_%s.bin" % (k, nm)))
            n += 1
        except IOError:
            print("MISS", k, nm)
sftp.close()
cli.close()
print("fetched %d ref bins" % n)

# ---- 本地: 解析 mon 逐帧 v2 行, numpy 复算同口径解码对拍 ----
v2rows = re.findall(r"v2 seq=(\d+) cmd=(\d+) mode=(\d+) conf=(-?[\d.]+) "
                    r"t_mp=([\d.]+) plan=(\[\[.*?\]\])", monlog)
print("mon v2 rows: %d" % len(v2rows))
n_pass = 0
for seq, cmd, mode, conf, tmp, plan in v2rows:
    k = int(seq) - 1
    pc = np.fromfile(os.path.join(DST, "outm_%02d_plan_cls.bin" % k),
                     np.float32).reshape(18)
    pr = np.fromfile(os.path.join(DST, "outm_%02d_plan_reg.bin" % k),
                     np.float32).reshape(18, 6, 2)
    c = int(cmd)
    best = c * 6 + int(np.argmax(pc[c * 6:(c + 1) * 6]))
    ref = pr[best]
    got = np.array(re.findall(r"\[([\d.-]+),([\d.-]+)\]", plan), np.float64)
    if got.shape != (6, 2):
        print("seq=%s PARSE FAIL %s" % (seq, got.shape))
        continue
    d = np.abs(got - ref).max()
    okf = d <= 2e-3 and best == int(mode)
    n_pass += okf
    print("seq=%2s fid=%2d mode=%d(%d) maxdiff=%.4g %s"
          % (seq, k, int(mode), best, d, "PASS" if okf else "FAIL"))
print("GATE: %d/%d PASS" % (n_pass, len(v2rows)))
print("M6B_DONE" if (ok and n_pass == 12) else "M6B_INCOMPLETE")

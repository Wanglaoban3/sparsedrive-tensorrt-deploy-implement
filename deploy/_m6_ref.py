# -*- coding: utf-8 -*-
"""M6a 同源参考: 板上以节点 M5 闭环 dump(m5fix_out)为 SRC 跑离线 MP 链
(repro/mini_pipeline_mp.sh), 拉回 outm_XX 到 work_dirs/preproc_ref/mpref.
用法: python _m6_ref.py [fetch]"""
import os
import sys
import time

import paramiko

ONLY_FETCH = len(sys.argv) > 1 and sys.argv[1] == "fetch"
HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", "mpref")
BD = "/opt/m0/trt-dev/mpref_out"
KEYS = ["motion_cls", "motion_reg", "plan_cls", "plan_reg", "plan_status",
        "next_history_instance_feature", "next_history_anchor",
        "next_history_period", "next_prev_instance_id",
        "next_prev_confidence", "next_history_ego_feature",
        "next_history_ego_anchor", "next_history_ego_period",
        "next_prev_ego_status"]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


if not ONLY_FETCH:
    # 同源符号农场: 节点 out_XX -> mpSRC/outv8_XX (链脚本按 outv8_$k 取)
    print(run("rm -rf /opt/m0/trt-dev/mpSRC %s; mkdir -p /opt/m0/trt-dev/mpSRC; "
              "for k in $(seq -w 0 80); do "
              "ln -s /opt/m0/trt-dev/m5fix_out/out_$k "
              "/opt/m0/trt-dev/mpSRC/outv8_$k; done; "
              "ls /opt/m0/trt-dev/mpSRC | wc -l").strip())
    # BOUNDS 不传, 用链脚本默认 "0 40" —— bash -c 单引号里再嵌 '0 40'
    # 会把外层引号拆断 (实测 BOUNDS 吃成 "0", k=40 复位丢失)
    nd = ("cd /opt/m0/trt-dev && bash repro/mini_pipeline_mp.sh "
          "/opt/m0/trt-dev/models/e_mp.engine /usr/local/lib/libdfaplug_v8.so "
          "/opt/m0/trt-dev/mpSRC %s 0 80 "
          "> /opt/m0/trt-dev/m6ref.log 2>&1 < /dev/null; "
          "echo rc=$? > %s/RC" % (BD, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
    o = ""
    ok = False
    for i in range(300):  # 10 min: 81 帧 × ~1.5s + 启动
        time.sleep(2)
        o = run("cat %s/RC 2>/dev/null" % BD)
        if o.strip().startswith("rc="):
            ok = True
            break
    print("rc:", o.strip() if ok else "TIMEOUT")
    print(run("tail -3 /opt/m0/trt-dev/m6ref.log").strip())
    print(run("tail -2 %s/../m6ref.log 2>/dev/null; "
              "grep -c '^done' /opt/m0/trt-dev/repro/mini_mp_mpref_out.log "
              "2>/dev/null").strip())

os.makedirs(DST, exist_ok=True)
sftp = cli.open_sftp()
n = 0
for k in range(81):
    for nm in KEYS:
        r = "%s/outm_%02d/%s.bin" % (BD, k, nm)
        try:
            sftp.get(r, os.path.join(DST, "outm_%02d_%s.bin" % (k, nm)))
            n += 1
        except IOError:
            print("MISS", r)
sftp.close()
cli.close()
print("FETCHED %d bins -> %s" % (n, DST))
print("M6REF_DONE")

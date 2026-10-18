# -*- coding: utf-8 -*-
"""mat4_inv 修复后闭环复测: legacy(e_T6 单引擎) / split(e_bb2+e_hd) /
mp(bb2+hd+e_mp 三级, M6a).
用法: python _m5_rerun.py legacy|split|mp [graph|nograph] [fetch]
板上跑 81 帧(输出在 out_XX/ 子目录, mp 另有 outm_XX/), 拉回 eval 键到
work_dirs/preproc_ref/<tag>; fetch 模式只抓取(任务已在板上跑完时用)."""
import os
import sys
import time

import paramiko

MODE = sys.argv[1] if len(sys.argv) > 1 else "legacy"
GRAPH = (sys.argv[2] if len(sys.argv) > 2 else "graph") == "graph"
ONLY_FETCH = len(sys.argv) > 3 and sys.argv[3] == "fetch"
TAG = {"legacy": "m3fix", "split": "m5fix", "mp": "m6fix"}[MODE]
HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
BD = "/opt/m0/trt-dev/%s_out" % TAG
KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
        "map_cls", "map_pts"]
MP_KEYS = ["motion_cls", "motion_reg", "plan_cls", "plan_reg", "plan_status",
           "next_history_instance_feature", "next_history_anchor",
           "next_history_period", "next_prev_instance_id",
           "next_prev_confidence", "next_history_ego_feature",
           "next_history_ego_anchor", "next_history_ego_period",
           "next_prev_ego_status"]

ENG = ("/opt/m0/trt-dev/models/e_T6.engine" if MODE == "legacy"
       else "/opt/m0/trt-dev/models/e_bb2.engine")
HD = "" if MODE == "legacy" else " --hd /opt/m0/trt-dev/models/e_hd.engine"
MP = " --mp /opt/m0/trt-dev/models/e_mp.engine" if MODE == "mp" else ""
GR = " --graph" if GRAPH else ""

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


if not ONLY_FETCH:
    print("kill:", run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; "
                       "sleep 1; pgrep -af 'sp_filesrc|sp_modelnode' "
                       "|| echo none").strip())
    print(run("rm -rf /dev/shm/sp_m3 %s; mkdir -p %s" % (BD, BD)).strip())
    fs = ("/usr/local/bin/sp_filesrc m3 1600 900 5 "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
          "--fresh --wait-cons 1 > %s/fs.log 2>&1 < /dev/null &" % BD)
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
    nd = ("/usr/local/bin/sp_modelnode m3 %s /usr/local/lib/libdfaplug_v8.so "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s%s%s%s "
          "--skip-lag --warmup 2 --frames 81 "
          "--img-from /opt/m0/trt-dev/vec/mini "
          "> %s/node.log 2>&1 < /dev/null; "
          "echo rc=$? > %s/RC" % (ENG, BD, HD, MP, GR, BD, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
    o = ""
    ok = False
    for i in range(240):  # 8 min 上限
        time.sleep(2)
        o = run("cat %s/RC 2>/dev/null" % BD)
        if o.strip().startswith("rc="):
            ok = True
            break
    print("rc:", o.strip() if ok else "TIMEOUT")
    print(run("tail -16 %s/node.log" % BD))
    run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; true")

os.makedirs(DST, exist_ok=True)
sftp = cli.open_sftp()
n = 0
for k in range(81):
    for nm in KEYS:
        r = "%s/out_%02d/%s.bin" % (BD, k, nm)
        try:
            sftp.get(r, os.path.join(DST, "out_%02d_%s.bin" % (k, nm)))
            n += 1
        except IOError:
            print("MISS", r)
    if MODE == "mp":
        for nm in MP_KEYS:
            r = "%s/outm_%02d/%s.bin" % (BD, k, nm)
            try:
                sftp.get(r, os.path.join(DST, "outm_%02d_%s.bin" % (k, nm)))
                n += 1
            except IOError:
                print("MISS", r)
for f in ("frame_log.tsv", "node.log", "fs.log"):
    try:
        sftp.get("%s/%s" % (BD, f), os.path.join(DST, f))
    except IOError:
        pass
sftp.close()
cli.close()
print("FETCHED %d bins -> %s" % (n, DST))
print("RERUN_DONE")

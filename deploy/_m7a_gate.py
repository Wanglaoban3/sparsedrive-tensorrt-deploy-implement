# -*- coding: utf-8 -*-
"""M7a 门禁三连: 双流(bb2∥hd+mp) 精度复跑 + 吞吐 + 单流对照.
用法: python _m7a_gate.py dual5|thr100|pipe100 [fetch]
  dual5   : filesrc 5fps, node --dual --graph --mp, 带 dump, 拉 20 键 x81
  thr100  : filesrc 100Hz, node --dual --graph --mp --no-dump (吞吐)
  pipe100 : filesrc 100Hz, node --graph --mp --no-dump (单流对照)
启动后盯 RC; fetch 模式只抓取(板上已跑完时用)."""
import os
import sys
import time

import paramiko

RUN = sys.argv[1] if len(sys.argv) > 1 else "dual5"
ONLY_FETCH = len(sys.argv) > 2 and sys.argv[2] == "fetch"
TAG = {"dual5": "m7fix", "thr100": "m7thr", "pipe100": "m7pipe",
       "pipe5": "m7srl"}[RUN]
FS_HZ = {"dual5": "5", "thr100": "100", "pipe100": "100", "pipe5": "5"}[RUN]
DUAL = " --dual" if RUN not in ("pipe100", "pipe5") else ""
NODUMP = " --no-dump" if RUN not in ("dual5", "pipe5") else ""
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
    fs = ("/usr/local/bin/sp_filesrc m3 1600 900 %s "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
          "--fresh --wait-cons 1 > %s/fs.log 2>&1 < /dev/null &" % (FS_HZ, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
    nd = ("/usr/local/bin/sp_modelnode m3 /opt/m0/trt-dev/models/e_bb2.engine "
          "/usr/local/lib/libdfaplug_v8.so "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s "
          "--hd /opt/m0/trt-dev/models/e_hd.engine "
          "--mp /opt/m0/trt-dev/models/e_mp.engine "
          "--warmup 2 --frames 81 --img-from /opt/m0/trt-dev/vec/mini "
          "--graph%s%s > %s/node.log 2>&1 < /dev/null; "
          "echo rc=$? > %s/RC" % (BD, DUAL, NODUMP, BD, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
    o = ""
    ok = False
    for i in range(240):
        time.sleep(2)
        o = run("cat %s/RC 2>/dev/null" % BD)
        if o.strip().startswith("rc="):
            ok = True
            break
    print("rc:", o.strip() if ok else "TIMEOUT")
    print(run("tail -20 %s/node.log" % BD))
    run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; true")

os.makedirs(DST, exist_ok=True)
sftp = cli.open_sftp()
if RUN in ("dual5", "pipe5"):
    n = 0
    for k in range(81):
        for nm in KEYS:
            r = "%s/out_%02d/%s.bin" % (BD, k, nm)
            try:
                sftp.get(r, os.path.join(DST, "out_%02d_%s.bin" % (k, nm)))
                n += 1
            except IOError:
                print("MISS", r)
        for nm in MP_KEYS:
            r = "%s/outm_%02d/%s.bin" % (BD, k, nm)
            try:
                sftp.get(r, os.path.join(DST, "outm_%02d_%s.bin" % (k, nm)))
                n += 1
            except IOError:
                print("MISS", r)
    print("FETCHED %d bins -> %s" % (n, DST))
for f in ("frame_log.tsv", "node.log", "fs.log"):
    try:
        sftp.get("%s/%s" % (BD, f), os.path.join(DST, f))
    except IOError:
        pass
sftp.close()
cli.close()
print("M7A_%s_DONE" % RUN.upper())

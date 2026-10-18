# -*- coding: utf-8 -*-
"""M-PROD Phase A 门禁: 量产化改动(watchdog/v3 信箱/退出码)后的闭环精度复跑.
用法: python _mprod_gate.py mp [fetch]
  mp: filesrc 5fps, node bb2+hd+mp --graph (与 M6a/M7a 基线同形状, 无 --dual),
      带 dump, 拉 20 键 x81 -> preproc_ref/mproda
对比基线: m7fix (map 0.7485 / EPA 0.6048,0.5023); 精度评估走 _mprod_eval.py."""
import os
import sys
import time

import paramiko

RUN = sys.argv[1] if len(sys.argv) > 1 else "mp"
ONLY_FETCH = len(sys.argv) > 2 and sys.argv[2] == "fetch"
# MPROD_TAG: 每阶段门禁用独立 tag (A=mproda B=mprodb...), 不覆盖历史参考
TAG = os.environ.get("MPROD_TAG", "mproda")
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
    # 必须 systemctl stop: pkill -9 会触发 Restart=on-failure 把 systemd
    # 发布端拉回同一 ring → 双发布端 → 独立 run 撞外来 seq (FATAL 12
    # seq misalign, mprodc 实测) / claim timeout
    print("stop:", run("systemctl stop sp-filesrc@m3 sp-modelnode@m3; "
                       "systemctl reset-failed sp-filesrc@m3 sp-modelnode@m3; "
                       "pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; "
                       "sleep 1; pgrep -af 'sp_filesrc|sp_modelnode' "
                       "|| echo none").strip())
    print(run("rm -rf /dev/shm/sp_m3 %s; mkdir -p %s" % (BD, BD)).strip())
    fs = ("/usr/local/bin/sp_filesrc m3 1600 900 5 "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
          "--fresh --wait-cons 1 > %s/fs.log 2>&1 < /dev/null &" % BD)
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
    nd = ("/usr/local/bin/sp_modelnode m3 /opt/m0/trt-dev/models/e_bb2.engine "
          "/usr/local/lib/libdfaplug_v8.so "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s "
          "--hd /opt/m0/trt-dev/models/e_hd.engine "
          "--mp /opt/m0/trt-dev/models/e_mp.engine "
          "--warmup 2 --frames 81 --img-from /opt/m0/trt-dev/vec/mini "
          "--graph > %s/node.log 2>&1 < /dev/null; "
          "echo rc=$? > %s/RC" % (BD, BD, BD))
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
    print(run("grep -E 'mode=|watch|mailbox|resync|MODELNODE|FATAL' "
              "%s/node.log | tail -12" % BD))
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
print("MPROD_%s_DONE" % RUN.upper())

# -*- coding: utf-8 -*-
"""M9a Task 4: L1 阈值重标定 (spec §9: 3min 稳态观察 → 分位数重标 → 再进门禁).

板上跑 SP_TRIG_OBSERVE=1 3min → 拉 observe.jsonl → 分位数 → 生成校准 thr.conf.
映射表 (物理阈值不动, 只重标统计敏感项; 其余记录观测证据):
  hard_brake.acc_thr  = p01(acc)        (~1%/窗持续触发率)
  hard_accel.acc_thr  = p99(acc)
  sharp_turn.yaw_thr  = p99(|yaw|)*0.8
  sharp_lat.lat_thr   = p99(|lat_acc|)*0.8
  reverse/u_turn/low_speed_crawl/lead_*/vru_*/construction_zone = 保持 v0 种子
  (物理量纲阈值: -0.5m/s, 120 度, 2m/s, 20/25m, 2/3m, 1.5m/s, 30m — 观测表
   只作分布合理性证据)
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python deploy\\_m9a_thr_calib.py [minutes=3]
留档: work_dirs/preproc_ref/m9a_thr/ (observe.jsonl + thr.conf + report.md)
"""
import datetime
import io
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
assert os.environ.get("BOARD_HOST") and os.environ.get("BOARD_PASS")
import paramiko  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = os.path.join(ROOT, "deploy", "prepost")
BD = "/opt/m0/trt-dev/prepost"
W = "/tmp/m9a_calib"
ARC = os.path.join(ROOT, "work_dirs", "preproc_ref", "m9a_thr")
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=180):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.channel.recv_exit_status(), \
        o.read().decode("utf-8", "replace"), \
        e.read().decode("utf-8", "replace")


print("== 板上编译 + 观察跑 %.0fmin ==" % MIN)
run("mkdir -p %s" % W)
sftp = cli.open_sftp()
for f in ["sp_trigger.cpp", "sp_result.h", "postproc.h", "sp_rule.h",
          "sp_rule_util.h", "sp_ruleload.h", "sp_egoring.h"]:
    sftp.put(os.path.join(PRE, f), "%s/%s" % (BD, f))
sftp.close()
rc, o, e = run("cd %s && g++ -O2 -std=c++14 -Wall -Wextra -pthread "
               "sp_trigger.cpp -o /usr/local/bin/sp_trigger -lrt -ldl" % BD)
assert rc == 0, "build failed:\n" + e[-1500:]
rc, o, e = run("rm -rf %s/out; mkdir -p %s/out; cd %s && "
               "SP_TRIG_RING=m3 SP_TRIG_HZ=10 SP_TRIG_OBSERVE=1 "
               "SP_TRIG_OUT=%s/out timeout -s TERM %d /usr/local/bin/"
               "sp_trigger" % (W, W, W, W, int(MIN * 60 + 10)),
               t=int(MIN * 60 + 60))
print("observe rc=%d" % rc)
print(e[-600:])

sftp = cli.open_sftp()
os.makedirs(ARC, exist_ok=True)
sftp.get(W + "/out/observe.jsonl", os.path.join(ARC, "observe.jsonl"))
sftp.close()
cli.close()

rows = [ln for ln in open(os.path.join(ARC, "observe.jsonl"),
                          encoding="utf-8").read().splitlines() if ln.strip()]
import json  # noqa: E402
recs = [json.loads(ln) for ln in rows]
print("observe rows:", len(recs))
assert len(recs) >= MIN * 60 * 3, "samples too few"


def col(name):
    return np.array([r[name] for r in recs if r.get(name) is not None],
                    dtype=float)


acc = col("acc")
yaw = col("yaw_abs")
lat = col("lat_abs")
speed = col("speed")
lead_d = col("lead_d")
vru_lat = col("vru_lat")
vru_cross = col("vru_cross")
cone_d = col("cone_d")

# 运动门控: 静止段 (mini 回环大量停车) 的航向抖动会产生假 yaw/lat 尖峰
# (实测 p99≈2π), 统计分位数只在 speed>1 m/s 的移动样本上取
mov = speed > 1.0
n_mov = int(mov.sum())
acc_m, yaw_m, lat_m = acc[mov], yaw[mov], lat[mov]
# acc 门控加严: 复位/边界帧后第一帧 speed 从 0 跳 8.5 → acc≈17 (伪影),
# 只取前后帧都在动的 acc 样本
spd_arr = speed
acc_pair = np.array([spd_arr[i] > 1.0 and spd_arr[i - 1] > 1.0
                     if i > 0 else False for i in range(len(acc))])
acc_m = acc[acc_pair]
print("moving samples: %d/%d (speed>1), acc pair-valid: %d" %
      (n_mov, len(recs), len(acc_m)))
assert len(acc_m) >= 30, "moving samples too few for calibration"

acc_p01, acc_p99 = np.percentile(acc_m, [1, 99])
yaw_p99 = np.percentile(yaw_m, 99)
lat_p99 = np.percentile(lat_m, 99)

lines = []
lines.append("# M9a 阈值重标定 (%s, %.0fmin, %d 帧)" %
             (datetime.date.today().isoformat(), MIN, len(recs)))
lines.append("")
lines.append("## 校准项 (统计敏感, 分位数重标)")
lines.append("")
lines.append("| 项 | 观测 p01/p99 (相关位) | 校准值 |")
lines.append("|---|---|---|")
lines.append("| hard_brake.acc_thr | p01=%.3f (moving n=%d) | %.2f |" %
             (acc_p01, n_mov, acc_p01))
lines.append("| hard_accel.acc_thr | p99=%.3f | %.2f |" % (acc_p99, acc_p99))
lines.append("| sharp_turn.yaw_thr | p99=%.3f | %.3f |" %
             (yaw_p99, yaw_p99 * 0.8))
lines.append("| sharp_lat.lat_thr | p99=%.3f | %.3f |" %
             (lat_p99, lat_p99 * 0.8))
lines.append("")
lines.append("注: mini 回环为机制验证数据 (静止段+det 噪声), 车队数据上"
             "须重跑本工具; 物理阈值项在真实分布上的证据如下。")
lines.append("")
lines.append("## 物理阈值项观测证据 (保持 v0 种子)")
lines.append("")
lines.append("| 项 | 种子 | 观测分布 |")
lines.append("|---|---|---|")
lines.append("| reverse.speed_thr | -0.5 | speed p01=%.3f min=%.3f |" %
             (np.percentile(speed, 1), speed.min()))
lines.append("| vru_near.lat_thr | 2.0 | vru_lat p05=%s n=%d |" %
             ("%.3f" % np.percentile(vru_lat, 5) if len(vru_lat) else "n/a",
              len(vru_lat)))
lines.append("| vru_cross.cross_thr | 1.5 | vru_cross p95=%s n=%d |" %
             ("%.3f" % np.percentile(vru_cross, 95) if len(vru_cross)
              else "n/a", len(vru_cross)))
lines.append("| lead_hard_brake.dist_thr | 20.0 | lead_d p50=%s n=%d |" %
             ("%.2f" % np.percentile(lead_d, 50) if len(lead_d) else "n/a",
              len(lead_d)))
lines.append("| construction_zone.dist_thr | 30.0 | cone_d min=%s n=%d |" %
             ("%.2f" % cone_d.min() if len(cone_d) else "n/a", len(cone_d)))
lines.append("| low_speed_crawl.speed_thr | 2.0 | speed p50=%.3f |" %
             np.percentile(speed, 50))
report = "\n".join(lines) + "\n"
open(os.path.join(ARC, "report.md"), "w", encoding="utf-8").write(report)
print(report)

# 生成校准 thr.conf (种子模板 + 校准段替换)
seed = open(os.path.join(ROOT, "deploy", "systemd", "thr.conf"),
            encoding="utf-8").read()
import re  # noqa: E402


def set_sec(text, sec, kv):
    pat = re.compile(r"\[%s\]\n(?:[a-z_]+=[0-9.\-]+\n)+" % sec)
    assert pat.search(text), "section %s not found" % sec
    body = "[%s]\n%s" % (sec, kv)
    return pat.sub(body, text, count=1)


cal = set_sec(seed, "hard_brake", "acc_thr=%.3f\ndur_s=0.5\n" % acc_p01)
cal = set_sec(cal, "hard_accel", "acc_thr=%.3f\ndur_s=0.5\n" % acc_p99)
cal = set_sec(cal, "sharp_turn", "yaw_thr=%.3f\ndur_s=0.5\n" % (yaw_p99 * .8))
cal = set_sec(cal, "sharp_lat", "lat_thr=%.3f\ndur_s=0.3\n" % (lat_p99 * .8))
open(os.path.join(ARC, "thr.conf"), "w", encoding="utf-8").write(cal)
open(os.path.join(ROOT, "deploy", "systemd", "thr.conf"), "w",
     encoding="utf-8").write(cal)
print("calibrated thr.conf -> deploy/systemd/thr.conf + 留档")
print("CALIB_DONE")

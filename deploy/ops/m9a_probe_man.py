# -*- coding: utf-8 -*-
"""验证: l2g 的车前向轴 = R 第 1 列 (lidar 90 度装转), 投影后 speed≈8.4."""
import os
import json
import math

import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)
_, o, _ = cli.exec_command(
    "head -8 /opt/m0/trt-dev/nv12_r0/manifest.jsonl | tail -6", timeout=30)
lines = [l for l in o.read().decode().splitlines() if l.strip()]
cli.close()

prev = None
for ln in lines:
    d = json.loads(ln)
    flat = [v for row in d["l2g"] for v in row]
    t = int(d["ts_ns"])
    cur = (t, flat)
    if prev is not None:
        dt = (t - prev[0]) / 1e9
        dx, dy = flat[3] - prev[1][3], flat[7] - prev[1][7]
        disp_ang = math.atan2(dy, dx)
        head_col0 = math.atan2(flat[4], flat[0])
        head_col1 = math.atan2(flat[5], flat[1])
        v_proj0 = (dx * math.cos(head_col0) + dy * math.sin(head_col0)) / dt
        v_proj1 = (dx * math.cos(head_col1) + dy * math.sin(head_col1)) / dt
        print("dt=%.1f |disp|=%.2f disp_ang=%+.2f col0=%+.2f col1=%+.2f "
              "v_col0=%+.2f v_col1=%+.2f hypot=%.2f" %
              (dt, math.hypot(dx, dy), disp_ang, head_col0, head_col1,
               v_proj0, v_proj1, math.hypot(dx, dy) / dt))
    prev = cur

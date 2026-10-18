# -*- coding: utf-8 -*-
"""终审 C1 复验: 扫 ctx 夹具, 验证 ±π 翻负伪影消失.

判据: 每个有 >=2 帧历史的 ctx, 速度模长 v_mag = |Δpos|/Δt (纯几何, 不经
heading); 修复前交付夹具 24 帧出现 speed≈-3.6~-3.9 (符号翻转) — 修复后
speed 应与 v_mag 同号且前向行驶恒正. 输出 count(speed < -1.0) 与最差样本.
用法: python deploy\\_m9a_unwrap_check.py [ctx_dir]
  默认 ctx_dir = work_dirs/preproc_ref/m9a_parity/ctx (parity 重跑产物)
"""
import glob
import io
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m9a_replay as rp  # noqa: E402

ctx_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "work_dirs", "preproc_ref", "m9a_parity", "ctx")

neg = []       # (seq, speed, v_mag)
checked = 0
for fn in sorted(glob.glob(os.path.join(ctx_dir, "ctx_*.bin"))):
    c = rp.read_ctx(fn)
    if c["n_hist"] < 2:
        continue
    cur, prv = c["ego"], c["hist_ego"][c["n_hist"] - 2]
    dt = (c["ts_ns"] - c["hist_ts"][c["n_hist"] - 2]) / 1e9
    if dt <= 0:
        continue
    vmag = math.hypot(float(cur["x"]) - float(prv["x"]),
                      float(cur["y"]) - float(prv["y"])) / dt
    spd = float(cur["speed"])
    checked += 1
    if spd < -1.0:
        neg.append((c["seq"], spd, vmag))

print("ctx checked: %d (n_hist>=2)" % checked)
print("speed < -1.0 frames: %d  (修复前同数据 24)" % len(neg))
for s, sp, vm in neg[:10]:
    print("  seq=%d speed=%.3f v_mag=%.3f" % (s, sp, vm))
if not neg:
    print("UNWRAP_CHECK_PASS")
else:
    print("UNWRAP_CHECK_FAIL")
sys.exit(0 if not neg else 1)

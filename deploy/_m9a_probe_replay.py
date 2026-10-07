# -*- coding: utf-8 -*-
"""本地二分: 找出 read_ctx/hard_brake 抛 IndexError 的 ctx 文件."""
import io
import os
import sys
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m9a_replay as rp  # noqa: E402

d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "work_dirs", "preproc_ref", "m9a_parity", "ctx")
thr = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "work_dirs", "preproc_ref", "m9a_parity", "thr.conf")
secs = rp.parse_thr(thr)
for fn in sorted(os.listdir(d)):
    if not fn.endswith(".bin"):
        continue
    try:
        c = rp.read_ctx(os.path.join(d, fn))
        assert len(c["hist_ts"]) == c["n_hist"], \
            "hist_ts %d != n_hist %d" % (len(c["hist_ts"]), c["n_hist"])
        s = {"ego.speed": float(c["ego"]["speed"])}
        for name, fr in rp.RULES.items():
            fr(c, s, dict(secs.get(name, {})))
    except Exception:
        print("OFFENDER:", fn)
        traceback.print_exc()
        break
else:
    print("ALL OK", len(os.listdir(d)))

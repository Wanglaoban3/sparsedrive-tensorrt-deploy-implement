# -*- coding: utf-8 -*-
"""本地重解析 M6b 门禁 (复用板上已抓的 mon.log + m6b dump, 不重跑板)."""
import os
import re

import numpy as np
import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", "m6b")
CMD = 2

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username="root", password=PW, timeout=15)
_, o, _ = c.exec_command("cat /opt/m0/trt-dev/m6b_out/mon.log", timeout=30)
monlog = o.read().decode("utf-8", "replace")
c.close()

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
    ci = int(cmd)
    best = ci * 6 + int(np.argmax(pc[ci * 6:(ci + 1) * 6]))
    ref = pr[best]
    got = np.array(re.findall(r"\[([\d.-]+),([\d.-]+)\]", plan), np.float64)
    if got.shape != (6, 2):
        print("seq=%s PARSE FAIL %s" % (seq, got.shape))
        continue
    d = np.abs(got - ref).max()
    okf = d <= 2e-3 and best == int(mode)
    n_pass += okf
    print("seq=%2s fid=%2d mode=%2d(%2d) maxdiff=%.4g %s"
          % (seq, k, int(mode), best, d, "PASS" if okf else "FAIL"))
print("GATE: %d/%d PASS" % (n_pass, len(v2rows)))
print("M6B_PARSE_DONE")

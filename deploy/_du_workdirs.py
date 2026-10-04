# -*- coding: utf-8 -*-
"""work_dirs 一级子目录体积盘点."""
import os
import sys

io = sys.stdout
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "line_buffering")


def du(p):
    t = 0
    for r, ds, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "work_dirs")
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d)
    if os.path.isdir(p):
        print("%9.1f MB  %s" % (du(p) / 1048576.0, d))

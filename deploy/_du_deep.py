# -*- coding: utf-8 -*-
"""细盘: preproc_ref 与 evaldata 一级子目录 + 散文件."""
import codecs
import os
import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "line_buffering")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def du(p):
    t = 0
    for r, ds, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


PR = os.path.join(ROOT, "work_dirs", "preproc_ref")
print("== preproc_ref ==")
for d in sorted(os.listdir(PR)):
    p = os.path.join(PR, d)
    if os.path.isdir(p):
        print("%9.1f MB  %s" % (du(p) / 1048576.0, d))
    else:
        print("%9.1f MB  %s" % (os.path.getsize(p) / 1048576.0, d))

EV = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
print("== evaldata ==")
for d in sorted(os.listdir(EV)):
    p = os.path.join(EV, d)
    if os.path.isdir(p):
        print("%9.1f MB  %s" % (du(p) / 1048576.0, d))
    else:
        print("%9.1f MB  %s" % (os.path.getsize(p) / 1048576.0, d))

SD = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2")
print("== sparsedrive_small_stage2 一级 ==")
for d in sorted(os.listdir(SD)):
    p = os.path.join(SD, d)
    if os.path.isdir(p):
        print("%9.1f MB  %s" % (du(p) / 1048576.0, d))
    else:
        print("%9.1f MB  %s" % (os.path.getsize(p) / 1048576.0, d))

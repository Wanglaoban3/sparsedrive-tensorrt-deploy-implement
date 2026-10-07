# -*- coding: utf-8 -*-
"""ctx bin 完整性校验: 头声明 vs 实际走查长度."""
import io
import os
import struct
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "work_dirs", "preproc_ref", "m9a_parity", "ctx")
EGO, TRK = 24, 48
files = sorted(os.listdir(d))
print("files:", len(files))
bad = []
for fn in files:
    b = open(os.path.join(d, fn), "rb").read()
    scene, nd, nh = struct.unpack_from("<III", b, 28)
    off = 40 + EGO + nd * TRK
    cnt = 0
    while off + 16 + 24 <= len(b) and cnt < nh:
        n = struct.unpack_from("<I", b, off + 24 + 8)[0]
        off += 24 + 16 + n * TRK
        cnt += 1
    if not (cnt == nh and off == len(b)):
        bad.append((fn, nh, cnt, off, len(b)))
print("bad:", len(bad))
for t in bad[:10]:
    print(t)

# -*- coding: utf-8 -*-
"""bucket a trtexec --dumpProfile table into module groups, per-iter ms.

usage: python profile_buckets.py <logfile> [<logfile2> ...]
rows look like:
[I]      123.45       0.1234       0.1234     1.2   LayerName
Total row: [I]     1748.56       0.2320       0.2320     100.0   Total
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROW = re.compile(r"(?:\[.*?\])?\s*\[I\]\s+([0-9.]+)\s+([0-9.]+)\s+"
                 r"([0-9.]+)\s+([0-9.]+)\s+(.+?)\s*$")


def bucket_of(name):
    n = name.lower()
    if n == "total":
        return "_total"
    if "reformatting copynode" in n:
        return "reformat"
    if "deformableaggregation" in n:
        return "dfa_plugin"
    if "foreignnode" in n:
        if "attn/inner_attn/pow" in n:
            return "myelin_decoder_1"
        return "myelin_other"
    if "img_backbone" in n or "img_neck" in n or "input_quantizer" in n \
            or "weight_quantizer" in n:
        return "backbone_neck_int8"
    if "weights_fc/matmul" in n or "kps" in n:
        return "kps_gemm"
    if n.startswith("/layers.") or "/decoder" in n or "head" in n:
        return "head_layers"
    if "map" in n:
        return "map_head"
    return "other"


for path in sys.argv[1:]:
    print(f"########## {path}")
    buckets = {}
    total = None
    nrows = 0
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = ROW.match(ln)
        if not m:
            continue
        tot_ms, avg_ms = float(m.group(1)), float(m.group(2))
        name = m.group(5)
        if name == "Total":
            total = avg_ms
            continue
        nrows += 1
        b = bucket_of(name)
        buckets.setdefault(b, [0.0, 0])
        buckets[b][0] += avg_ms
        buckets[b][1] += 1
    print(f"rows={nrows}  total_row={total}")
    order = ["backbone_neck_int8", "dfa_plugin", "myelin_decoder_1",
             "myelin_other", "map_head", "kps_gemm", "head_layers",
             "reformat", "other"]
    for b in order:
        if b in buckets:
            ms, cnt = buckets[b]
            print(f"  {b:22s} {ms/1.0:8.4f} ms/iter  ({cnt} layers)")
    for b, (ms, cnt) in sorted(buckets.items()):
        if b not in order:
            print(f"  {b:22s} {ms/1.0:8.4f} ms/iter  ({cnt} layers)")
    print()

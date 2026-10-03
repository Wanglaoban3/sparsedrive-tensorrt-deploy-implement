# -*- coding: utf-8 -*-
"""e_T6+v8 profile 分布: Time/Avg/Median/%/Layer 表 -> 桶汇总 + top 行."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

PATHS = [
    r"artifacts\profile\v8_prof4.log",
    r"artifacts\profile\t6_prof.log",
]

ROW = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(.+)$")
IMARK = "[I]"


def parse(path):
    rows = []
    intable = False
    for ln in open(path, encoding="utf-8", errors="replace"):
        i = ln.find(IMARK)
        if i < 0:
            continue
        body = ln[i + 4:].strip()
        if "=== Profile" in body:
            intable = True
            continue
        if not intable:
            continue
        if body.startswith("Time(ms)"):
            continue
        if "Total" in body.split()[-1:][0] if body.split() else False:
            pass
        m = ROW.match(body)
        if not m:
            if "Total" in body:
                pass
            continue
        t, avg, med, pct, name = (float(m.group(1)), float(m.group(2)),
                                  float(m.group(3)), float(m.group(4)),
                                  m.group(5).strip())
        if name == "Total":
            continue
        rows.append((name, avg, med, pct))
    return rows


def bucket(name):
    if "DeformableAggregation" in name:
        return "DFA 插件 (12 调用)"
    if "ForeignNode" in name:
        return "Myelin 融合区 (ForeignNode)"
    if "Reformat" in name:
        return "Reformat"
    if "/img_backbone/" in name or "/img_neck/" in name:
        return "backbone+neck INT8"
    return "head 侧小算子 (MatMul/Elementwise等)"


for p in PATHS:
    rows = parse(p)
    if not rows:
        print("NO ROWS:", p)
        continue
    s = sum(r[1] for r in rows)
    print("=" * 92)
    print(f"### {p}  rows={len(rows)}  sum(Avg)={s:.2f} ms")
    agg = {}
    for name, avg, med, pct in rows:
        agg[bucket(name)] = agg.get(bucket(name), 0.0) + avg
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {v:8.3f} ms  {100 * v / s:5.1f}%  {k}")
    print("  -- top 15 层 --")
    for name, avg, med, pct in sorted(rows, key=lambda r: -r[1])[:15]:
        print(f"  {avg:8.4f} ms  {pct:5.1f}%  {name[:86]}")

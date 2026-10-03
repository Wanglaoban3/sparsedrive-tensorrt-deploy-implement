# -*- coding: utf-8 -*-
"""e_T6+v8 profile 分布: 解析 trtexec Profile 表 -> top 行 + 桶汇总."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

PATHS = [
    r"artifacts\profile\v8_prof4.log",
    r"artifacts\profile\t6_prof.log",
]

MARK = re.compile(r"\[I\] (.*)$")


def parse(path):
    lines = open(path, encoding="utf-8", errors="replace").readlines()
    rows, total, intable = [], None, False
    for ln in lines:
        m = MARK.search(ln)
        if not m:
            continue
        body = m.group(1)
        if "=== Profile ===" in body:
            intable = True
            continue
        if not intable:
            continue
        if "GPU Compute Time" in body:
            mm = re.search(r"mean ([0-9.]+)", body)
            total = float(mm.group(1)) if mm else None
            intable = False
            continue
        # 表行: <name>  avg  min  max  pct%
        nums = re.findall(r"[0-9]+\.[0-9]+", body)
        if len(nums) < 4:
            continue
        name = body[: body.rindex(nums[-4])].strip()
        avg, mn, mx, pct = (float(x) for x in nums[-4:])
        rows.append((name, avg, mn, mx, pct))
    return rows, total


def bucket(name):
    if "DeformableAggregation" in name:
        return "DFA插件(12调用)"
    if "ForeignNode" in name or "Myelin" in name:
        return "Myelin大融合区(ForeignNode)"
    if "Reformat" in name:
        return "Reformat"
    if name.startswith("LocalID:") or "conv_" in name or name.endswith(
            "conv") or "Conv" in name:
        return "conv类(backbone/neck为主)"
    return "其他小算子"


for p in PATHS:
    rows, total = parse(p)
    if not rows:
        print("NO ROWS:", p)
        continue
    s = sum(r[1] for r in rows)
    print("=" * 90)
    print(f"{p}  rows={len(rows)}  sum={s:.2f}ms  GPU_mean={total}ms")
    print("-- top 22 --")
    for name, avg, mn, mx, pct in sorted(rows, key=lambda r: -r[1])[:22]:
        print(f"  {avg:8.3f}ms ({pct:4.1f}%)  {name[:78]}")
    agg = {}
    for name, avg, mn, mx, pct in rows:
        agg[bucket(name)] = agg.get(bucket(name), 0.0) + avg
    print("-- buckets --")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {v:8.3f}ms  {100 * v / s:5.1f}%  {k}")

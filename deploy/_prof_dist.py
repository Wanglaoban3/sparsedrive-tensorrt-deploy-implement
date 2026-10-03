# -*- coding: utf-8 -*-
"""解析 e_T6+v8 profile 日志: top 行 + 桶汇总."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

PATHS = [
    r"<REPO>"
    r"\deploy\artifacts\profile\t6_prof.log",
    r"<REPO>"
    r"\deploy\artifacts\profile\v8_prof4.log",
]

# trtexec dumpProfile 行: "... Name ...  avg ms  min ms  max ms  pct%"
ROW = re.compile(r"^(.*\S)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*%?\s*$")


def parse(path):
    rows = []
    total = None
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.rstrip()
        if "GPU Compute Time" in ln:
            m = re.search(r"min ([0-9.]+).*max ([0-9.]+).*mean ([0-9.]+)", ln)
            if m:
                total = float(m.group(3))
            continue
        m = ROW.match(ln)
        if not m:
            continue
        name = m.group(1)
        # 去掉 name 前缀的时间列干扰: 取最后 4 个数字为 avg/min/max/pct
        nums = re.findall(r"[0-9]*\.[0-9]+", name)
        if len(nums) < 3:
            continue
        avg, mn, mx = (float(x) for x in nums[-3:])
        idx = name.rindex(nums[-1]) + len(nums[-1])
        name = name[:name.rindex(nums[-3])].strip()
        if "Latency" in name or "Throughput" in name or name == "":
            continue
        rows.append((name, avg, mn, mx))
    return rows, total


def bucket(name):
    if "DeformableAggregation" in name:
        return "DFA 插件(12调用)"
    if "ForeignNode" in name or "#Myelin" in name:
        return "Myelin融合区(ForeignNode)"
    if "Reformat" in name:
        return "Reformat"
    if ("conv" in name.lower() or "Conv" in name
            or "backbone" in name.lower() or "_int8" in name
            or "Q/DQ" in name or "(Re)" in name):
        return "backbone+neck(INT8 conv等)"
    if "SoftMax" in name or "softmax" in name.lower():
        return "独立softmax等小算子"
    return "其他(head/decoder小算子)"


for p in PATHS:
    try:
        rows, total = parse(p)
    except FileNotFoundError:
        print("MISSING", p)
        continue
    if not rows:
        print("NO ROWS PARSED", p)
        continue
    s = sum(r[1] for r in rows)
    print("=" * 100)
    print(p.split("\\")[-1], "rows:", len(rows), "sum:", round(s, 2),
          "ms  (GPU mean:", total, "ms)")
    print("-- top 25 rows (avg ms / iter) --")
    for name, avg, mn, mx in sorted(rows, key=lambda r: -r[1])[:25]:
        print(f"  {avg:8.4f}  {name[:80]}")
    agg = {}
    for name, avg, mn, mx in rows:
        agg[bucket(name)] = agg.get(bucket(name), 0.0) + avg
    print("-- buckets --")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {v:8.3f} ms  {100*v/s:5.1f}%  {k}")

# -*- coding: utf-8 -*-
"""悬空引用校验: 文档里提到的每个 *.py 文件名必须真实存在于仓库.
ALLOW = 历史叙事点名、文件已删 (git 历史可查) 的白名单; 只收 docs/ 纯叙事
提及, 不收 AGENTS.md 的引用 (AGENTS 是工作手册, 引用必须真实存在).
docs/superpowers/ 为历史 specs/plans 归档, 整体跳过."""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW = {
    # 2026-10-18 重整: OPTIMIZATION_SUMMARY/FULL_CHAIN_REPRO 历史轮次叙事点名
    "analyze_spans.py", "board_bb2.py", "board_mod_ab.py", "board_mp.py",
    "board_split_build.py", "census_det.py", "dfaplug_dump.cu",
    "dfaplug_v10.cu", "extract_dfa_consts.py", "extract_region.py",
    "make_v5_p2h.py", "onnx2engine3.cpp", "probe_fa_io.py",
    "sim_dfa_v8.py", "smoke_mini_pipeline.py", "split_analysis.py",
    "strip_casts.py", "test_dfa_ab.cu", "test_dfa_kernels.cu",
    "test_dfa_pure.py", "test_dfa_real2.cu", "trace_dfa_inputs.py",
    "_bx.py", "_m6_cmp.py", "_m6_ref.py", "_m6b_gate.py",
    "_m7a_bit2.py", "_m7a_cmp.py", "_m7a_gate.py", "_m8_gate.py",
    "_m8_nvbuf.py", "_m8_probe.py",
    # 外部仓库/系统头/文档示例占位/板上路径 — 非本仓文件
    "4.cu", "8.py", "_rewrite_report.json", "events.py",
    "mini_pipeline_sp2.sh", "nvrtc_compile.cpp", "script.sh", "soak.sh",
    "stat.h", "stdio.h", "stdlib.h", "x.cu", "x.sh", "xxx.sh",
}
on_disk = set()
for dp, _, fns in os.walk(ROOT):
    dirs = dp.split(os.sep)
    if any(x in dirs for x in (".git", "work_dirs", "evaldata")):
        continue
    for fn in fns:
        on_disk.add(fn)

docs = ["AGENTS.md"] + [os.path.join(dp, fn) for dp, _, fns in
                        os.walk(os.path.join(ROOT, "docs"))
                        for fn in fns if fn.endswith(".md")
                        and "superpowers" not in dp]
pat = re.compile(r"\b([A-Za-z0-9_]+\.(?:py|cpp|h|cu|sh|conf|json))\b")
missing = {}
for f in docs:
    txt = open(f, encoding="utf-8", errors="replace").read()
    for m in pat.findall(txt):
        if m in ALLOW:
            continue
        if m not in on_disk and m not in missing:
            missing[m] = os.path.relpath(f, ROOT)

if missing:
    print("DANGLING REFS:")
    for k, v in sorted(missing.items()):
        print("  %-36s (referenced in %s)" % (k, v))
else:
    print("NO_DANGLING_REFS (all documented files exist)")

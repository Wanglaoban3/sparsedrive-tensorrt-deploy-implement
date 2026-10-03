# -*- coding: utf-8 -*-
"""暂存 + 审查 + 提交."""
import io
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = r"H:\projects\sparsedrive-tensorrt-deploy-implement"
GIT = r"C:\Users\wangyi\cmd\git.exe"
os.chdir(ROOT)


def g(*a, **kw):
    r = subprocess.run([GIT] + list(a), capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       timeout=kw.get("timeout", 600))
    return r.stdout + r.stderr


print(g("add", "-A"))
out = g("status", "--short")
lines = out.splitlines()
from collections import Counter
c = Counter(ln[:2].strip() or "??" for ln in lines)
print("staged:", dict(c), "total", len(lines))

# 安全审查: 不该出现的
bad = [ln for ln in lines if any(
    k in ln.lower() for k in ("agents.md", ".engine", ".onnx", "data/",
                              "mingit", "board_pass"))]
print("suspicious:", len(bad))
for b in bad[:10]:
    print("  ", b[:120])

# 凭据终检: 暂存内容里不得有 IP/密码
out = g("grep", "--cached", "-nE", "192\\.168\\.2\\.104|password.{0,4}nvidia",
        "--", ".")
print("credential grep on index:", "CLEAN" if not out.strip() else out[:600])

# 抽样看将提交的新文件
news = [ln[3:] for ln in lines if ln.startswith("A ") or ln.startswith("?? ")]
print(f"\nnew files: {len(news)}, sample:")
for p in news[:15]:
    print("  ", p)

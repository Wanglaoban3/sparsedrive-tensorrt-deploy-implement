# -*- coding: utf-8 -*-
"""v11 DFA gather 向量化: push 源码 -> 板端模块编译 -> 真实 IO A/B + 分相计时.
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _v11_bench.py [iters]
不重编引擎 (插件按 名字/版本/命名空间 运行时绑定); 二进制进 /usr/local/bin."""
import os
import sys
import time

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "deploy")
FILES = ["dfaplug_v11.cu", "test_dfa_v11.cu"]
RDIR = "/opt/m0/trt-dev/src"
DUMP = "/opt/m0/trt-dev/dfa_eT6"
BIN = "/usr/local/bin/test_dfa_v11"

iters = sys.argv[1] if len(sys.argv) > 1 else "50"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("[stderr]\n" + err) if err.strip() else "")


w = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)

sftp = cli.open_sftp()
for fn in FILES:
    sftp.put(os.path.join(SRC, fn), "%s/%s" % (RDIR, fn))
    w.write("push %s\n" % fn)
sftp.close()
w.write(run("cd %s && sed -i 's/\\r$//' %s && ls -l %s"
            % (RDIR, " ".join(FILES), FILES[0])))

w.write("---- nvcc compile ----\n")
t0 = time.time()
r = run("cd %s && nvcc -O3 -arch=sm_87 -o %s test_dfa_v11.cu 2>&1 | tail -25"
        "; echo BUILD_RC=${PIPESTATUS[0]}" % (RDIR, BIN), t=600)
w.write(r)
if "error" in r.lower() and "BUILD_RC=0" not in r:
    w.write("COMPILE_FAILED\n")
    cli.close()
    raise SystemExit(1)
w.write("compile %.1fs\n" % (time.time() - t0))

w.write("---- A/B bench (real IO) ----\n")
r = run("%s %s %s 10" % (BIN, DUMP, iters), t=900)
w.write(r)
# map gather 的 split 扫描 (诊断: 并行度平台确认) + det 侧小 split 探查
for tag, only, splits in (("map", "4", ("16", "32", "64")),
                          ("det", "0", ("1", "2", "4"))):
    for sp in splits:
        r = run("DFA_V11_SPLIT=%s %s %s %s 20 %s"
                % (sp, BIN, DUMP, iters, only), t=600)
        keep = [ln for ln in r.splitlines()
                if ln.startswith("TIME dfa") or "DONE" in ln]
        w.write("[split=%s only=%s]\n%s\n" % (sp, only, "\n".join(keep)))
w.write("V11_BENCH_SCRIPT_DONE\n")
cli.close()

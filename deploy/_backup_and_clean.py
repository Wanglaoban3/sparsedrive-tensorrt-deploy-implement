# -*- coding: utf-8 -*-
"""板侧: 1) 交付引擎+插件拉回本地 models/ 留档(核 md5); 2) 删已拉回的 out 目录."""
import codecs
import hashlib
import os
import sys
import time

import paramiko

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "line_buffering")
HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL = os.path.join(ROOT, "models")
os.makedirs(MDL, exist_ok=True)

DEV = "/opt/m0/trt-dev"
FILES = [
    (DEV + "/models/e_T6.engine", "e_T6.engine"),
    (DEV + "/models/e_bb2.engine", "e_bb2.engine"),
    (DEV + "/models/e_hd.engine", "e_hd.engine"),
    (DEV + "/models/e_mp.engine", "e_mp.engine"),
    ("/usr/local/lib/libdfaplug_v8.so", "libdfaplug_v8.so"),
]
DEL_DIRS = ["m3fix_out", "m6fix_out", "mpref_out", "m6b_out", "probe_out"]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out, err


print("== 板侧体积 ==")
o, _ = run("du -sh %s/models %s/m5fix_out %s/mpSRC %s/vec "
           "%s/repro 2>/dev/null; du -sh %s/*_out %s/probe_out 2>/dev/null"
           % (DEV, DEV, DEV, DEV, DEV, DEV, DEV))
print(o)

sftp = cli.open_sftp()
print("== 拉取备份 ==")
for rp, ln in FILES:
    lp = os.path.join(MDL, ln)
    t0 = time.time()
    sftp.get(rp, lp)
    md5 = hashlib.md5(open(lp, "rb").read()).hexdigest()
    print("  %-18s %8.1f MB  %5.1fs  md5 %s" % (
        ln, os.path.getsize(lp) / 1048576.0, time.time() - t0, md5))
sftp.close()

print("== 删除已拉回的 out 目录 ==")
o, _ = run("cd %s && rm -rf %s && echo CLEAN_OK && ls -d %s/*_out 2>/dev/null"
           % (DEV, " ".join(DEL_DIRS), DEV))
print(o)

o, _ = run("df -h /opt/m0 | tail -1")
print("disk:", o.strip())
cli.close()
print("BOARD_CLEAN_DONE")

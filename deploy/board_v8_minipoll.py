# -*- coding: utf-8 -*-
"""poll run_v8final: print 100-iter profile rows at PROF4 marker; at
V8FINAL_DONE verify mini log, tar+fetch outv8_XX to evaldata/mini_eng_v8."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
RDIR = "/opt/m0/trt-dev"
P4 = RDIR + "/mods/V8PROF4_DONE"
FIN = RDIR + "/mods/V8FINAL_DONE"
LROOT = (r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2\evaldata")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


t0 = time.time()
p4 = fin = False
while time.time() - t0 < 90 * 60:
    st = run(f"ls {P4} {FIN} 2>/dev/null")
    if not p4 and "V8PROF4_DONE" in st:
        p4 = True
        print("== 100-iter v8 profile rows ==")
        print(run("grep 'DeformableAggregation' " + RDIR
                  + "/v8_prof4.log | grep -v Reformat"))
        print(run("grep ' Total' " + RDIR + "/v8_prof4.log | tail -1"))
        print(run("grep 'dfa_v8' " + RDIR + "/v8_prof4.log | head -3"))
    if "V8FINAL_DONE" in st:
        fin = True
        break
    time.sleep(60)
if not fin:
    print("TIMEOUT; state:\n" + run("tail -5 " + RDIR + "/v8final_run.log; "
                                    "tail -3 " + RDIR
                                    + "/repro/mini_pipeline_v8.log"))
    sys.exit(3)

print("== mini log tail ==")
print(run("tail -4 " + RDIR + "/repro/mini_pipeline_v8.log"))
print(run("grep -c '^done' " + RDIR + "/repro/mini_pipeline_v8.log; "
          "grep FAIL " + RDIR + "/repro/mini_pipeline_v8.log; true"))
print(run("df -h /opt/m0 | tail -1"))

print("== tar + fetch ==")
print(run("cd " + RDIR + "/vec/mini && tar czf /tmp/outv8.tgz outv8_* && "
          "ls -la /tmp/outv8.tgz", t=300))
sftp = cli.open_sftp()
ldir = os.path.join(LROOT, "mini_eng_v8")
os.makedirs(ldir, exist_ok=True)
sftp.get("/tmp/outv8.tgz", os.path.join(ldir, "outv8.tgz"))
run("rm -f /tmp/outv8.tgz")
meta = os.path.join(LROOT, "mini_eng", "mini_meta.npz")
sftp.close()
cli.close()

import tarfile
tf = tarfile.open(os.path.join(ldir, "outv8.tgz"))
tf.extractall(ldir)
tf.close()
os.remove(os.path.join(ldir, "outv8.tgz"))
import shutil
shutil.copy(meta, os.path.join(ldir, "mini_meta.npz"))
n = len([d for d in os.listdir(ldir) if d.startswith("outv8_")])
tot = sum(len(os.listdir(os.path.join(ldir, d)))
          for d in os.listdir(ldir) if d.startswith("outv8_"))
print(f"extracted: {n} frame dirs, {tot} files total")
print("FETCH_DONE")

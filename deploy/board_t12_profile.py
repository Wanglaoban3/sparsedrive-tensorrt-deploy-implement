# -*- coding: utf-8 -*-
"""wait for e_T12 build, run trtexec profile with t6-identical flags,
fetch log locally, then run bucket comparison."""
import io
import os
import subprocess
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
L = r"<REPO>"
OUT = os.path.join(L, "deploy", "artifacts", "profile")
os.makedirs(OUT, exist_ok=True)

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=600):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    rc = out.channel.recv_exit_status()
    return out.read().decode("utf-8", "replace"), rc


for i in range(45):
    done = run("cat /opt/m0/trt-dev/repro/E_T12_DONE 2>/dev/null")[0].strip()
    if done:
        print(f"BUILD DONE: {done}")
        break
    if i % 5 == 0:
        print(f"[wait {i}m]")
    time.sleep(60)
else:
    print("build wait TIMEOUT")
    sys.exit(1)

print(run("tail -6 /opt/m0/trt-dev/repro/e_T12_build.log")[0])
print(run("ls -la /opt/m0/trt-dev/models/e_T12.engine")[0])

print("running trtexec profile (300 iters, ~2min)...")
txt, rc = run(
    "cd /opt/m0/trt-dev && /usr/src/tensorrt/bin/trtexec "
    "--loadEngine=models/e_T12.engine "
    "--plugins=/usr/local/lib/libdfaplug_v3.so --dumpProfile "
    "--exportProfile=repro/t12_profile.json --warmUp=200 "
    "--iterations=300 --avgRuns=10 --useSpinWait "
    "> repro/t12_prof.log 2>&1; echo rc=$?", t=900)
print(txt)
if "rc=0" not in txt:
    print("PROFILE RUN FAILED")
    print(run("tail -30 /opt/m0/trt-dev/repro/t12_prof.log")[0])
    sys.exit(1)

sftp = cli.open_sftp()
sftp.get("/opt/m0/trt-dev/repro/t12_prof.log",
         os.path.join(OUT, "t12_prof.log"))
sftp.close()
cli.close()
print(f"fetched t12_prof.log ({os.path.getsize(os.path.join(OUT, 't12_prof.log'))} B)")
subprocess.run([sys.executable, os.path.join(L, "deploy",
                                               "profile_buckets.py"),
                os.path.join(OUT, "t6_prof.log"),
                os.path.join(OUT, "t12_prof.log")])

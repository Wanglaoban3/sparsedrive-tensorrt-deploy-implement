# -*- coding: utf-8 -*-
"""probe v3: 清僵尸 → filesrc → sp_modelnode 2帧 --dump-img 2 → 拉 in_XX.bin."""
import os
import time

import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", "vecmeta")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace")


print("kill:", run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; "
                   "sleep 1; pgrep -af 'sp_filesrc|sp_modelnode' || echo none"))
BD = "/opt/m0/trt-dev/probe_out"
run("rm -rf /dev/shm/sp_m3 %s; mkdir -p %s" % (BD, BD))
fs = ("/usr/local/bin/sp_filesrc m3 1600 900 5 "
      "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
      "--fresh --wait-cons 1 > %s/fs.log 2>&1 < /dev/null &" % BD)
run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
nd = ("/usr/local/bin/sp_modelnode m3 /opt/m0/trt-dev/models/e_T6.engine "
      "/usr/local/lib/libdfaplug_v8.so "
      "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s "
      "--warmup 2 --frames 2 --img-from /opt/m0/trt-dev/vec/mini "
      "--dump-img 2 --no-dump > %s/node.log 2>&1 < /dev/null; "
      "echo rc=$? > %s/RC" % (BD, BD, BD))
run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
ok = False
for i in range(90):
    time.sleep(2)
    o = run("cat %s/RC 2>/dev/null" % BD)
    if o.strip().startswith("rc="):
        ok = True
        break
print("rc:", o.strip() if ok else "TIMEOUT")
print(run("tail -5 %s/node.log" % BD))
print(run("ls %s" % BD))
run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; true")

sftp = cli.open_sftp()
for k in (0, 1):
    sftp.get("%s/in_%02d.bin" % (BD, k),
             os.path.join(DST, "node_in_%02d.bin" % k))
sftp.close()
cli.close()
print("PROBE_FETCHED")

# -*- coding: utf-8 -*-
"""M8 门禁: 设备池零拷贝采集 vs mapped-shm 基线.
用法: python _m8_gate.py base|dma [fetch]
走板上 nv12_r0 真实 NV12 路径 (不走 --img-from, 那会绕过 preproc):
node --dump-img 2 落 img_XX.bin (preproc 输出), 两模式逐字节比对 +
pre_ms 对照. 节点 --graph --mp 全链, 帧数 = 板上 manifest 帧数."""
import os
import sys
import time

import paramiko

RUN = sys.argv[1] if len(sys.argv) > 1 else "base"
ONLY_FETCH = len(sys.argv) > 2 and sys.argv[2] == "fetch"
DMA_FS = " --dma" if RUN == "dma" else ""
DMA_ND = " --dma" if RUN == "dma" else ""
TAG = {"base": "m8base", "dma": "m8dma"}[RUN]
HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
BD = "/opt/m0/trt-dev/%s_out" % TAG

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


if not ONLY_FETCH:
    print("kill:", run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; "
                       "sleep 1; pgrep -af 'sp_filesrc|sp_modelnode' "
                       "|| echo none").strip())
    print(run("rm -rf /dev/shm/sp_m3 /tmp/sp_dma_m3.sock %s; mkdir -p %s"
              % (BD, BD)).strip())
    fs = ("/usr/local/bin/sp_filesrc m3 1600 900 5 "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
          "--fresh --wait-cons 1%s > %s/fs.log 2>&1 < /dev/null &"
          % (DMA_FS, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
    nd = ("/usr/local/bin/sp_modelnode m3 /opt/m0/trt-dev/models/e_bb2.engine "
          "/usr/local/lib/libdfaplug_v8.so "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s "
          "--hd /opt/m0/trt-dev/models/e_hd.engine "
          "--mp /opt/m0/trt-dev/models/e_mp.engine "
          "--warmup 2 --graph --dump-img 2%s "
          "> %s/node.log 2>&1 < /dev/null; "
          "echo rc=$? > %s/RC" % (BD, DMA_ND, BD, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
    o = ""
    ok = False
    for i in range(120):
        time.sleep(2)
        o = run("cat %s/RC 2>/dev/null" % BD)
        if o.strip().startswith("rc="):
            ok = True
            break
    print("rc:", o.strip() if ok else "TIMEOUT")
    print(run("grep -E 'dmapool|mode=|mailbox writes|pre |e2e|MODELNODE'"
              " %s/node.log | tail -12" % BD))
    run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; true")

os.makedirs(DST, exist_ok=True)
sftp = cli.open_sftp()
for f in ("img_00.bin", "img_01.bin", "frame_log.tsv", "node.log", "fs.log"):
    try:
        sftp.get("%s/%s" % (BD, f), os.path.join(DST, f))
    except IOError:
        print("MISS", f)
sftp.close()
cli.close()
print("M8_%s_DONE" % RUN.upper())

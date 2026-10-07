# -*- coding: utf-8 -*-
"""种子实验: 节点 legacy f0 状态 → 离线链 in_01, 从 f1 跑到 80.
本地推 8 个状态文件, 板端备份原 prev_* 后覆盖, 启动链 (marker 轮询)."""
import os
import sys
import time

import paramiko

HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3 = os.path.join(ROOT, "work_dirs", "preproc_ref", "m3")

ROT = [("next_det_feat", "prev_det_feat"),
       ("next_det_anchor", "prev_det_anchor"),
       ("next_det_conf", "prev_det_conf"),
       ("next_det_instance_id", "prev_det_id"),
       ("next_id_count", "prev_id_count"),
       ("next_map_feat", "prev_map_feat"),
       ("next_map_anchor", "prev_map_anchor"),
       ("next_map_conf", "prev_map_conf")]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=30):
    _, o, e = cli.exec_command(cmd, timeout=t)
    rc = o.channel.recv_exit_status()
    return rc, o.read().decode("utf-8", "replace"), e.read().decode(
        "utf-8", "replace")


sftp = cli.open_sftp()
IN1 = "/opt/m0/trt-dev/vec/mini/in_01"
rc, o, _ = run("mkdir -p %s/seed_bak && cp -n %s/prev_*.bin %s/seed_bak/ 2>/dev/null; ls %s/seed_bak | wc -l" % (IN1, IN1, IN1, IN1))
print("backup existing prev_*: %s files" % o.strip())
for src, dst in ROT:
    lp = os.path.join(M3, "out_00_%s.bin" % src)
    sftp.put(lp, "%s/%s.bin" % (IN1, dst))
    print("seeded %s <- %s (%d B)" % (dst, src, os.path.getsize(lp)))
sftp.close()

rc, o, _ = run("cat > /opt/m0/trt-dev/seed_chain.sh <<'EOF'"
               "\n#!/bin/bash\ncd /opt/m0/trt-dev\n"
               "bash repro/mini_pipeline_v8.sh 1 80\n"
               "echo SEED_CHAIN_DONE\nEOF", 10)
print("script written rc=%d" % rc)
rc, o, _ = run("rm -f /opt/m0/trt-dev/seed_done; "
               "(setsid nohup bash /opt/m0/trt-dev/seed_chain.sh "
               "> /opt/m0/trt-dev/seed_chain.log 2>&1 < /dev/null &); echo GO",
               10)
print("launched:", o.strip())
cli.close()
print("SEED_LAUNCHED")

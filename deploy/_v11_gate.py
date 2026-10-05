# -*- coding: utf-8 -*-
"""v11 插件闭环门禁: 编译 libdfaplug_v11.so -> 81 帧节点闭环 (bb2+hd+graph)
-> 拉回 eval 键 + frame_log. 引擎不重编 (插件按 名字/版本/命名空间 绑定).
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _v11_gate.py [fetch]
板上输出 /opt/m0/trt-dev/v11g1_out; 拉回 work_dirs/preproc_ref/v11g1."""
import os
import sys
import time

import paramiko

ONLY_FETCH = len(sys.argv) > 1 and sys.argv[1] == "fetch"
TAG = "v11g1"
HOST = os.environ["BOARD_HOST"]
PW = os.environ["BOARD_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
BD = "/opt/m0/trt-dev/%s_out" % TAG
KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
        "map_cls", "map_pts"]

ENG = "/opt/m0/trt-dev/models/e_bb2.engine"
HD = " --hd /opt/m0/trt-dev/models/e_hd.engine"
PLUGIN = "/usr/local/lib/libdfaplug_v11.so"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username="root", password=PW, timeout=15)


def run(cmd, t=90):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (("\n[stderr] " + err) if err.strip() else "")


w = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)

if not ONLY_FETCH:
    w.write("---- push + compile libdfaplug_v11.so ----\n")
    sftp0 = cli.open_sftp()
    try:
        sftp0.mkdir("/opt/m0/trt-dev/src")
    except IOError:
        pass
    sftp0.put(os.path.join(ROOT, "deploy", "dfaplug_v11.cu"),
              "/opt/m0/trt-dev/src/dfaplug_v11.cu")
    sftp0.close()
    r = run("cd /opt/m0/trt-dev/src && sed -i 's/\\r$//' dfaplug_v11.cu && "
            "nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC "
            "-o /usr/local/lib/libdfaplug_v11.so dfaplug_v11.cu -lnvinfer "
            "2>&1 | tail -6; echo SO_RC=${PIPESTATUS[0]}")
    w.write(r)
    if "SO_RC=0" not in r:
        w.write("SO_BUILD_FAILED\n")
        cli.close()
        raise SystemExit(1)
    w.write(run("ls -l /usr/local/lib/libdfaplug_v11.so; md5sum "
                "/usr/local/lib/libdfaplug_v11.so "
                "/usr/local/lib/libdfaplug_v8.so").strip() + "\n")

    w.write("---- 81-frame closed loop (v11 plugin) ----\n")
    # [c]/[e] 字符类防 pkill -f 自匹配: wrapper bash 的命令行含这些字样,
    # 裸 pattern 会把自己连同后续 mkdir 一起杀掉 (node.log 都不出现的根因)
    w.write(run("pkill -9 -f 'sp_filesr[c]'; pkill -9 -f 'sp_modelnod[e]'; "
                "sleep 1; rm -rf /dev/shm/sp_m3 /dev/shm/sp_res_sp_result_m3; "
                "rm -rf %s && mkdir -p %s && echo CLEAN_OK && df -h /opt/m0 "
                "| tail -1" % (BD, BD)).strip() + "\n")
    # SP_BUS_LEASE_MS=600000: 默认 5000ms lease 下, 文件 IO 停顿 (闪存 GC
    # 实测可达 ~7s) 会让发布端 force_stale 强收消费者槽 → seq 覆盖 → FATAL 12
    # (v11g1 首跑实测, watchdog 无 code=13 证明停顿在 kWsIo 文件段, 与插件
    # 无关)。调大 lease 后瞬态停顿只让发布端 blocked 等待, 真死锁仍有
    # 30x1s claim timeout → exit 20 兜底。
    fs = ("SP_BUS_LEASE_MS=600000 /usr/local/bin/sp_filesrc m3 1600 900 5 "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl /opt/m0/trt-dev/nv12_r0 "
          "--fresh --wait-cons 1 > %s/fs.log 2>&1 < /dev/null &" % BD)
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % fs)
    # 注意: HD 必须拼进 nd (AGENTS.md: 旗标没进 format 串 = 静默丢单引擎,
    # 症状 infer 15ms/dump 只有 Reshape_9/无 det_cls)。launch 后验日志指纹。
    nd = ("/usr/local/bin/sp_modelnode m3 %s %s "
          "/opt/m0/trt-dev/nv12_r0/manifest.jsonl %s%s --graph "
          "--warmup 2 --frames 81 --img-from /opt/m0/trt-dev/vec/mini "
          "> %s/node.log 2>&1 < /dev/null; "
          "echo rc=$? > %s/RC" % (ENG, PLUGIN, BD, HD, BD, BD))
    run("(setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &); echo GO" % nd)
    o = ""
    ok = False
    for i in range(240):  # 8 min 上限
        time.sleep(2)
        o = run("cat %s/RC 2>/dev/null" % BD)
        if o.strip().startswith("rc="):
            ok = True
            break
    w.write("rc: %s\n" % (o.strip() if ok else "TIMEOUT"))
    # 指纹校验: hd 引擎必须在跑 (hd 段非零), 否则数据作废
    fp = run("grep -c 'e_hd.engine' %s/node.log; grep -o 'hd_ms[^,]*' "
             "%s/frame_log.tsv 2>/dev/null | head -1; "
             "ls %s/out_00/ 2>/dev/null | grep -c det_cls" % (BD, BD, BD))
    w.write("fingerprint (hd-echo/hd-ms/det_cls-files): %s\n" % fp.strip())
    w.write(run("tail -14 %s/node.log" % BD).strip() + "\n")
    run("pkill -9 -f sp_filesrc; pkill -9 -f sp_modelnode; true")

os.makedirs(DST, exist_ok=True)
sftp = cli.open_sftp()
n = 0
for k in range(81):
    for nm in KEYS:
        r = "%s/out_%02d/%s.bin" % (BD, k, nm)
        try:
            sftp.get(r, os.path.join(DST, "out_%02d_%s.bin" % (k, nm)))
            n += 1
        except IOError:
            w.write("MISS %s\n" % r)
for f in ("frame_log.tsv", "node.log", "fs.log"):
    try:
        sftp.get("%s/%s" % (BD, f), os.path.join(DST, f))
    except IOError:
        pass
sftp.close()
cli.close()
w.write("FETCHED %d bins -> %s\n" % (n, DST))
w.write("V11_GATE_DONE\n")

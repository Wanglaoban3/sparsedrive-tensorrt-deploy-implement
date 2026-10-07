# -*- coding: utf-8 -*-
"""M-PROD B2 金标工具: gen / tamper / verify.

gen    帧0 全链张量采集 ×N (selftest-dump 模式, 与自检同一代码路径) →
       逐张量容差 = 3×跨run最大绝对偏差 (下限 1e-3, f16 噪声底; M7 定案
       跨 run 非 bit 确定, 严禁 hash) → 推板 /opt/m0/trt-dev/golden/<ring>/
       {frame0/*.bin, tol.txt, meta.txt(引擎/插件 md5 指纹)} + chmod 0444
tamper 破坏指定金标张量 4 字节 (FT: 篡改金标 → SELFTEST_FAIL 用例)
verify 板端逐张量 bin 尺寸清单 (诊断)

用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _prod_golden.py gen
          [--runs 10] [--ring m3] [--models-dir P] [--plugin P]
      python _prod_golden.py tamper det_cls [--ring m3]
前置: _prod_install.py 已安装且健康 (dump 模式 attach 真环, 不消费).
"""
import argparse
import os
import struct
import time

import numpy as np
import paramiko

RING = "m3"
GOLDEN_R = "/opt/m0/trt-dev/golden/%s" % RING
MODELS = "/opt/m0/trt-dev/models"
PLUGIN = "/usr/local/lib/libdfaplug_v8.so"
MANIFEST = "/opt/m0/trt-dev/nv12_r0/manifest.jsonl"
# 与节点 self_cmp 同清单 (sp_modelnode.cpp)
CMP_NAMES = ["det_cls", "det_quality", "det_bbox", "map_cls", "map_pts",
             "motion_cls", "plan_cls", "plan_reg", "plan_status"]
LOCAL = os.path.join("work_dirs", "golden_cal")

cli = None


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    try:
        return o.read().decode("utf-8", "replace") + \
            e.read().decode("utf-8", "replace"), 0
    except Exception:
        return "[channel-timeout] " + cmd, 1


def launch(cmd, log):
    inner = "(setsid nohup %s > %s 2>&1 < /dev/null &); echo GO" % (cmd, log)
    _, o, _ = cli.exec_command(inner, timeout=10)
    assert "GO" in o.read().decode("utf-8", "replace")


def wait_marker(marker, log, deadline_s):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        out, _ = run("grep -c '%s' %s 2>/dev/null; true" % (marker, log))
        if out.strip().startswith("1"):
            return True
        time.sleep(1)
    return False


def board_md5(path):
    out, _ = run("md5sum -b '%s'" % path)
    return out.split()[0].strip() if out.strip() else ""


def gen(runs):
    # dump 模式 attach 真环 (不消费不注册), 环不存在则前置失败
    out, _ = run("test -e /dev/shm/sp_%s && echo RING_OK" % RING)
    if "RING_OK" not in out:
        raise SystemExit("环 /dev/shm/sp_%s 不存在, 先 _prod_install.py" % RING)
    args = ("--hd %s/e_hd.engine --mp %s/e_mp.engine --warmup 2"
            % (MODELS, MODELS))
    for i in range(runs):
        d = "/tmp/gdump_%d" % i
        log = d + ".log"
        cmd = ("/usr/local/bin/sp_modelnode %s %s/e_bb2.engine %s %s %s %s"
               " --selftest-dump %s"
               % (RING, MODELS, PLUGIN, MANIFEST, d, args, d))
        run("rm -rf %s %s" % (d, log))
        launch(cmd, log)
        if not wait_marker("GOLDEN_DUMP_DONE", log, 180):
            print(run("tail -15 %s" % log)[0])
            raise SystemExit("run %d 未完成" % i)
        print("run %d/%d done" % (i + 1, runs))
    # 拉回
    sftp = cli.open_sftp()
    os.makedirs(LOCAL, exist_ok=True)
    data = {nm: [] for nm in CMP_NAMES}
    for i in range(runs):
        rd = "/tmp/gdump_%d/frame0" % i
        for nm in CMP_NAMES:
            lp = os.path.join(LOCAL, "run%d_%s.bin" % (i, nm))
            sftp.get("%s/%s.bin" % (rd, nm), lp)
            data[nm].append(np.fromfile(lp, dtype=np.float32))
    sftp.close()
    # 容差: 逐张量 3×跨run最大绝对偏差 vs run0, 下限 1e-3
    tol_lines = []
    for nm in CMP_NAMES:
        arr = np.stack(data[nm])
        dev = float(np.abs(arr - arr[0:1]).max())
        tol = max(3.0 * dev, 1e-3)
        tol_lines.append("%s %.6g" % (nm, tol))
        print("  %-12s maxdev=%.4g tol=%.4g" % (nm, dev, tol))
    # 指纹
    fps = {"bb2_md5": board_md5(MODELS + "/e_bb2.engine"),
           "main_md5": board_md5(MODELS + "/e_hd.engine"),
           "mp_md5": board_md5(MODELS + "/e_mp.engine"),
           "plugin_md5": board_md5(PLUGIN)}
    if not all(fps.values()):
        raise SystemExit("md5 失败: %s" % fps)
    meta = "".join("%s = %s\n" % (k, v) for k, v in fps.items())
    # 推板
    run("mkdir -p %s/frame0" % GOLDEN_R)
    sftp = cli.open_sftp()
    for nm in CMP_NAMES:
        sftp.put(os.path.join(LOCAL, "run0_%s.bin" % nm),
                 "%s/frame0/%s.bin" % (GOLDEN_R, nm))
    with sftp.open(GOLDEN_R + "/tol.txt", "w") as f:
        f.write("\n".join(tol_lines) + "\n")
    with sftp.open(GOLDEN_R + "/meta.txt", "w") as f:
        f.write(meta.replace("\r\n", "\n"))
    sftp.close()
    run("sed -i 's/\\r$//' %s/tol.txt %s/meta.txt; "
        "chmod 0444 %s/frame0/*.bin %s/tol.txt %s/meta.txt"
        % (GOLDEN_R, GOLDEN_R, GOLDEN_R, GOLDEN_R, GOLDEN_R))
    print("GOLDEN gen 完成: %s (%d runs)" % (GOLDEN_R, runs))
    print(meta, end="")


def tamper(name):
    p = "%s/frame0/%s.bin" % (GOLDEN_R, name)
    run("chmod 644 %s" % p)
    # 偏移 96 处写 NaN 位型 (7fc00000 小端 = 00 00 c0 7f)
    out, _ = run("printf '\\x00\\x00\\xc0\\x7f' | dd of=%s bs=1 seek=96 "
                 "conv=notrunc 2>&1; chmod 444 %s; md5sum %s" % (p, p, p))
    print("tamper %s -> %s" % (name, out.strip()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "tamper", "verify"])
    ap.add_argument("name", nargs="?", default="det_cls")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--plugin", default=PLUGIN)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["BOARD_HOST"], username="root",
                password=os.environ["BOARD_PASS"], timeout=15)
    a = ap.parse_args()
    PLUGIN = a.plugin  # gen() 引用全局; 金标指纹/固件路径随插件切换
    if a.mode == "gen":
        gen(a.runs)
    elif a.mode == "tamper":
        tamper(a.name)
    else:
        out, _ = run("ls -l %s/frame0/ 2>/dev/null; cat %s/tol.txt 2>/dev/null"
                     % (GOLDEN_R, GOLDEN_R))
        print(out)
    cli.close()

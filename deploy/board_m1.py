# -*- coding: utf-8 -*-
"""board_m1.py — M1 board bring-up: compile prepost on the Orin X, push the
R0 NV12 replay set, run filesrc->sub end-to-end, then crash injection.

Credentials come from the environment (never committed):
  BOARD_HOST / BOARD_PASS

Stages (run all, or pick with --stage):
  build     sftp push deploy/prepost sources; g++ on board; bins -> /usr/local/bin
  data      push work_dirs/nv12_r0 subset (--nframes, default 12) + manifest
  run       sp_filesrc @2Hz (loop) + sp_sub consumer; verify zero errors
  crash     kill -9 sub mid-hold; pub must force-recycle after lease; sub rejoins

Usage:
  python deploy/board_m1.py --stage build,data,run
  python deploy/board_m1.py --stage crash --nframes 12
"""
import argparse
import io
import json
import os
import sys
import time

import paramiko


def eval_json_line(ln):
    return json.loads(ln)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPOST = os.path.join(ROOT, "deploy", "prepost")
# /opt/m0 is noexec; binaries go to /usr/local/bin (allowed install target),
# sources + data live under /opt/m0/trt-dev/prepost.
BDIR = "/opt/m0/trt-dev/prepost"
NV12 = "/opt/m0/trt-dev/nv12_r0"

SRCS = ["sp_bus.h", "sp_bus.cpp", "image_source.h", "file_source.h",
        "file_source.cpp", "sp_pub.cpp", "sp_sub.cpp", "sp_inspect.cpp",
        "sp_filesrc.cpp", "sp_kernels.h", "sp_kernels.cu",
        "preproc.h", "preproc.cu", "sp_preproc_test.cpp",
        "postproc.h", "sp_result.h", "sp_resultmon.cpp",
        "sp_dmapool.h", "sp_dmapool.cpp", "sp_watch.h",
        "sp_modelnode.cpp"]


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["BOARD_HOST"], username="root",
                password=os.environ["BOARD_PASS"], timeout=15)
    return cli


def run(cli, cmd, t=300):
    _, out, err = cli.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    return rc, o, e


def launch(cli, cmd, log_path):
    """Detach a long-running process on the board and return immediately.

    Paren-wrap + setsid + absolute paths: `cd X && setsid ... & echo` keeps
    the wrapper bash in do_wait and the channel never EOFs; the subshell
    form orphans the job instantly.
    """
    inner = "(setsid nohup %s > %s 2>&1 < /dev/null &); echo LAUNCH_GO" % (
        cmd, log_path)
    _, out, _ = cli.exec_command(inner, timeout=10)
    o = out.read().decode("utf-8", "replace")
    assert "LAUNCH_GO" in o, "launch failed: %r" % o


def stage_build(cli):
    run(cli, "mkdir -p %s /usr/local/bin" % BDIR)
    sftp = cli.open_sftp()
    for f in SRCS:
        sftp.put(os.path.join(PREPOST, f), BDIR + "/" + f)
    sftp.close()
    print("pushed %d sources" % len(SRCS))
    # Orin has CUDA: build sp_sub with the cudaHostRegister path enabled
    # (mmap PROT_READ + one-shot register) and the M1.5 fence kernels
    # (.cu via nvcc, sm_87, module-level per compile discipline).
    build = r"""
set -e
cd {bd}
g++ -O2 -std=c++14 -Wall -Wextra -pthread -c sp_bus.cpp -o sp_bus.o
g++ -O2 -std=c++14 -Wall -Wextra -pthread -c sp_dmapool.cpp -o sp_dmapool.o \
    -I/usr/local/cuda/include
g++ -O2 -std=c++14 -Wall -Wextra -pthread sp_pub.cpp sp_bus.o -o sp_pub -lrt
g++ -O2 -std=c++14 -Wall -Wextra -pthread sp_inspect.cpp sp_bus.o -o sp_inspect -lrt
nvcc -O3 -arch=sm_87 -c sp_kernels.cu -o sp_kernels.o
nvcc -O3 -arch=sm_87 -c preproc.cu -o preproc.o
if [ -f /usr/local/cuda/include/cuda_runtime.h ]; then
  g++ -O2 -std=c++14 -Wall -Wextra -pthread sp_sub.cpp sp_bus.o sp_kernels.o \
      -o sp_sub -lrt -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
  echo SUB_CUDA=yes
else
  g++ -O2 -std=c++14 -Wall -Wextra -pthread -DSP_NO_CUDA sp_sub.cpp sp_bus.o \
      -o sp_sub -lrt
  echo SUB_CUDA=no
fi
g++ -O2 -std=c++14 -Wall -Wextra -pthread sp_filesrc.cpp file_source.cpp \
    sp_bus.o sp_dmapool.o -o sp_filesrc -lrt \
    -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda
g++ -O2 -std=c++14 sp_preproc_test.cpp preproc.o \
    -o sp_preproc_test -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart
g++ -O2 -std=c++14 -Wall -Wextra -pthread sp_resultmon.cpp sp_bus.o \
    -o sp_resultmon -lrt
g++ -O2 -std=c++14 sp_modelnode.cpp preproc.o sp_bus.o sp_dmapool.o \
    -o sp_modelnode -I/usr/local/cuda/include -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -lcuda -lnvinfer -ldl -lrt -lpthread
cp -f sp_pub sp_sub sp_inspect sp_filesrc sp_preproc_test sp_modelnode \
    sp_resultmon /usr/local/bin/
echo BUILD_DONE
""".format(bd=BDIR)
    rc, o, e = run(cli, build, t=600)
    print(o[-800:])
    if e.strip():
        print("stderr:", e[-800:])
    assert "BUILD_DONE" in o, "board build failed"


def stage_data(cli, nframes):
    local = os.path.join(ROOT, "work_dirs", "nv12_r0")
    manifest = os.path.join(local, "manifest.jsonl")
    with open(manifest) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    header = lines[0]
    frames = lines[1:1 + nframes]
    run(cli, "mkdir -p %s" % NV12)
    run(cli, "cd %s && for k in $(seq -f %%02g 0 %d); do mkdir -p frame_$k; done"
        % (NV12, nframes - 1))
    sftp = cli.open_sftp()
    with sftp.open(NV12 + "/manifest.jsonl", "w") as fp:
        fp.write(header + "\n" + "\n".join(frames) + "\n")
    t0 = time.time()
    total = 0
    for ln in frames:
        rec = eval_json_line(ln)
        for c, rel in enumerate(rec["cams"]):
            lp = os.path.join(local, rel.replace("/", os.sep))
            sftp.put(lp, NV12 + "/" + rel)
            total += 1
    sftp.close()
    print("pushed %d NV12 files in %.1fs" % (total, time.time() - t0))
    rc, o, _ = run(cli, "ls %s | head -3; du -sh %s" % (NV12, NV12))
    print(o)


def board_cleanup(cli, names=("sp_pub", "sp_sub", "sp_filesrc",
                              "sp_modelnode", "sp_preproc_test",
                              "tegrastats")):
    n = " ".join("pkill -9 -x %s 2>/dev/null;" % x for x in names)
    run(cli, n + " true")


def stage_run(cli, ring="m1"):
    board_cleanup(cli)
    cmd = ("/usr/local/bin/sp_filesrc {r} 1600 900 2 {nv}/manifest.jsonl "
           "{nv} --loop --fresh".format(r=ring, nv=NV12))
    run(cli, "rm -f %s/fs.log" % BDIR)
    launch(cli, cmd, BDIR + "/fs.log")
    time.sleep(2)
    # 2Hz source: 8 frames ≈ 4.5s of paced consumption
    rc, o, e = run(cli, "/usr/local/bin/sp_sub {r} 8 --hold-ms 8 "
                   "--wait-create; echo SUB_RC=$?".format(r=ring), t=180)
    print(o)
    if e.strip():
        print("stderr:", e[-400:])
    assert "crc_err=0" in o and "gaps=0" in o, "consumer saw errors"
    rc, o, _ = run(cli, "/usr/local/bin/sp_inspect {r}; tail -3 {b}/fs.log"
                   .format(r=ring, b=BDIR))
    print(o)
    board_cleanup(cli)
    print("RUN OK")


def stage_crash(cli, ring="m1c"):
    board_cleanup(cli)
    cmd = "/usr/local/bin/sp_pub {r} 1600 900 20 0 --fresh".format(r=ring)
    run(cli, "rm -f %s/pub.log %s/sub_a.log" % (BDIR, BDIR))
    launch(cli, cmd, BDIR + "/pub.log")
    time.sleep(1)
    # slow consumer, long run: hold 100ms so it is mid-hold at kill time
    launch(cli, "/usr/local/bin/sp_sub {r} 0 --hold-ms 100 --wait-create"
           .format(r=ring), BDIR + "/sub_a.log")
    time.sleep(4)
    rc, o, _ = run(cli, "pkill -9 -x sp_sub; echo KILLED")
    print(o.strip())
    time.sleep(7)  # lease default 5s
    rc, o, _ = run(cli, "/usr/local/bin/sp_sub {r} 10 --hold-ms 5; "
                   "echo REJOIN_RC=$?".format(r=ring), t=120)
    print(o)
    assert "crc_err=0" in o and "gaps=0" in o, "rejoin consumer saw errors"
    rc, o, _ = run(cli, "grep -c 'claim timeout' {b}/pub.log; "
                   "/usr/local/bin/sp_inspect {r}".format(b=BDIR, r=ring))
    print(o)
    assert "forced=1" in o or "forced=2" in o, \
        "publisher never force-recycled the dead consumer's ref"
    board_cleanup(cli)
    print("CRASH OK (lease steal verified: forced recycle + clean rejoin)")


def stage_fence(cli, ring="m1f"):
    """M1.5 fence 验证:消费端持引用直到 kernel 完成事件 → 发布端永不覆写。
    fence 必须 fence_err=0。race 模式(发射即释放)在本板 nvgpu 上是
    GPU 读-发布端写并发 → 通道 fault 整进程崩,本身就是危害的最强演示,
    故不作 clean-mismatch 断言,只在报告里注明。"""
    board_cleanup(cli)
    cmd = "/usr/local/bin/sp_pub {r} 1600 900 300 0 --fresh".format(r=ring)
    launch(cli, cmd, BDIR + "/pub_f.log")
    time.sleep(0.5)
    rc, fence_out, _ = run(cli, "/usr/local/bin/sp_sub {r} 20 --fence; "
                           "echo FENCE_RC=$?".format(r=ring), t=180)
    print(fence_out)
    board_cleanup(cli)
    fence_done = [l for l in fence_out.splitlines() if "done" in l]
    assert fence_done and "fence_err=0" in fence_done[-1], \
        "fence mode MUST have zero mismatches (got %s)" % (
            fence_done[-1] if fence_done else "no done line")
    print("FENCE OK (零不一致; race 对照在本板表现为 GPU 通道 fault, "
          "见 docs 记录)")


def stage_m2(cli):
    """M2 板端数值验证: sp_preproc_test 跑帧 0/1/40 → 拉回本地逐像素对比."""
    import json
    # v2 manifest + 帧 0/1/40 数据就位 (旧板上可能是 v1 子集)
    run(cli, "mkdir -p %s/frame_01 %s/frame_40" % (NV12, NV12))
    sftp = cli.open_sftp()
    local_r0 = os.path.join(ROOT, "work_dirs", "nv12_r0")
    sftp.put(os.path.join(local_r0, "manifest.jsonl"),
             NV12 + "/manifest.jsonl")
    for k in (1, 40):
        for c in range(6):
            rel = "frame_%02d/cam_%d.nv12" % (k, c)
            sftp.put(os.path.join(local_r0, rel.replace("/", os.sep)),
                     NV12 + "/" + rel)
    sftp.close()
    print("pushed v2 manifest + frames 1,40")
    rc, o, e = run(cli, "/usr/local/bin/sp_preproc_test %s/manifest.jsonl "
                   "%s 0 %s/pre_00.bin; echo RC=$?"
                   % (NV12, NV12, BDIR), t=300)
    print(o)
    if "PREPROC_TEST_DONE" not in o:
        print("stderr:", e[-800:])
        raise SystemExit("preproc test frame 0 failed")
    for k in (1, 40):
        rc, o, e = run(cli, "/usr/local/bin/sp_preproc_test %s/manifest.jsonl "
                       "%s %d %s/pre_%02d.bin; echo RC=$?"
                       % (NV12, NV12, k, BDIR, k), t=300)
        print(o[-200:])
        assert "PREPROC_TEST_DONE" in o, "frame %d failed" % k
    # 拉回
    sftp = cli.open_sftp()
    local = os.path.join(ROOT, "work_dirs", "preproc_ref")
    os.makedirs(local, exist_ok=True)
    for k in (0, 1, 40):
        sftp.get("%s/pre_%02d.bin" % (BDIR, k),
                 os.path.join(local, "board_%02d.bin" % k))
    sftp.close()
    print("fetched 3 preproc outputs -> work_dirs/preproc_ref/")


def stage_m3(cli, nframes=81, warmup=2, img_from=None, graph=False,
             serial=False, fetch="full", fps=5, bb=None, hd=None, mp=None):
    """M3/M4: node 双缓冲流水线闭环; filesrc --wait-cons 等引擎就绪才开拍
    (5fps 慢于 node 稳态 → 不超圈不丢帧), node 按清单序消费 → 拉回输出 +
    frame_log.tsv (分级时间戳, 本地算 p50/p90/p99).
    nframes > 81 自动 filesrc --loop + node --loop (清单回绕, 稳定性压测).
    fetch: full = 每帧评测张量 + 代表帧时序张量; log = 仅日志/时间戳
    (--no-dump, 吞吐不受落盘影响, 用于性能与稳定性跑).
    M5/M6: --bb e_bb2 --hd e_hd (拆分链), --mp e_mp (二级引擎, 另拉 outm)."""
    loop = nframes > 81
    board_cleanup(cli)
    run(cli, "rm -rf %s/m3out; rm -f %s/m3node.log %s/fs3.log %s/tegra.log"
             " /dev/shm/sp_m3" % (BDIR, BDIR, BDIR, BDIR))
    # filesrc 先起建环, --wait-cons 1 等消费端注册才开拍
    # (fps=0 → 不限速发布, 背压节流 → 测节点吞吐上限)
    launch(cli, "/usr/local/bin/sp_filesrc m3 1600 900 {fps} "
           "{nv}/manifest.jsonl {nv} --fresh --wait-cons 1{lp}".format(
               nv=NV12, fps=fps, lp=" --loop" if loop else ""), BDIR + "/fs3.log")
    launch(cli, "/usr/bin/tegrastats --interval 200", BDIR + "/tegra.log")
    eng_arg = bb if bb else "/opt/m0/trt-dev/models/e_T6.engine"
    nodecmd = ("/usr/local/bin/sp_modelnode m3 "
               "{eng} "
               "/usr/local/lib/libdfaplug_v8.so {nv}/manifest.jsonl "
               "{b}/m3out --warmup {w} --frames {n}{g}{s}{nd}{lp}{img}"
               "{hdp}{mpp}".format(
                   b=BDIR, nv=NV12, n=nframes, w=warmup, eng=eng_arg,
                   g=" --graph" if graph else "",
                   s=" --serial" if serial else "",
                   nd=" --no-dump" if fetch == "log" else "",
                   lp=" --loop" if loop else "",
                   img=" --img-from " + img_from if img_from else "",
                   hdp=" --hd " + hd if hd else "",
                   mpp=" --mp " + mp if mp else ""))
    if os.environ.get("SP_ONE_STREAM"):
        nodecmd = "env SP_ONE_STREAM=1 " + nodecmd
    launch(cli, nodecmd, BDIR + "/m3node.log")
    # 等 node 完成 (超时随帧数伸缩: 5fps 节奏 ≈ nframes/5 s, 留 8 倍余量)
    budget = max(600, nframes * 8 / 5)
    for _ in range(int(budget)):
        time.sleep(1)
        _, o, _ = run(cli, "grep -c MODELNODE_DONE %s/m3node.log 2>/dev/null"
                      % BDIR)
        if o.strip().startswith("1"):
            break
    else:
        _, o, _ = run(cli, "tail -20 %s/m3node.log; tail -5 %s/fs3.log"
                      % (BDIR, BDIR))
        print(o)
        raise SystemExit("modelnode did not finish in time")
    _, o, _ = run(cli, "tail -12 %s/m3node.log; tail -3 %s/fs3.log"
                  % (BDIR, BDIR))
    print(o)
    board_cleanup(cli)
    # 拉回: 日志/时间戳总是拉; fetch=full 再拉张量
    sftp = cli.open_sftp()
    local = os.path.join(ROOT, "work_dirs", "preproc_ref", "m3")
    os.makedirs(local, exist_ok=True)
    for lf in ("frame_log.tsv", "result.jsonl", "m3node.log", "fs3.log",
               "tegra.log"):
        remote = "%s/m3out/%s" % (BDIR, lf) if lf in ("frame_log.tsv",
                                                      "result.jsonl") \
            else "%s/%s" % (BDIR, lf)
        try:
            sftp.get(remote, os.path.join(local, lf))
        except IOError:
            print("warn: missing %s" % remote)
    n_fetch = 0
    if fetch == "full":
        key = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
               "map_cls", "map_pts"]
        deep = ["next_det_feat", "next_det_anchor", "next_det_conf",
                "next_map_feat", "next_map_anchor", "next_map_conf",
                "det_instance_id", "next_det_instance_id", "next_id_count"]
        deep_frames = {0, 1, 39, 40, 41, 80}
        for k in range(nframes):
            rd = "%s/m3out/out_%02d" % (BDIR, k)
            names = list(key) + (deep if k in deep_frames else [])
            for nm in names:
                sftp.get("%s/%s.bin" % (rd, nm),
                         os.path.join(local, "out_%02d_%s.bin" % (k, nm)))
                n_fetch += 1
        if mp:
            # M6a: mp 侧全量拉回 (对拍同源参考 + eval_mp_mini)
            for k in range(nframes):
                rd = "%s/m3out/outm_%02d" % (BDIR, k)
                try:
                    names = [f[:-4] for f in sftp.listdir(rd)
                             if f.endswith(".bin")]
                except IOError:
                    print("warn: missing outm_%02d" % k)
                    continue
                for nm in names:
                    sftp.get("%s/%s.bin" % (rd, nm),
                             os.path.join(local, "outm_%02d_%s.bin" % (k, nm)))
                    n_fetch += 1
    sftp.close()
    print("fetched %d tensors + logs -> work_dirs/preproc_ref/m3/" % n_fetch)
    board_cleanup(cli)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="build,data,run")
    ap.add_argument("--nframes", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--img-from", default=None,
                    help="跳过前处理, 直接 H2D 链路二 img (目录根/模板/文件)")
    ap.add_argument("--graph", action="store_true",
                    help="M4: 推理+反馈捕获 CUDA graph (按 parity 两张)")
    ap.add_argument("--serial", action="store_true",
                    help="M4: 串行模式 (M3 基线, 无重叠)")
    ap.add_argument("--fetch", default="full", choices=["full", "log"],
                    help="m3 拉回内容: full=张量, log=仅日志/时间戳")
    ap.add_argument("--fps", type=float, default=5,
                    help="filesrc 发布节奏 fps; 0=不限速(背压节流, 测吞吐)")
    ap.add_argument("--bb", default=None,
                    help="M5: e_bb2 引擎路径 (argv[2]; 需配 --hd)")
    ap.add_argument("--hd", default=None,
                    help="M5: e_hd 引擎路径 (拆分链)")
    ap.add_argument("--mp", default=None,
                    help="M6a: e_mp 二级引擎路径")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    cli = connect()
    try:
        for s in stages:
            print("== stage:", s)
            if s == "build":
                stage_build(cli)
            elif s == "data":
                stage_data(cli, args.nframes)
            elif s == "run":
                stage_run(cli)
            elif s == "crash":
                stage_crash(cli)
            elif s == "fence":
                stage_fence(cli)
            elif s == "m2":
                stage_m2(cli)
            elif s == "m3":
                stage_m3(cli, args.nframes, args.warmup, args.img_from,
                         args.graph, args.serial, args.fetch, args.fps,
                         args.bb, args.hd, args.mp)
            else:
                raise SystemExit("unknown stage " + s)
    finally:
        cli.close()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Motion/planning board campaign: build fp16 e_mp from the temporal graph
v5_mp.onnx only (single recurrent engine, zero-state reset at scene starts —
same convention as the perception reset chain), run the reset chain (e_T6)
then the second-stage mp chain over the 81 mini frames, pull mp dumps for
local eval.

Stages: push,tool,engine,chainr,chainmp,prof,fetch (default in order)
  push    onnx graph + mp zero state + repro/mini_pipeline_mp.sh
  tool    build onnx2engine_f32 (no --f32-names used here; plain fp16 build)
  engine  models/e_mp.engine (--fp16 only)
  chainr  e_T6 + reset chain -> vec/mini_r/outv8_XX (engine-1 side)
  chainmp mp chain -> vec/mini_m/outm_XX
  prof    trtexec e_mp profile
  fetch   pull outm_XX + engine-1 det tensors -> evaldata/mini_mp_eng
          (merged out_XX layout, directly eval_mp_mini.py-compatible)

Usage (credentials runtime-injected only, never stored):
  set BOARD_HOST=...&& set BOARD_PASS=...&& python deploy/board_mp.py --stage push,tool,engine
  set BOARD_HOST=...&& set BOARD_PASS=...&& python deploy/board_mp.py --stage chainr,chainmp,fetch
"""
import argparse
import io
import os
import shutil
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "/opt/m0/trt-dev"
MODS = DEV + "/mods"
SRC_B = DEV + "/src/onnx2engine_f32.cpp"
BIN_B = "/usr/local/bin/onnx2engine_f32"
CHAIN_R = DEV + "/repro/mini_pipeline_reset.sh"
CHAIN_MP = DEV + "/repro/mini_pipeline_mp.sh"
ONNX_R = MODS + "/v5_mp.onnx"
ENG_R = DEV + "/models/e_mp.engine"
OUT_R = DEV + "/vec/mini_r"
OUT_M = DEV + "/vec/mini_m"
STATE0 = DEV + "/vec/mini_mpstate0"
PLUGIN = "/usr/local/lib/libdfaplug_v8.so"
E_T6 = DEV + "/models/e_T6.engine"

LOCAL_ONNX_R = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                            "v5_mp.onnx")
LOCAL_CPP = os.path.join(ROOT, "deploy", "onnx2engine.cpp")
LOCAL_CHAIN_R = os.path.join(ROOT, "repro", "mini_pipeline_reset.sh")
LOCAL_CHAIN_MP = os.path.join(ROOT, "repro", "mini_pipeline_mp.sh")
WD = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2")
PULL_V8R = os.path.join(WD, "evaldata", "mini_eng_v8r")
PULL_MP = os.path.join(WD, "evaldata", "mini_mp_eng")
META_SRC = os.path.join(WD, "evaldata", "mini_eng", "mini_meta.npz")

# mp queue zero state (first-graph frame semantics; first graph ignores the
# values but the bins must exist with the right shapes)
MP_ZERO = [
    ("history_instance_feature", "f32", (1, 900, 4, 256)),
    ("history_anchor", "f32", (1, 900, 4, 11)),
    ("history_period", "i32", (1, 900)),
    ("prev_instance_id", "i32", (1, 900), -1),
    ("prev_confidence", "f32", (1, 900)),
    ("history_ego_feature", "f32", (1, 1, 4, 256)),
    ("history_ego_anchor", "f32", (1, 1, 4, 11)),
    ("history_ego_period", "i32", (1, 1)),
    ("prev_ego_status", "f32", (1, 1, 10)),
]

BUILD_SH = r"""
set -e
cd {dev}
g++ -O2 -o /usr/local/bin/onnx2engine_f32 src/onnx2engine_f32.cpp \
  -I include -I /usr/local/cuda/include \
  -L /usr/local/cuda/lib64 -lnvinfer -lnvonnxparser -lcudart -ldl
echo TOOL_BUILD_OK
"""

ENGINE_SH = r"""
set -e
cd {dev}
rm -f mods/BUILD_MP_DONE models/e_mp.engine mods/e_mp_first.engine \
  mods/v5_mp_first.onnx mods/build_mp_first.log
(setsid nohup bash -c '
  /usr/local/bin/onnx2engine_f32 mods/v5_mp.onnx \
    models/e_mp.engine --fp16 --ws-mb 2048 \
    > mods/build_mp.log 2>&1
  rc=$?
  echo "rc=$rc" > mods/BUILD_MP_DONE
' > /dev/null 2>&1 < /dev/null &) ;
echo ENGINE_LAUNCHED
"""


def connect():
    import paramiko
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


def wait_log(cli, log, done_substr, deadline_s, step=30):
    t0 = time.time()
    last = ""
    while time.time() - t0 < deadline_s:
        rc, o, _ = run(cli, "tail -c 500 %s 2>/dev/null" % log)
        tail = o.strip().replace("\n", " | ")[-170:]
        if tail != last:
            print("  [%5ds] %s" % (time.time() - t0, tail), flush=True)
            last = tail
        if done_substr in o:
            return True
        time.sleep(step)
    return False


def _zero_files(tmp):
    puts = []
    for spec in MP_ZERO:
        nm, dt, shape = spec[0], spec[1], spec[2]
        if dt == "i32" and len(spec) > 3:
            arr = np.full(shape, spec[3], np.int32)
        else:
            arr = np.zeros(shape, np.int32 if dt == "i32" else np.float32)
        p = os.path.join(tmp, nm + ".bin")
        arr.tofile(p)
        puts.append(p)
    return puts


def stage_push(cli):
    run(cli, "mkdir -p %s %s %s %s" % (MODS, OUT_R, OUT_M, STATE0))
    sftp = cli.open_sftp()
    sftp.put(LOCAL_ONNX_R, ONNX_R)
    sftp.put(LOCAL_CHAIN_R, CHAIN_R)
    sftp.put(LOCAL_CHAIN_MP, CHAIN_MP)
    sftp.close()
    run(cli, "sed -i 's/\\r//' %s %s" % (CHAIN_R, CHAIN_MP))
    import tempfile
    tmp = tempfile.mkdtemp()
    puts = _zero_files(tmp)
    sftp = cli.open_sftp()
    for p in puts:
        sftp.put(p, "%s/%s" % (STATE0, os.path.basename(p)))
    sftp.close()
    shutil.rmtree(tmp, ignore_errors=True)
    rc, o, _ = run(cli, "ls -la %s | tail -3; ls %s/*.onnx; "
                        "ls %s | wc -l" % (STATE0, MODS, STATE0))
    print(o)


def stage_tool(cli):
    run(cli, "mkdir -p %s/src" % DEV)
    sftp = cli.open_sftp()
    sftp.put(LOCAL_CPP, SRC_B)
    sftp.close()
    run(cli, "sed -i 's/\\r//' %s" % SRC_B)
    rc, o, e = run(cli, BUILD_SH.format(dev=DEV), t=600)
    print(o[-800:], e[-400:] if rc else "")
    assert "TOOL_BUILD_OK" in o, "tool build failed"
    rc, o, _ = run(cli, BIN_B + " 2>&1 | head -2")
    print("tool smoke:", o.strip().splitlines()[:2])


def stage_engine(cli):
    rc, o, e = run(cli, ENGINE_SH.format(dev=DEV))
    assert "ENGINE_LAUNCHED" in o, "engine launch failed: %r %r" % (o, e)
    ok = wait_log(cli, MODS + "/BUILD_MP_DONE", "rc=", 3600)
    if not ok:
        rc, o, _ = run(cli, "grep -iE 'error|assert' %s/build_mp.log "
                           "| tail -10" % MODS)
        print("BUILD PROBLEM:\n", o)
        raise SystemExit("e_mp build did not finish; see log tail above")
    rc, o, _ = run(cli, "cat %s/BUILD_MP_DONE; tail -3 %s/build_mp.log"
                   % (MODS, MODS))
    print(o)
    assert "rc=0" in o, "e_mp build failed"
    rc, o, _ = run(cli, "ls -la %s; md5sum %s" % (ENG_R, ENG_R))
    print(o)


def stage_chainr(cli):
    run(cli, "rm -rf %s; mkdir -p %s" % (OUT_R, OUT_R))
    cmd = ("(setsid nohup bash %s models/e_T6.engine "
           "%s %s 0 80 40 "
           "> %s/chainr.log 2>&1 < /dev/null &); echo GO"
           % (CHAIN_R, PLUGIN, OUT_R, MODS))
    _, out, _ = cli.exec_command(cmd, timeout=10)
    assert "GO" in out.read().decode(), "chainr launch failed"
    wait_log(cli, DEV + "/repro/mini_reset_mini_r.log",
             "RESET_CHAIN_DONE", 2400)
    rc, o, _ = run(cli, "tail -3 %s; ls %s | wc -l; "
                        "ls %s/outv8_00 | wc -l"
                   % (DEV + "/repro/mini_reset_mini_r.log", OUT_R, OUT_R))
    print(o)
    assert "RESET_CHAIN_DONE" in o, "chainr did not finish"


def stage_chainmp(cli):
    run(cli, "rm -rf %s; mkdir -p %s" % (OUT_M, OUT_M))
    cmd = ("(setsid nohup bash %s models/e_mp.engine "
           "%s %s %s 0 80 \"0 40\" "
           "> %s/chainmp.log 2>&1 < /dev/null &); echo GO"
           % (CHAIN_MP, PLUGIN, OUT_R, OUT_M, MODS))
    _, out, _ = cli.exec_command(cmd, timeout=10)
    assert "GO" in out.read().decode(), "chainmp launch failed"
    wait_log(cli, DEV + "/repro/mini_mp_mini_m.log", "MP_CHAIN_DONE", 2400)
    rc, o, _ = run(cli, "tail -3 %s; ls %s | wc -l; "
                        "ls %s/outm_00 | wc -l"
                   % (DEV + "/repro/mini_mp_mini_m.log", OUT_M, OUT_M))
    print(o)
    assert "MP_CHAIN_DONE" in o, "chainmp did not finish"


def stage_prof(cli):
    rc, o, _ = run(cli,
                   "/usr/src/tensorrt/bin/trtexec "
                   "--loadEngine=%s/models/e_mp.engine "
                   "--iterations=100 2>&1 | grep -E 'GPU Compute Time|mean|median' "
                   "| head -6" % DEV, t=900)
    print("== e_mp ==\n%s" % o)


def _pull_dir(cli, remote_dir, local_dir, prefix, only=None):
    sftp = cli.open_sftp()
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for k in range(81):
        rd = "%s/%s_%02d" % (remote_dir, prefix, k)
        ld = os.path.join(local_dir, "%s_%02d" % (prefix, k))
        os.makedirs(ld, exist_ok=True)
        try:
            names = [f for f in sftp.listdir(rd) if f.endswith(".bin")]
        except IOError:
            print("MISSING", rd)
            continue
        if only:
            names = [f for f in names if f in only]
        for f in names:
            sftp.get("%s/%s" % (rd, f), os.path.join(ld, f))
            n += 1
    sftp.close()
    return n


def stage_fetch(cli):
    det_side = {"det_cls.bin", "det_bbox.bin", "det_quality.bin",
                "det_instance_id.bin"}
    n1 = _pull_dir(cli, OUT_R, PULL_V8R, "outv8", only=det_side)
    n2 = _pull_dir(cli, OUT_M, PULL_MP, "outm")
    print("pulled %d + %d files" % (n1, n2))
    # merge into eval_mp_mini.py layout: out_XX = engine-1 det side + mp side
    merged = PULL_MP.replace("mini_mp_eng", "mini_mp_eng")
    for k in range(81):
        md = os.path.join(PULL_MP, "out_%02d" % k)
        os.makedirs(md, exist_ok=True)
        v8 = os.path.join(PULL_V8R, "outv8_%02d" % k)
        mp = os.path.join(PULL_MP, "outm_%02d" % k)
        for f in ("det_cls.bin", "det_bbox.bin", "det_quality.bin"):
            src = os.path.join(v8, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(md, f))
        src = os.path.join(v8, "det_instance_id.bin")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(md, "det_id.bin"))
        if os.path.isdir(mp):
            for f in os.listdir(mp):
                shutil.copy(os.path.join(mp, f), os.path.join(md, f))
    shutil.copy(META_SRC, os.path.join(PULL_MP, "mini_meta.npz"))
    rc, o, _ = run(cli, "md5sum %s" % ENG_R)
    print(o)
    logs = [(DEV + "/repro/mini_reset_mini_r.log", "mini_reset_mini_r.log"),
            (DEV + "/repro/mini_mp_mini_m.log", "mini_mp_mini_m.log"),
            (MODS + "/build_mp.log", "build_mp.log")]
    sftp = cli.open_sftp()
    os.makedirs(os.path.join(ROOT, "deploy", "artifacts"), exist_ok=True)
    for rp, ln in logs:
        try:
            sftp.get(rp, os.path.join(ROOT, "deploy", "artifacts", ln))
        except IOError:
            print("log missing:", rp)
    sftp.close()
    print("fetch done; eval with: python deploy\\eval_mp_mini.py "
          "--eng work_dirs\\sparsedrive_small_stage2\\evaldata\\mini_mp_eng")


STAGES = {
    "push": stage_push,
    "tool": stage_tool,
    "engine": stage_engine,
    "chainr": stage_chainr,
    "chainmp": stage_chainmp,
    "prof": stage_prof,
    "fetch": stage_fetch,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    default="push,tool,engine,chainr,chainmp,prof,fetch")
    args = ap.parse_args()
    cli = connect()
    try:
        for s in args.stage.split(","):
            s = s.strip()
            if not s:
                continue
            print("=== stage %s ===" % s, flush=True)
            STAGES[s](cli)
    finally:
        cli.close()


if __name__ == "__main__":
    main()

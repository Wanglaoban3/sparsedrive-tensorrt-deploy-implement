# -*- coding: utf-8 -*-
"""Map-head accuracy fix campaign on the board (R9/R10, see
docs/OPTIMIZATION_SUMMARY.md round 9).

Two orthogonal fixes, one board session:
  F1 (zero-compile) chain scene reset: repro/mini_pipeline_reset.sh skips
     the prev_* blind copy at scene-start frames and restores in_40 zero
     history first. Run with the DELIVERED e_T6 -> dump vec/mini_r/outv8_XX.
  F2 (one rebuild) map-head FP32: onnx2engine --f32-names (map-side layer
     name list from deploy/_gen_f32_names.py, fp16-region/backbone kept)
     builds models/e_T6m.engine, then reset chain -> vec/mini_m/outv8_XX.
Plus trtexec profile of e_T6m for the speed cost, then fetch both dumps.

Stages: push,tool,chainr,engine,chainm,prof,fetch  (default runs in order)

Usage (credentials runtime-injected only, never stored):
  set BOARD_HOST=...&& set BOARD_PASS=...&& python deploy/board_mapfix.py --stage push,tool
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
LIST_B = MODS + "/map_f32_names.txt"
SRC_B = DEV + "/src/onnx2engine_f32.cpp"
BIN_B = "/usr/local/bin/onnx2engine_f32"
CHAIN_B = DEV + "/repro/mini_pipeline_reset.sh"
ENG_M = DEV + "/models/e_T6m.engine"
OUT_R = DEV + "/vec/mini_r"
OUT_M = DEV + "/vec/mini_m"

LOCAL_CPP = os.path.join(ROOT, "deploy", "onnx2engine.cpp")
LOCAL_LIST = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                          "map_f32_names.txt")
LOCAL_CHAIN = os.path.join(ROOT, "repro", "mini_pipeline_reset.sh")
# pulled dumps land here (outv8_XX prefix inside, map eval compatible)
PULL_R = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                      "evaldata", "mini_eng_v8r")
PULL_M = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                      "evaldata", "mini_eng6m")
META_SRC = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                        "evaldata", "mini_eng", "mini_meta.npz")

# zero history for a scene-start frame (prep_mini_inputs semantics)
ZERO_FILES = [
    ("prev_det_feat", "f32", (1, 600, 256)),
    ("prev_det_anchor", "f32", (1, 600, 11)),
    ("prev_det_conf", "f32", (1, 600)),
    ("prev_det_id", "i32", (1, 600), -1),
    ("prev_id_count", "i32", (1, 1)),
    ("prev_map_feat", "f32", (1, 33, 256)),
    ("prev_map_anchor", "f32", (1, 33, 40)),
    ("prev_map_conf", "f32", (1, 33)),
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
rm -f mods/BUILD_T6M_DONE
(setsid nohup bash -c '
  env TRT_LOG_LEVEL=VERBOSE /usr/local/bin/onnx2engine_f32 \
    models/v5_P1h.onnx models/e_T6m.engine \
    --int8 --fp16 --ws-mb 2048 \
    --f32-names mods/map_f32_names.txt \
    --plugins /usr/local/lib/libdfaplug_v8.so \
    > mods/build_t6m.log 2>&1 < /dev/null
  echo done > mods/BUILD_T6M_DONE
' > /dev/null 2>&1 < /dev/null &); echo ENGINE_LAUNCHED
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
    """Wait until log tail contains done_substr; print tail changes."""
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


def stage_push(cli):
    run(cli, "mkdir -p %s %s %s" % (MODS, OUT_R, OUT_M))
    sftp = cli.open_sftp()
    sftp.put(LOCAL_CPP, SRC_B)
    sftp.put(LOCAL_LIST, LIST_B)
    sftp.put(LOCAL_CHAIN, CHAIN_B)
    sftp.close()
    # CRLF insurance (files written on Windows)
    run(cli, "sed -i 's/\\r//' %s %s" % (CHAIN_B, LIST_B))
    # restore scene-start zero history (in_40 was blind-overwritten)
    import tempfile
    tmp = tempfile.mkdtemp()
    puts = []
    for spec in ZERO_FILES:
        nm, dt, shape = spec[0], spec[1], spec[2]
        arr = (np.full(shape, -1, np.int32) if dt == "i32" and len(spec) > 3
               else np.zeros(shape, np.int32 if dt == "i32" else np.float32))
        p = os.path.join(tmp, nm + ".bin")
        arr.tofile(p)
        puts.append(p)
    sftp = cli.open_sftp()
    for p in puts:
        sftp.put(p, "%s/vec/mini/in_40/%s" % (DEV, os.path.basename(p)))
    sftp.close()
    shutil.rmtree(tmp, ignore_errors=True)
    rc, o, _ = run(cli, "ls -la %s/vec/mini/in_40/prev_map_conf.bin "
                        "&& wc -l %s" % (DEV, LIST_B))
    print(o)


def stage_tool(cli):
    rc, o, e = run(cli, BUILD_SH.format(dev=DEV), t=600)
    print(o[-800:], e[-400:] if rc else "")
    assert "TOOL_BUILD_OK" in o, "tool build failed"
    rc, o, _ = run(cli, BIN_B + " 2>&1 | head -2")
    print("tool smoke:", o.strip().splitlines()[:2])


def stage_chainr(cli):
    # delivered engine + reset chain
    run(cli, "rm -rf %s; mkdir -p %s" % (OUT_R, OUT_R))
    cmd = ("(setsid nohup bash %s models/e_T6.engine "
           "/usr/local/lib/libdfaplug_v8.so %s 0 80 40 "
           "> %s/chain.log 2>&1 < /dev/null &); echo GO" % (CHAIN_B, OUT_R, MODS))
    _, out, _ = cli.exec_command(cmd, timeout=10)
    assert "GO" in out.read().decode(), "chainr launch failed"
    wait_log(cli, DEV + "/repro/mini_reset_mini_r.log", "RESET_CHAIN_DONE", 2400)
    rc, o, _ = run(cli, "tail -3 %s/repro/mini_reset_mini_r.log; "
                       "ls %s | wc -l; ls %s/outv8_00 | wc -l"
                   % (DEV, OUT_R, OUT_R))
    print(o)
    assert "RESET_CHAIN_DONE" in o, "chainr did not finish"
    # defensive: zero history back (in_40 untouched by the run, but keep
    # standalone-stage runs safe)
    stage_push_zeros_only(cli)


def stage_push_zeros_only(cli):
    import tempfile
    tmp = tempfile.mkdtemp()
    puts = []
    for spec in ZERO_FILES:
        nm, dt, shape = spec[0], spec[1], spec[2]
        arr = (np.full(shape, -1, np.int32) if dt == "i32" and len(spec) > 3
               else np.zeros(shape, np.int32 if dt == "i32" else np.float32))
        p = os.path.join(tmp, nm + ".bin")
        arr.tofile(p)
        puts.append(p)
    sftp = cli.open_sftp()
    for p in puts:
        sftp.put(p, "%s/vec/mini/in_40/%s" % (DEV, os.path.basename(p)))
    sftp.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("in_40 zero history restored")


def stage_engine(cli):
    rc, o, e = run(cli, ENGINE_SH.format(dev=DEV))
    assert "ENGINE_LAUNCHED" in o, "engine launch failed: %r %r" % (o, e)
    ok = wait_log(cli, MODS + "/build_t6m.log", "=== OK", 5400)
    rc, o, _ = run(cli, "tail -6 %s/build_t6m.log" % MODS)
    print(o)
    if not ok:
        rc, o, _ = run(cli, "grep -iE 'error|assert|nvrtc' %s/build_t6m.log "
                           "| tail -10" % MODS)
        print("BUILD PROBLEM:\n", o)
        raise SystemExit("e_T6m build did not finish; see log tail above")
    assert "=== OK" in o, "build finished but no OK marker"


def stage_chainm(cli):
    run(cli, "rm -rf %s; mkdir -p %s" % (OUT_M, OUT_M))
    cmd = ("(setsid nohup bash %s models/e_T6m.engine "
           "/usr/local/lib/libdfaplug_v8.so %s 0 80 40 "
           "> %s/chainm.log 2>&1 < /dev/null &); echo GO" % (CHAIN_B, OUT_M, MODS))
    _, out, _ = cli.exec_command(cmd, timeout=10)
    assert "GO" in out.read().decode(), "chainm launch failed"
    wait_log(cli, DEV + "/repro/mini_reset_mini_m.log", "RESET_CHAIN_DONE", 2400)
    rc, o, _ = run(cli, "tail -3 %s/repro/mini_reset_mini_m.log; "
                       "ls %s | wc -l" % (DEV, OUT_M))
    print(o)
    assert "RESET_CHAIN_DONE" in o, "chainm did not finish"


def stage_prof(cli):
    for eng in ("e_T6", "e_T6m"):
        rc, o, _ = run(cli,
                       "/usr/src/tensorrt/bin/trtexec "
                       "--loadEngine=%s/models/%s.engine "
                       "--plugins=/usr/local/lib/libdfaplug_v8.so "
                       "--iterations=100 2>&1 | grep -E 'GPU Compute Time|mean|median' "
                       "| head -6" % (DEV, eng), t=900)
        print("== %s ==\n%s" % (eng, o))


def _pull_dir(cli, remote_dir, local_dir):
    sftp = cli.open_sftp()
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for k in range(81):
        rd = "%s/outv8_%02d" % (remote_dir, k)
        ld = os.path.join(local_dir, "outv8_%02d" % k)
        os.makedirs(ld, exist_ok=True)
        try:
            names = [f for f in sftp.listdir(rd) if f.endswith(".bin")]
        except IOError:
            print("MISSING", rd)
            continue
        for f in names:
            sftp.get("%s/%s" % (rd, f), os.path.join(ld, f))
            n += 1
    sftp.close()
    return n


def stage_fetch(cli):
    n1 = _pull_dir(cli, OUT_R, PULL_R)
    n2 = _pull_dir(cli, OUT_M, PULL_M)
    for d in (PULL_R, PULL_M):
        shutil.copy(META_SRC, os.path.join(d, "mini_meta.npz"))
    print("pulled %d + %d files" % (n1, n2))
    rc, o, _ = run(cli, "md5sum %s/models/e_T6m.engine "
                       "%s/vec/mini_r/outv8_00/map_pts.bin "
                       "%s/vec/mini_m/outv8_00/map_pts.bin" % (DEV, DEV, DEV))
    print(o)
    logs = [(DEV + "/repro/mini_reset_mini_r.log", "mini_reset_mini_r.log"),
            (DEV + "/repro/mini_reset_mini_m.log", "mini_reset_mini_m.log"),
            (MODS + "/build_t6m.log", "build_t6m.log")]
    sftp = cli.open_sftp()
    for rp, ln in logs:
        try:
            sftp.get(rp, os.path.join(ROOT, "deploy", "artifacts", ln))
        except IOError:
            print("log missing:", rp)
    sftp.close()


STAGES = {
    "push": stage_push,
    "tool": stage_tool,
    "chainr": stage_chainr,
    "engine": stage_engine,
    "chainm": stage_chainm,
    "prof": stage_prof,
    "fetch": stage_fetch,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="push,tool,chainr,engine,chainm,prof,fetch")
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

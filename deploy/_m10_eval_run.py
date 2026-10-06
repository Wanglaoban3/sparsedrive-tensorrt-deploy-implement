# -*- coding: utf-8 -*-
"""单 run 全指标 eval (det/map/EPA-L2-col), mprode 等 A/B 门禁通用:
  1) _mprod_mkeval.py <tag>   -> evaldata/mini_<tag>_{map,mp}
  2) det 别名目录 outv8_XX    -> deploy env eval_t6_mini_v8.py (det mAP/NDS)
  3) 默认 python eval_t6_mini_map.py --eng mini_<tag>_map (map mAP, shapely 2.x)
  4) deploy env eval_mp_mini.py --eng mini_<tag>_mp (EPA/L2/col)
用法: python _m10_eval_run.py <tag>   (前提: work_dirs/preproc_ref/<tag> 已归档)
日志: work_dirs/preproc_ref/<tag>_eval_{det,map,mp}.log"""
import io
import os
import shutil
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TAG = sys.argv[1] if len(sys.argv) > 1 else "mprode_b"
SRC = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
ED = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
PYDEP = r"H:\miniconda3\envs\sparsedrive_deploy\python.exe"
assert os.path.isfile(os.path.join(SRC, "frame_log.tsv")), "run dir missing"


def sh(cmd, log):
    print("==", " ".join(os.path.basename(c) if i == 1 else c
                         for i, c in enumerate(cmd[:3])), "->", log)
    with open(log, "w", encoding="utf-8") as lg:
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT,
                             cwd=ROOT)
    assert rc == 0, "rc=%d: %s" % (rc, log)
    txt = open(log, encoding="utf-8").read()
    print(txt[-1500:])
    return txt


# 1) mkeval: map + mp 目录
r = subprocess.call([sys.executable, os.path.join(ROOT, "deploy",
                                                  "_mprod_mkeval.py"), TAG])
assert r == 0, "mkeval failed"

# 2) det: outv8_XX 别名目录 + deploy env 评估
d_det = os.path.join(ED, "mini_%s_det" % TAG)
ref = os.path.join(ED, "mini_eng_v8")
det_keys = ["det_cls", "det_bbox", "det_quality", "det_instance_id"]
shutil.rmtree(d_det, ignore_errors=True)
for k in range(81):
    od = os.path.join(d_det, "outv8_%02d" % k)
    os.makedirs(od, exist_ok=True)
    for nm in det_keys:
        shutil.copyfile(os.path.join(SRC, "out_%02d_%s.bin" % (k, nm)),
                        os.path.join(od, nm + ".bin"))
shutil.copyfile(os.path.join(ref, "mini_meta.npz"),
                os.path.join(d_det, "mini_meta.npz"))
env = dict(os.environ, EVAL_ENG_DIR=d_det)
lg = os.path.join(SRC, "..", "%s_eval_det.log" % TAG)
with open(lg, "w", encoding="utf-8") as f:
    rc = subprocess.call([PYDEP, os.path.join(ROOT, "deploy",
                                              "eval_t6_mini_v8.py")],
                         env=env, stdout=f, stderr=subprocess.STDOUT,
                         cwd=ROOT)
assert rc == 0, "det eval rc=%d (%s)" % (rc, lg)
print("== det (eval_t6_mini_v8) tail ==")
print(open(lg, encoding="utf-8").read()[-900:])

# 3) map: 默认 python (shapely 2.x)
lg_map = os.path.join(SRC, "..", "%s_eval_map.log" % TAG)
art_map = os.path.join(ROOT, "deploy", "artifacts",
                       "eval_m10_%s_map.json" % TAG)
sh([sys.executable, os.path.join(ROOT, "deploy", "eval_t6_mini_map.py"),
    "--eng", os.path.join(ED, "mini_%s_map" % TAG), "--art", art_map],
   lg_map)

# 4) mp: deploy env (EPA/L2/col)
lg_mp = os.path.join(SRC, "..", "%s_eval_mp.log" % TAG)
sh([PYDEP, os.path.join(ROOT, "deploy", "eval_mp_mini.py"), "--eng",
    os.path.join(ED, "mini_%s_mp" % TAG)], lg_mp)
print("EVAL_RUN_DONE", TAG)

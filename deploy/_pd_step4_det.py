# -*- coding: utf-8 -*-
"""Step4a: mkeval mprodd + det mAP 评估 (eval_v8_mini.json 备份/恢复)."""
import io
import os
import shutil
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TAG = "mprodd"
SRC = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
ED = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
DMAP = os.path.join(ED, "mini_%s_map" % TAG)
PYDEP = r"H:\miniconda3\envs\sparsedrive_deploy\python.exe"
ART = os.path.join(ROOT, "deploy", "artifacts", "eval_v8_mini.json")
BAK = ART + ".mprodd_bak"

r = subprocess.call([sys.executable, os.path.join(ROOT, "deploy",
                                                  "_mprod_mkeval.py"), TAG])
assert r == 0, "mkeval failed"
assert os.path.isdir(DMAP) and os.path.isdir(os.path.join(ED, "mini_%s_mp"
                                                          % TAG))

# eval_t6_mini_v8.py 吃 outv8_XX 布局 (mini_eng_v8 约定), mkeval 产出
# out_XX (map/mp 口径) — det 单独组装一份 outv8_XX 别名目录
DDET = os.path.join(ED, "mini_%s_det" % TAG)
REF = os.path.join(ED, "mini_eng_v8")
DET_KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id"]
shutil.rmtree(DDET, ignore_errors=True)
for k in range(81):
    od = os.path.join(DDET, "outv8_%02d" % k)
    os.makedirs(od, exist_ok=True)
    for nm in DET_KEYS:
        shutil.copyfile(os.path.join(SRC, "out_%02d_%s.bin" % (k, nm)),
                        os.path.join(od, nm + ".bin"))
shutil.copyfile(os.path.join(REF, "mini_meta.npz"),
                os.path.join(DDET, "mini_meta.npz"))
print("det dir assembled:", DDET)

shutil.copyfile(ART, BAK)
env = dict(os.environ, EVAL_ENG_DIR=DDET)
with open(os.path.join(ROOT, "work_dirs", "preproc_ref",
                       "mprodd_det_eval.log"), "w", encoding="utf-8") as lg:
    rc = subprocess.call([PYDEP, os.path.join(ROOT, "deploy",
                                              "eval_t6_mini_v8.py")],
                         env=env, stdout=lg, stderr=subprocess.STDOUT,
                         cwd=ROOT)
shutil.copyfile(BAK, ART)  # 恢复基线 json
assert rc == 0, "det eval failed rc=%d" % rc
tail = open(os.path.join(ROOT, "work_dirs", "preproc_ref",
                         "mprodd_det_eval.log"), encoding="utf-8").read()[-1200:]
print(tail)
print("STEP4A_DONE")

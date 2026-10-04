# -*- coding: utf-8 -*-
"""decode e_T6 + DFA-v8-plugin outputs on nuScenes mini -> official mAP/NDS.
Same decode as eval_t6_mini.py but reads mini_eng_v8 (outv8_XX) and compares
against both FP32 baseline and the v3-plugin engine mini result."""
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)
os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")
os.environ.setdefault("SPARSEDRIVE_DFA_PCHUNK", "32")

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)
from eval_nuscenes import DET_ONLY_EVAL_MODE  # noqa: E402

BASE = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2")
ENG = os.environ.get("EVAL_ENG_DIR", "") or os.path.join(
    BASE, "evaldata", "mini_eng_v8")
NUM_OUT = 300


def load_out(k, name, shape, dt=np.float32):
    a = np.fromfile(os.path.join(ENG, "outv8_%02d" % k, name + ".bin"), dt)
    return a.reshape(shape)


def decode_sample(k):
    cls = 1.0 / (1.0 + np.exp(-load_out(k, "det_cls", (900, 10))))
    q = load_out(k, "det_quality", (900, 2))
    box = load_out(k, "det_bbox", (900, 11))
    if not np.isfinite(cls).all() or not np.isfinite(box).all():
        print(f"WARNING: outv8_{k:02d} has non-finite outputs, using zeros")
        return {"boxes_3d": torch.zeros((0, 10)),
                "scores_3d": torch.zeros((0,)),
                "labels_3d": torch.zeros((0,), dtype=torch.int64)}
    flat = cls.reshape(-1)
    top = np.argsort(-flat, kind="stable")[:NUM_OUT]
    scores = flat[top]
    cls_ids = top % 10
    anchor_idx = top // 10
    cent = 1.0 / (1.0 + np.exp(-q[anchor_idx, 0]))
    scores = scores * cent
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    cls_ids = cls_ids[order]
    sel = box[anchor_idx][order]
    xy = sel[:, :3]
    whl = np.exp(sel[:, 3:6])
    yaw = np.arctan2(sel[:, 6], sel[:, 7])
    vel = sel[:, 8:11]
    boxes = np.concatenate([xy, whl, yaw[:, None], vel], axis=1)
    return {"boxes_3d": torch.from_numpy(boxes.astype(np.float32)),
            "scores_3d": torch.from_numpy(scores.astype(np.float32)),
            "labels_3d": torch.from_numpy(cls_ids.astype(np.int64))}


cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
if cfg.get("custom_imports", None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg["custom_imports"])
ds_cfg = copy.deepcopy(cfg.data.val)
ds_cfg["test_mode"] = True
ds_cfg.pop("type", None)
ds = NuScenes3DDataset(**ds_cfg)
meta = np.load(os.path.join(ENG, "mini_meta.npz"), allow_pickle=True)
tokens = list(meta["tokens"])
assert len(tokens) == len(ds), (len(tokens), len(ds))
for i in range(len(ds)):
    assert ds.data_infos[i]["token"] == tokens[i], i
print("token order OK,", len(ds), "samples")

outputs = [{"img_bbox": decode_sample(i)} for i in range(len(ds))]
sc = np.concatenate([o["img_bbox"]["scores_3d"] for o in outputs])
print(f"scores: max={sc.max():.4f} p99={np.percentile(sc,99):.4f} "
      f"mean={sc.mean():.4f}")

ds.work_dir = os.path.join(ROOT, "deploy", "artifacts", "eval_v8_mini")
os.makedirs(ds.work_dir, exist_ok=True)
ret = ds.evaluate(outputs, eval_mode=DET_ONLY_EVAL_MODE)
summary = {k: float(v) for k, v in ret.items()
           if isinstance(v, (int, float))}
summary = {k: v for k, v in summary.items()
           if k.endswith(("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE",
                          "mAAE"))}
out_json = os.path.join(ROOT, "deploy", "artifacts", "eval_v8_mini.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, indent=2))

t6 = json.load(open(os.path.join(ROOT, "deploy", "artifacts",
                                 "eval_t6_mini.json"), encoding="utf-8"))
t6map = t6.get("img_bbox_NuScenes/mAP")
t6nds = t6.get("img_bbox_NuScenes/NDS")
v8map = summary.get("img_bbox_NuScenes/mAP")
v8nds = summary.get("img_bbox_NuScenes/NDS")
print("=" * 60)
print("T3 engine + v3 plug:  mAP=%.4f NDS=%.4f" % (t6map, t6nds))
print("T3 engine + v8 plug:  mAP=%.4f NDS=%.4f" % (v8map, v8nds))
print("v8 vs v3 delta:       d_mAP=%+.4f d_NDS=%+.4f"
      % (v8map - t6map, v8nds - t6nds))
fp = os.path.join(ROOT, "deploy", "artifacts", "eval_fp32_val",
                  "metrics_summary.json")
ms = json.load(open(fp, encoding="utf-8"))
print("(FP32 full-val 参考:  mAP=%.4f NDS=%.4f — 非 mini 集)"
      % (ms.get("mean_ap"), ms.get("nd_score")))

# -*- coding: utf-8 -*-
"""Det mAP/NDS of the FP32 PyTorch model on mini, from the mini_sp32 dumps
(same decode as eval_t6_mini_sp.py, different dump dir)."""
import argparse
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

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)
from eval_nuscenes import DET_ONLY_EVAL_MODE  # noqa: E402

ENG = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                   "evaldata", "mini_sp32")
NUM_OUT = 300

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eng", default=ENG,
                    help="dump dir with out_XX + mini_meta.npz")
    ap.add_argument("--art", default=os.path.join(
        ROOT, "deploy", "artifacts", "eval_sp32_mini_det.json"))
    args = ap.parse_args()
    ENG = args.eng


def load_out(k, name, shape, dt=np.float32):
    return np.fromfile(os.path.join(ENG, "out_%02d" % k, name + ".bin"),
                       dt).reshape(shape)


def decode_sample(k):
    cls = 1.0 / (1.0 + np.exp(-load_out(k, "det_cls", (900, 10))))
    q = load_out(k, "det_quality", (900, 2))
    box = load_out(k, "det_bbox", (900, 11))
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
tokens = [str(t) for t in meta["tokens"]]
assert len(tokens) == len(ds)
for i in range(len(ds)):
    assert ds.data_infos[i]["token"] == tokens[i], i
print("token order OK,", len(ds), "samples")

outputs = [{"img_bbox": decode_sample(i)} for i in range(len(ds))]
ds.work_dir = os.path.join(
    os.path.splitext(args.art)[0] + "_dir")
os.makedirs(ds.work_dir, exist_ok=True)
ret = ds.evaluate(outputs, eval_mode=DET_ONLY_EVAL_MODE)
summary = {k: float(v) for k, v in ret.items()
           if isinstance(v, (int, float))}
summary = {k: v for k, v in summary.items()
           if k.endswith(("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE",
                          "mAAE"))}
out_json = args.art
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, indent=2))

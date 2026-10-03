# -*- coding: utf-8 -*-
"""decode e_T12 engine outputs on nuScenes mini -> official mAP/NDS,
compared against the FP32 torch baseline and the e_T6 gate (0.4184)."""
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
ENG = os.path.join(BASE, "evaldata", "mini_eng")
NUM_OUT = 300
T6_MAP = 0.4184  # e_T6 gate from docs/OPTIMIZATION_SUMMARY.md


def load_out(k, name, shape, dt=np.float32):
    import os as _os
    for pref in ("out7_%02d", "out6_%02d", "out_%02d"):
        d = pref % k
        if _os.path.exists(os.path.join(ENG, d, name + ".bin")):
            a = np.fromfile(os.path.join(ENG, d, name + ".bin"), dt)
            return a.reshape(shape)
    raise FileNotFoundError(f"{name}.bin not found for frame {k:02d}")


def decode_sample(k):
    cls = 1.0 / (1.0 + np.exp(-load_out(k, "det_cls", (900, 10))))
    q = load_out(k, "det_quality", (900, 2))
    box = load_out(k, "det_bbox", (900, 11))
    if not np.isfinite(cls).all() or not np.isfinite(box).all():
        print(f"WARNING: out7_{k:02d} has non-finite outputs, using zeros")
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

ds.work_dir = os.path.join(ROOT, "deploy", "artifacts", "eval_t12_mini")
os.makedirs(ds.work_dir, exist_ok=True)
ret = ds.evaluate(outputs, eval_mode=DET_ONLY_EVAL_MODE)
summary = {k: float(v) for k, v in ret.items()
           if isinstance(v, (int, float))}
summary = {k: v for k, v in summary.items()
           if k.endswith(("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE",
                          "mAAE"))}
out_json = os.path.join(ROOT, "deploy", "artifacts", "eval_t12_mini.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, indent=2))

fp = os.path.join(ROOT, "deploy", "artifacts", "eval_fp32")
ms = json.load(open(os.path.join(fp, "metrics_summary.json"),
                    encoding="utf-8"))
f32 = {"mAP": ms.get("mean_ap"), "NDS": ms.get("nd_score")}
print("=" * 60)
print("FP32 mini baseline: mAP=%.4f NDS=%.4f" % (f32["mAP"], f32["NDS"]))
t12map = summary.get("img_bbox_NuScenes/mAP")
t12nds = summary.get("img_bbox_NuScenes/NDS")
if t12map is not None:
    print("T12 engine mini:    mAP=%.4f NDS=%.4f" % (t12map, t12nds))
    print("T6 gate:            mAP=%.4f" % T6_MAP)
    print("delta vs FP32:      d_mAP=%+.4f d_NDS=%+.4f"
          % (t12map - f32["mAP"], (t12nds or 0) - f32["NDS"]))
    ok = t12map >= T6_MAP - 1e-9
    print("GATE %s (>= %.4f: got %.4f)"
          % ("PASS" if ok else "FAIL", T6_MAP, t12map))
    print("T12_MINI_%s" % ("PASS" if ok else "FAIL"))

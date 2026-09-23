"""Compare native forward (simple_test) vs the ONNX-contract path
(SparseDriveONNXWrapper.forward -> det_head.forward_onnx) on a few
mini-val samples.

Both run with first-frame semantics (native banks reset per sample;
wrapper gets zeroed history), so the only differences left are the
deploy-path logic itself (explicit temporal inputs, trt_friendly_topk,
anchor handling).  Reports raw-tensor deltas (cls/pred/quality) and a
decoded-box comparison for the first sample.

Usage (project root): python deploy/cmp_native_vs_onnx.py [--n 10]
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)
os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")

import projects.mmdet3d_plugin  # noqa: E402,F401

from eval_nuscenes import build_eval_dataset, build_fp32  # noqa: E402
from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402
from ptq_sensitivity import sample_to_inputs  # noqa: E402
from qat import build_model_det_map  # noqa: E402


def reset_banks(model):
    model.head.det_head.instance_bank.reset()
    model.head.map_head.instance_bank.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--checkpoint", default="ckpt/sparsedrive_stage2.pth")
    args = ap.parse_args()
    torch.manual_seed(0)

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    cfg.model.head.task_config = cfg.task_config
    cfg.model.train_cfg = None

    model = build_fp32(args.config, args.checkpoint)
    wrapper = SparseDriveONNXWrapper(model)

    ds = build_eval_dataset(cfg, None, None, None, 0)
    print(f"mini-val frames: {len(ds)}; comparing first {args.n}")

    det_head = model.head.det_head
    print(f"\n{'idx':>4s} {'maxd_cls':>10s} {'maxd_pred':>10s} "
          f"{'maxd_qual':>10s} {'n_native':>8s} {'n_onnx':>7s}")
    worst = {"cls": 0.0, "pred": 0.0, "qual": 0.0}
    for i in range(args.n):
        item = ds[i]
        from mmcv.parallel import DataContainer as DC
        data = {k: (v._data if isinstance(v, DC) else v)
                for k, v in item.items()}
        data = {k: v for k, v in data.items()}

        # ---- native path (exact simple_test route) ----
        reset_banks(model)
        cap = {}
        orig_fwd = det_head.forward

        def spy(fm, metas, _orig=orig_fwd, _cap=cap):
            out = _orig(fm, metas)
            _cap["det"] = out
            return out

        det_head.forward = spy
        model.eval()
        with torch.no_grad():
            native_res = model(return_loss=False, rescale=True, **data)
        det_head.forward = orig_fwd
        nat = cap["det"]
        nat_cls = nat["classification"][-1]
        nat_pred = nat["prediction"][-1]
        nat_qual = nat["quality"][-1] if nat.get("quality") is not None \
            else None

        # ---- ONNX-contract path ----
        reset_banks(model)
        inputs = sample_to_inputs(item, wrapper)
        with torch.no_grad():
            outs = wrapper(*inputs)
        # output order: det_cls(0), det_bbox(1), det_quality(2), ...
        w_cls, w_pred, w_qual = outs[0], outs[1], outs[2]

        d_cls = (nat_cls - w_cls).abs().max().item()
        d_pred = (nat_pred - w_pred).abs().max().item()
        d_qual = ((nat_qual - w_qual).abs().max().item()
                  if nat_qual is not None else float("nan"))

        # decoded box count for context
        nnat = native_res[0]["img_bbox"]["scores_3d"].shape[0]
        worst["cls"] = max(worst["cls"], d_cls)
        worst["pred"] = max(worst["pred"], d_pred)
        if d_qual == d_qual:
            worst["qual"] = max(worst["qual"], d_qual)
        print(f"{i:4d} {d_cls:10.3e} {d_pred:10.3e} {d_qual:10.3e} "
              f"{nnat:8d} {'-':>7s}")

    print(f"\nworst over {args.n} samples: cls={worst['cls']:.3e} "
          f"pred={worst['pred']:.3e} qual={worst['qual']:.3e}")
    print("CMP_DONE")


if __name__ == "__main__":
    main()

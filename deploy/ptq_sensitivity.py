"""ModelOpt PTQ sensitivity analysis for the SparseDrive det+map deploy model.

Pipeline (mirrors the export path exactly: SparseDriveONNXWrapper, first-frame
history inputs, real calibration images from nuScenes-mini):

  1. fp32 reference outputs for N eval samples
  2. MTQ INT8 (W8A8, max calibration) over the wrapper  -> full-INT8 drift
  3. leave-one-out sensitivity: for every quantized module, disable its
     quantizers (keeping all calibrated ranges), re-measure output drift;
     the drop vs full-INT8 drift = that module's sensitivity contribution
  4. progressive skip-list curve: keep the top-K sensitive modules in FP,
     quantize everything else -> drift(K)
  5. artifacts: JSON + Markdown report with recommended skip list

Usage (from project root, env with torch+mmcv+modelopt):
    python deploy/ptq_sensitivity.py \
        --config projects/configs/sparsedrive_small_stage2.py \
        --checkpoint ckpt/sparsedrive_stage2.pth \
        --calib-samples 16 --eval-samples 16
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401 (registers plugin modules)
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402

sys.path.append(os.path.join(ROOT, "tools"))
from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402

OUTPUT_GROUPS = {
    "det_cls": 0, "det_bbox": 1, "det_quality": 2,
    "det_instance_feature": 3, "det_anchor_embed": 4, "det_instance_id": 5,
    "next_det_feat": 6, "next_det_anchor": 7, "next_det_conf": 8,
    "next_det_instance_id": 9, "next_id_count": 10,
    "map_cls": 11, "map_pts": 12,
    "map_instance_feature": 13, "map_anchor_embed": 14,
    "next_map_feat": 15, "next_map_anchor": 16, "next_map_conf": 17,
    "ego_feature_map": 18,
}
# metrics on these continuous-output groups (ids are integer-coded, skip)
DRIFT_GROUPS = [
    "det_cls", "det_bbox", "det_quality", "det_instance_feature",
    "next_det_feat", "next_det_anchor", "next_det_conf",
    "map_cls", "map_pts", "map_instance_feature", "ego_feature_map",
]


def build_wrapper(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    if "head" in cfg.model:
        cfg.model.head.task_config = cfg.task_config
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.cuda().eval()
    wrapper = SparseDriveONNXWrapper(model)
    return cfg, wrapper


def build_samples(cfg, n_samples):
    """Load real samples from the mini infos pkl through the test pipeline."""
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa
    from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (
        NuScenes3DDataset,  # noqa: F401
    )

    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg.test_mode = True
    ds_cfg.pop("type", None)
    dataset = NuScenes3DDataset(**ds_cfg)
    n = min(n_samples, len(dataset))
    samples = []
    for i in range(n):
        item = dataset[i]
        samples.append(item)
    return samples


def sample_to_inputs(sample, wrapper, device="cuda"):
    from mmcv.parallel import DataContainer as DC

    def uw(x):
        return x._data if isinstance(x, DC) else x

    sample = {k: uw(v) for k, v in sample.items()}
    m = sample["metas"] if "metas" in sample else sample
    img = sample["img"]
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    if img.dim() == 4:  # [6,3,H,W] -> [1,6,3,H,W]
        img = img.unsqueeze(0)
    proj = sample["projection_mat"]
    if not torch.is_tensor(proj):
        proj = torch.from_numpy(np.array(proj, dtype=np.float32))
    proj = proj.unsqueeze(0) if proj.dim() == 3 else proj

    model = wrapper.model
    n_det = model.head.det_head.instance_bank.num_temp_instances
    n_map = model.head.map_head.instance_bank.num_temp_instances
    embed = 256
    dev = device
    inputs = (
        img.float().to(dev), proj.float().to(dev),
        torch.zeros(1, n_det, embed, device=dev),
        torch.zeros(1, n_det, 11, device=dev),
        torch.zeros(1, n_det, device=dev),
        torch.full((1, n_det), -1, dtype=torch.int32, device=dev),
        torch.zeros(1, 1, dtype=torch.int32, device=dev),
        torch.zeros(1, n_map, embed, device=dev),
        torch.zeros(1, n_map, 40, device=dev),
        torch.zeros(1, n_map, device=dev),
        torch.eye(4, device=dev).unsqueeze(0),
        torch.tensor([0.5], dtype=torch.float32, device=dev),
    )
    return inputs


def forward_outputs(wrapper, inputs):
    with torch.no_grad():
        outs = wrapper(*inputs)
    return [o.detach().float().cpu() for o in outs]


def output_drift(ref, test):
    """Weighted relative-L2 + cosine per output group, aggregated."""
    per_group = {}
    for g in DRIFT_GROUPS:
        i = OUTPUT_GROUPS[g]
        a, b = ref[i].flatten().float(), test[i].flatten().float()
        denom = a.norm().clamp_min(1e-8)
        rel_l2 = ((a - b).norm() / denom).item()
        cos = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
        per_group[g] = {"rel_l2": rel_l2, "cos": cos}
    mean_l2 = float(np.mean([v["rel_l2"] for v in per_group.values()]))
    mean_cos = float(np.mean([v["cos"] for v in per_group.values()]))
    return mean_l2, mean_cos, per_group


def quantized_modules(wrapper):
    """All Quant* leaf modules by name (QuantLinear/QuantConv2d/QuantMaxPool2d
    ...).  INT8_DEFAULT_CFG wraps ~112 modules with weight+input+output
    quantizers; QuantMaxPool2d has no weight_quantizer but still quantizes
    activations, so filter on ANY *_quantizer attr, not just weight."""
    return [
        (name, mod)
        for name, mod in wrapper.named_modules()
        if mod.__class__.__name__.startswith("Quant")
        and any(a.endswith("_quantizer") for a in dir(mod))
    ]


def set_module_quant(wrapper, names, enabled):
    """Enable/disable every quantizer (weight, input AND output) of the
    given Quant modules.  MTQ 0.11 INT8_DEFAULT_CFG also puts an
    output_quantizer on each module; leaving those on keeps the graph
    quantized even when weight/input are disabled."""
    attr = []
    for n in names:
        mod = dict(quantized_modules(wrapper)).get(n)
        suffixes = [a for a in dir(mod) if a.endswith("_quantizer")] \
            if mod is not None else []
        if not suffixes:
            suffixes = ["weight_quantizer", "input_quantizer",
                        "output_quantizer"]
        attr += [f"{n}.{s}" for s in suffixes]
    for a in attr:
        mtq.set_quantizer_attribute(wrapper, a, {"enable": enabled})


def eval_drift(wrapper, sample_inputs, ref_out):
    outs = [forward_outputs(wrapper, si) for si in sample_inputs]
    l2s, coss = [], []
    for r, t in zip(ref_out, outs):
        l2, cos, _ = output_drift(r, t)
        l2s.append(l2)
        coss.append(cos)
    return float(np.mean(l2s)), float(np.mean(coss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--checkpoint", default="ckpt/sparsedrive_stage2.pth")
    ap.add_argument("--calib-samples", type=int, default=16)
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--topk-curve", type=int, default=8)
    ap.add_argument("--out-dir", default="deploy/artifacts")
    args = ap.parse_args()
    os.chdir(ROOT)

    torch.manual_seed(0)
    np.random.seed(0)

    cfg, wrapper = build_wrapper(args.config, args.checkpoint)
    print("building samples ...")
    samples = build_samples(cfg, max(args.calib_samples, args.eval_samples))
    sample_inputs = [sample_to_inputs(s, wrapper) for s in samples]
    calib_inputs = sample_inputs[: args.calib_samples]
    eval_inputs = sample_inputs[: args.eval_samples]

    print("fp32 reference ...")
    t0 = time.time()
    ref_out = [forward_outputs(wrapper, si) for si in eval_inputs]
    print(f"  {time.time()-t0:.1f}s")

    # mtq.quantize() flips the whole model to train() mode; a calib forward
    # in train mode would let BatchNorm overwrite its running stats with
    # 16-sample batch statistics - a permanent, quantization-independent
    # drift source.  Force eval inside the loop AND restore the BN buffers
    # afterwards.
    bn_backup = {
        k: v.detach().clone()
        for k, v in wrapper.state_dict().items()
        if "running_mean" in k or "running_var" in k
        or "num_batches_tracked" in k
    }

    def calib_loop(model):
        model.eval()
        with torch.no_grad():
            for si in calib_inputs:
                model(*si)

    print("MTQ INT8 quantize+calibrate ...")
    t0 = time.time()
    mtq.quantize(wrapper, mtq.INT8_DEFAULT_CFG, calib_loop)
    if bn_backup:
        wrapper.load_state_dict(bn_backup, strict=False)
    wrapper.eval()
    print(f"  {time.time()-t0:.1f}s")
    l2_int8, cos_int8 = eval_drift(wrapper, eval_inputs, ref_out)
    print(f"full-INT8 drift: rel_l2={l2_int8:.5f} cos={cos_int8:.5f}")

    qmods = quantized_modules(wrapper)
    names = [n for n, _ in qmods]
    print(f"quantized modules: {len(names)}")

    # baseline drift with everything quantized (== l2_int8); per-module:
    # disable -> drift(module) ; sensitivity = drift(module) - l2_int8
    sens = {}
    for k, name in enumerate(names):
        set_module_quant(wrapper, [name], False)
        l2, cos = eval_drift(wrapper, eval_inputs, ref_out)
        set_module_quant(wrapper, [name], True)
        gain = l2_int8 - l2  # positive: disabling this module REDUCED error
        sens[name] = {"rel_l2_without": l2, "cos_without": cos,
                      "sensitivity": gain}
        print(f"[{k+1}/{len(names)}] {name}: l2_wo={l2:.5f} "
              f"sens={gain:+.5f}", flush=True)

    ranked = sorted(sens.items(), key=lambda kv: -kv[1]["sensitivity"])

    # progressive skip curve
    curve = []
    ks = sorted(set([0] + [max(1, (i + 1) * len(ranked) // args.topk_curve)
                           for i in range(args.topk_curve)] +
                     [min(len(ranked), 16), min(len(ranked), 32)]))
    for k in ks:
        skip = [p for p, _ in ranked[:k]]
        set_module_quant(wrapper, skip, False)
        l2, cos = eval_drift(wrapper, eval_inputs, ref_out)
        set_module_quant(wrapper, skip, True)
        curve.append({"keep_fp_topk": k, "rel_l2": l2, "cos": cos})
        print(f"skip-top{k}: rel_l2={l2:.5f} cos={cos:.5f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples,
        "full_int8": {"rel_l2": l2_int8, "cos": cos_int8},
        "n_quantized_modules": len(names),
        "module_sensitivity": sens,
        "ranked": [[p, v["sensitivity"]] for p, v in ranked],
        "skip_curve": curve,
    }
    out_json = os.path.join(args.out_dir, "sparsedrive_ptq_sensitivity.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved", out_json)


if __name__ == "__main__":
    main()

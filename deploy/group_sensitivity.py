"""Group-level task-metric sensitivity: which functional groups hurt mAP/NDS
when quantized?

Per-module leave-one-out with rel_l2 found a flat, long-tailed landscape
(no dominant layer set), and per-module mAP deltas (~0.001) sit far below
the subset-mAP noise floor.  Groups, however, move mAP by whole points, so
a fixed few-frame paired comparison resolves them.

For every group: fresh model -> load fp32 weights -> quantize ALL -> calibrate
-> keep the group in FP -> run the fixed 81-frame mini-val subset -> paired
mAP/NDS vs fp32 (devkit eval runs in a background thread while the GPU does
the next group).  Fresh calibration per variant keeps amax consistent with
the FP group being present.

Usage (from project root):
    python deploy/group_sensitivity.py
"""

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)
os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")
os.environ.setdefault("SPARSEDRIVE_DFA_PCHUNK", "32")

import projects.mmdet3d_plugin  # noqa: E402,F401
from mmcv import Config  # noqa: E402
import mmcv  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402

from eval_nuscenes import (  # noqa: E402
    build_eval_dataset, _load_weights, run_inference, DET_ONLY_EVAL_MODE,
)
from task_sentinel import results_path_for, evaluate_async  # noqa: E402
from ptq_sensitivity import set_module_quant, quantized_modules  # noqa: E402
from flashmha_qkv import pack_to_linears  # noqa: E402


def bucket_rules(name):
    """Return the group name for one quantized-module name (bare model
    naming, no wrapper prefix)."""
    if name.startswith("img_backbone."):
        if (name.startswith("img_backbone.conv1")
                or name.startswith("img_backbone.maxpool")
                or ".layer1." in name):
            return "backbone_shallow"
        if ".layer2." in name:
            return "backbone_mid"
        return "backbone_deep"
    if name.startswith("img_neck"):
        return "img_neck"
    if name.startswith("depth_branch"):
        return "depth_branch"
    for head in ("head.det_head", "head.map_head"):
        if name.startswith(head + ".layers."):
            short = head.split(".")[-1]
            if ".attn." in name:
                return f"{short}_attn"
            if ".cls_layers" in name or ".quality_layers" in name:
                return f"{short}_output"
            if (".kps_generator" in name or ".weights_fc" in name
                    or ".output_proj" in name):
                return f"{short}_dfa"
            if ".layers." in name:
                return f"{short}_ffn"
            return f"{short}_other"
        if name.startswith(head + "."):
            return f"{head.split('.')[-1]}_enc"
    return "other"


def build_variant(cfg, cfg_path, fp32_ckpt, keep_fp):
    """Fresh fp32 -> quantize-all -> calibrate with keep_fp already in FP.

    The group must be disabled INSIDE the calibration loop: amax of the
    remaining quantizers has to be measured on the actual mixed-precision
    graph, otherwise downstream quantizers keep amax calibrated on an
    all-INT8 forward and the variant looks artificially worse."""
    from mmdet.models import build_detector
    from qat import calib_inputs_from_train

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    pack_to_linears(model)
    _load_weights(model, fp32_ckpt)
    model.cuda().eval()

    cfg2 = Config.fromfile(cfg_path)
    wrapper_c, calib_inputs = calib_inputs_from_train(cfg2, model, 16)
    bn_backup = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if "running_mean" in k or "running_var" in k
        or "num_batches_tracked" in k
    }

    state = {"disabled": False}

    def _calib(m):
        # disable BEFORE any calibration forward.  The disable must target
        # the BARE model (module names without the wrapper's "model."
        # prefix; mtq.set_quantizer_attribute silently no-ops unmatched
        # names), while the forward goes through the wrapper (its input
        # contract matches calib_inputs).
        if not state["disabled"]:
            if keep_fp:
                set_module_quant(model, keep_fp, False)
            state["disabled"] = True
        m.eval()
        with torch.no_grad():
            for si in calib_inputs:
                m(*si)

    mtq.quantize(wrapper_c, mtq.INT8_DEFAULT_CFG, _calib)
    if bn_backup:
        model.load_state_dict(bn_backup, strict=False)
    if keep_fp:
        # ensure the group is off even if the calib loop saw no batches
        set_module_quant(model, keep_fp, False)
    model.cuda().eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--fp32-ckpt", default="ckpt/sparsedrive_stage2.pth")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--groups", default="",
                    help="comma list to restrict groups (default: all)")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    if "head" in cfg.model:
        cfg.model.head.task_config = cfg.task_config
    cfg.model.train_cfg = None

    version = "v1.0-mini"
    ds = build_eval_dataset(cfg, version, None, None, args.max_samples)
    print(f"subset: {len(ds)} frames")

    # enumerate quantized modules once on a throwaway build to get buckets
    probe = build_variant(cfg, args.config, args.fp32_ckpt, keep_fp=None)
    buckets = {}
    for name, _ in quantized_modules(probe):
        buckets.setdefault(bucket_rules(name), []).append(name)
    del probe
    torch.cuda.empty_cache()
    print("group sizes (quantized modules per bucket):")
    for g, names in sorted(buckets.items()):
        print(f"  {g:18s} {len(names):3d}  e.g. {names[0]}")

    wanted = [g for g in args.groups.split(",") if g] or sorted(buckets)
    variants = [dict(tag="fp32", keep=None, fp32=True),
                dict(tag="ctrl_int8", keep=[])]
    for g in wanted:
        variants.append(dict(tag=f"grp_{g}", keep=buckets[g]))
    variants.append(dict(tag="ctrl_allfp", keep=[n for names in
                                                 buckets.values()
                                                 for n in names]))

    from eval_nuscenes import build_fp32
    pool = ThreadPoolExecutor(max_workers=2)
    jobs, rows = [], {}
    for v in variants:
        tag = v["tag"]
        cached = results_path_for(tag, version)
        if os.path.exists(cached):
            outputs = mmcv.load(cached)
            print(f"[{tag}] reusing cache", flush=True)
        else:
            t0 = time.time()
            if v.get("fp32"):
                model = build_fp32(args.config, args.fp32_ckpt)
            else:
                model = build_variant(cfg, args.config, args.fp32_ckpt,
                                      keep_fp=v["keep"])
            outputs = run_inference(model, ds)
            print(f"[{tag}] build+infer {time.time()-t0:.0f}s "
                  f"({0 if v['keep'] is None else len(v['keep'])} modules "
                  f"kept FP)", flush=True)
            mmcv.dump(outputs, cached)
            del model
            torch.cuda.empty_cache()
        evaluate_async(ds, outputs, tag, version, pool, jobs)
        del outputs

    for job in jobs:
        tag, summary = job.result()
        rows[tag] = summary
        ref = rows.get("fp32")
        if ref:
            m = summary.get("img_bbox_NuScenes/mAP")
            n = summary.get("img_bbox_NuScenes/NDS")
            print(f"  {tag}: mAP={m:.4f} ({m-ref['img_bbox_NuScenes/mAP']:+.4f}) "
                  f"NDS={n:.4f} ({n-ref['img_bbox_NuScenes/NDS']:+.4f})",
                  flush=True)

    out = os.path.join(ROOT, "deploy", "artifacts", "group_sensitivity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_frames": len(ds), "version": version, "rows": rows,
                   "buckets": {g: len(v) for g, v in buckets.items()}},
                  f, ensure_ascii=False, indent=2)
    print("saved", out)
    print("GROUP_SENSITIVITY_DONE")


if __name__ == "__main__":
    main()

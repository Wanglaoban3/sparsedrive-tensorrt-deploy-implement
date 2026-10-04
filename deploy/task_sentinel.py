"""Task-metric sentinel: paired few-frame mAP/NDS check after each
quantization-config change (sensitivity analysis -> skip variants -> QAT).

Same fixed frame subset for every variant; the DELTA vs fp32 is the signal
(absolute mAP on a subset is statistically noisy, paired deltas are not).
GPU runs variant inference sequentially while the CPU devkit eval of the
previous variant runs in a background thread (the actual parallelism
available on a single-GPU box).

Usage (from project root):
    python deploy/task_sentinel.py --ks 0,14,28,56 \
        --qat-ckpt ckpt/sparsedrive_stage2_qat.pth \
        --skip-json deploy/artifacts/sparsedrive_ptq_sensitivity.json
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

from eval_nuscenes import (  # noqa: E402
    build_eval_dataset, build_fp32, build_quantized, run_inference,
    DET_ONLY_EVAL_MODE,
)


def results_path_for(tag, version):
    name = "results.pkl" if "trainval" in version else "results_mini.pkl"
    return os.path.join(ROOT, "deploy", "artifacts", f"eval_{tag}", name)


def evaluate_async(ds, outputs, tag, version, pool, jobs):
    """Run the devkit eval for one variant in a background thread, on a
    shallow dataset copy with its own work_dir (avoid json collisions)."""
    ds_copy = copy.copy(ds)
    out_dir = os.path.join(ROOT, "deploy", "artifacts", f"eval_{tag}")
    ds_copy.work_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    def _job():
        t0 = time.time()
        ret = ds_copy.evaluate(copy.deepcopy(outputs),
                               eval_mode=DET_ONLY_EVAL_MODE)
        summary = {k: float(v) for k, v in ret.items()
                   if isinstance(v, (int, float))}
        with open(os.path.join(out_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[eval done] {tag} ({time.time()-t0:.0f}s)", flush=True)
        return tag, summary

    jobs.append(pool.submit(_job))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--fp32-ckpt", default="ckpt/sparsedrive_stage2.pth")
    ap.add_argument("--qat-ckpt", default=None)
    ap.add_argument("--qat-skip-k", type=int, default=28)
    ap.add_argument("--skip-json", default=None)
    ap.add_argument("--ks", default="",
                    help="comma list of skip-K PTQ variants, e.g. 0,14,28,56")
    ap.add_argument("--version", default=None)
    ap.add_argument("--ann-file", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", default=True,
                    help="reuse finished inference results pkl if present")
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

    version = args.version or "v1.0-mini"
    ds = build_eval_dataset(cfg, args.version, args.ann_file,
                            args.data_root, args.max_samples)
    n = len(ds)
    print(f"sentinel subset: {n} frames (version={ds.version})")
    if args.max_samples and args.max_samples < n:
        idx = np.random.RandomState(0).permutation(n)[:args.max_samples]
        sub = torch.utils.data.Subset(ds, sorted(idx.tolist()))
    else:
        sub = ds

    variants = [dict(tag="fp32", mode="fp32", ckpt=args.fp32_ckpt,
                     skip_k=None)]
    for k in [int(x) for x in args.ks.split(",") if x != ""]:
        variants.append(dict(tag=f"ptq_skip{k}", mode="quant",
                             ckpt=args.fp32_ckpt, skip_k=k))
    if args.qat_ckpt:
        variants.append(dict(tag=f"qat_skip{args.qat_skip_k}", mode="quant",
                             ckpt=args.qat_ckpt, skip_k=args.qat_skip_k))

    pool = ThreadPoolExecutor(max_workers=2)
    jobs = []
    rows = {}
    for v in variants:
        tag = v["tag"]
        cached = results_path_for(tag, version) if args.reuse else None
        if cached and os.path.exists(cached):
            outputs = mmcv.load(cached)
            print(f"[{tag}] reusing {cached}", flush=True)
            evaluate_async(ds, outputs, tag, version, pool, jobs)
            del outputs
            continue

        if v["mode"] == "fp32":
            model = build_fp32(args.config, v["ckpt"])
        else:
            model = build_quantized(cfg, args.config, v["ckpt"],
                                    args.skip_json, v["skip_k"])
        t0 = time.time()
        outputs = run_inference(model, sub)
        print(f"[{tag}] inference {time.time()-t0:.0f}s", flush=True)
        mmcv.dump(outputs, results_path_for(tag, version))
        evaluate_async(ds, outputs, tag, version, pool, jobs)
        del model, outputs
        torch.cuda.empty_cache()

    for job in jobs:
        tag, summary = job.result()
        rows[tag] = summary
        ref = rows.get("fp32")
        if ref:
            print(f"  {tag}: mAP={summary.get('img_bbox_NuScenes/mAP', float('nan')):.4f} "
                  f"({summary.get('img_bbox_NuScenes/mAP', 0)-ref.get('img_bbox_NuScenes/mAP', 0):+.4f}) "
                  f"NDS={summary.get('img_bbox_NuScenes/NDS', float('nan')):.4f} "
                  f"({summary.get('img_bbox_NuScenes/NDS', 0)-ref.get('img_bbox_NuScenes/NDS', 0):+.4f})",
                  flush=True)

    out = os.path.join(ROOT, "deploy", "artifacts", "task_sentinel.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_frames": len(sub), "version": version, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print("saved", out)
    print("SENTINEL_DONE")


if __name__ == "__main__":
    main()

"""ModelOpt QAT for the SparseDrive det+map deploy model.

Flow:
  1. build det+map model (same task_config as export)
  2. MTQ INT8 quantize (+ optional skip list from ptq_sensitivity.json)
     with max calibration over mini samples
  3. short fine-tune on nuScenes-mini with the repo's real task losses
     (forward_train) - fake-quant active throughout (straight-through
     estimator; DFA runs the pure-torch port which is differentiable)
  4. save QAT checkpoint + before/after drift report

Usage (from project root):
    python deploy/qat.py --config projects/configs/sparsedrive_small_stage2.py \
        --checkpoint ckpt/sparsedrive_stage2.pth --iters 200 --lr 2e-5 \
        [--skip-json deploy/artifacts/sparsedrive_ptq_sensitivity.json]
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


def build_model_det_map(cfg_path, ckpt_path):
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
    return cfg, model


def build_train_dataset(cfg, max_samples=None):
    from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (
        NuScenes3DDataset,  # noqa: F401
    )

    ds_cfg = copy.deepcopy(cfg.data.train)
    ds_cfg.test_mode = False
    ds_cfg.pop("type", None)
    ds = NuScenes3DDataset(**ds_cfg)
    # Do NOT truncate data_infos: get_ann_info looks ahead up to ~6 future
    # frames for planning GT, so trailing indices would overflow.  Keep the
    # full list and expose only the first max_samples indices.
    if max_samples is not None and max_samples < len(ds):
        full = ds

        class _FirstN(torch.utils.data.Dataset):
            def __len__(self):
                return max_samples

            def __getitem__(self, idx):
                return full[idx]

        ds = _FirstN()
    return ds


def collate(items):
    """Batch raw dataset items into the layout the heads expect:
    - img_metas stays a list of per-sample meta dicts
      (instance_bank reads metas["img_metas"][i]["T_global"])
    - fixed-shape numpy arrays stack into one float tensor
      (projection_mat -> [B, cams, 4, 4], timestamp -> [B], ego_status ...)
    - variable-size GT (gt_labels_3d / gt_bboxes_3d / gt_map_*) becomes a
      list of per-sample tensors (the sampler pads them itself)
    - focal (per-sample per-cam arrays) is concatenated to [B*num_cams]
    """
    from mmcv.parallel import DataContainer as DC

    def dc_unwrap(v):
        return v._data if isinstance(v, DC) else v

    items = [{k: dc_unwrap(v) for k, v in it.items()} for it in items]
    out = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        if k == "focal":
            out[k] = torch.as_tensor(
                np.concatenate([np.asarray(x).reshape(-1) for x in vals]),
                dtype=torch.float32)
            continue
        if k == "gt_depth":
            # per-sample list[level][cams,h,w] -> list[level][bs,cams,h,w]
            # (DenseDepthNet.loss zips levels with the depth predictions)
            out[k] = [torch.as_tensor(np.stack(lv), dtype=torch.float32)
                      for lv in zip(*vals)]
            continue
        if isinstance(vals[0], torch.Tensor):
            try:
                out[k] = torch.stack(vals, 0)
                continue
            except RuntimeError:
                out[k] = vals  # variable-size GT tensors -> keep list
                continue
        if isinstance(vals[0], np.ndarray):
            try:
                out[k] = torch.as_tensor(np.stack(vals, 0))
                continue
            except ValueError:
                out[k] = [torch.as_tensor(np.asarray(x)) for x in vals]
                continue
        if (isinstance(vals[0], list) and vals[0]
                and isinstance(vals[0][0], torch.Tensor)):
            try:
                out[k] = [torch.stack(v, 0) for v in zip(*vals)]
                continue
            except RuntimeError:
                out[k] = vals
                continue
        if isinstance(vals[0], (int, float)):
            out[k] = torch.as_tensor(vals, dtype=torch.float32)
            continue
        out[k] = vals
    return out


def to_cuda(v):
    if isinstance(v, torch.Tensor):
        return v.cuda(non_blocking=True)
    if isinstance(v, list):
        return [to_cuda(x) for x in v]
    return v


def calib_inputs_from_train(cfg, model, n_calib):
    """Build first-frame wrapper inputs from train-pipeline samples."""
    ds = build_train_dataset(cfg, max_samples=n_calib * 2)
    from export_onnx_det_map import SparseDriveONNXWrapper  # noqa

    wrapper = SparseDriveONNXWrapper(model)
    from ptq_sensitivity import sample_to_inputs  # reuse

    outs = []
    for i in range(min(n_calib, len(ds))):
        item = ds[i]
        outs.append(sample_to_inputs(item, wrapper))
    return wrapper, outs


def quantizer_modules(model):
    mods = []
    for name, mod in model.named_modules():
        if mod.__class__.__name__.startswith("Quant"):
            mods.append((name, mod))
    return mods


def apply_skip_from_report(model, report_path):
    """Keep the top-K most sensitive modules in FP per the PTQ report.

    K = smallest point on the skip curve with rel_l2 <= 50% of full-INT8;
    if no point reaches that target (error spread over many small
    contributions), fall back to the curve's argmin so we still ship the
    best measured configuration instead of skipping everything.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    ranked = [p for p, _ in rep.get("ranked", [])]
    if not ranked:
        return []
    curve = rep.get("skip_curve", [])
    target = rep["full_int8"]["rel_l2"] * 0.5
    k = None
    for c in sorted(curve, key=lambda c: c["keep_fp_topk"]):
        if c["rel_l2"] <= target:
            k = c["keep_fp_topk"]
            break
    if k is None:
        best = min(curve, key=lambda c: c["rel_l2"])
        k = best["keep_fp_topk"]
        print(f"skip-list: no point reaches 50% target ({target:.4f}); "
              f"using curve argmin k={k} (rel_l2={best['rel_l2']:.4f})")
    keep_fp = set(ranked[:k])
    disabled = 0
    from ptq_sensitivity import set_module_quant  # 3-attr aware
    for name in ranked[:k]:
        try:
            set_module_quant(model, [name], False)
            disabled += 1
        except Exception:
            pass
    print(f"skip-list: kept {disabled}/{len(keep_fp)} sensitive modules "
          f"in FP (weight+input+output quantizers disabled)")
    return keep_fp


def main():
    # pure-torch DFA is memory-hungry with autograd; chunk its anchor axis
    # and checkpoint the op by default so QAT fits in 16 GB
    os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")
    os.environ.setdefault("SPARSEDRIVE_DFA_CKPT", "1")
    # must land before the first CUDA allocation in this process
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--checkpoint", default="ckpt/sparsedrive_stage2.pth")
    ap.add_argument("--calib-samples", type=int, default=16)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--skip-json", default=None)
    ap.add_argument("--skip-k", type=int, default=None,
                    help="override skip-list size (default: report policy)")
    ap.add_argument("--out-dir", default="deploy/artifacts")
    ap.add_argument("--save-ckpt", default="ckpt/sparsedrive_stage2_qat.pth")
    args = ap.parse_args()
    os.chdir(ROOT)
    torch.manual_seed(0)
    np.random.seed(0)

    cfg, model = build_model_det_map(args.config, args.checkpoint)

    # split FlashMHA's packed in_proj Parameter into q/k/v Linears so MTQ
    # can quantize QKV like every other Linear (+21 modules -> ~133 total)
    from flashmha_qkv import pack_to_linears
    n_attn = pack_to_linears(model)
    print(f"QKV surgery: {n_attn} FlashMHA modules -> q/k/v Linears")

    print("MTQ quantize + calibrate ...")
    sys.path.append(os.path.join(ROOT, "deploy"))
    wrapper, calib_inputs = calib_inputs_from_train(
        cfg, model, args.calib_samples)

    # drift bookkeeping: fp32 reference on eval samples (first-frame wrapper
    # inputs, same contract as deployment), compared after calib and QAT
    from ptq_sensitivity import (build_samples, sample_to_inputs,
                                 forward_outputs, output_drift)
    eval_samples = build_samples(cfg, args.eval_samples)
    eval_inputs = [sample_to_inputs(s, wrapper) for s in eval_samples]
    model.eval()
    ref_out = [forward_outputs(wrapper, si) for si in eval_inputs]

    def drift_now():
        l2s, coss = [], []
        for r_, si in zip(ref_out, eval_inputs):
            out = forward_outputs(wrapper, si)
            l2, cos, _ = output_drift(r_, out)
            l2s.append(l2)
            coss.append(cos)
        return float(np.mean(l2s)), float(np.mean(coss))

    # mtq.quantize() flips the model to train() mode; calibrating in train
    # mode lets BatchNorm overwrite its running stats with 16-sample batch
    # statistics (a permanent drift source), so force eval in the loop and
    # restore the BN buffers afterwards
    bn_backup = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if "running_mean" in k or "running_var" in k
        or "num_batches_tracked" in k
    }

    def calib_loop(m):
        m.eval()
        with torch.no_grad():
            for si in calib_inputs:
                m(*si)

    mtq.quantize(wrapper, mtq.INT8_DEFAULT_CFG, calib_loop)
    if bn_backup:
        model.load_state_dict(bn_backup, strict=False)
    model.eval()
    if args.skip_k is not None:
        # manual override: keep exactly the top-K ranked modules in FP
        with open(args.skip_json, "r", encoding="utf-8") as f:
            ranked = [p for p, _ in json.load(f).get("ranked", [])]
        from ptq_sensitivity import set_module_quant
        for name in ranked[:args.skip_k]:
            try:
                set_module_quant(wrapper, [name], False)
            except Exception:
                pass
        print(f"skip-list: manual top-{args.skip_k} kept in FP")
    elif args.skip_json and os.path.exists(args.skip_json):
        apply_skip_from_report(wrapper, args.skip_json)
    mtq.print_quant_summary(wrapper)
    l2_ptq, cos_ptq = drift_now()
    print(f"PTQ (post-calib+skip) drift: rel_l2={l2_ptq:.5f} "
          f"cos={cos_ptq:.5f}", flush=True)
    # rollback anchor: the best state the drift gate can fall back to
    ptq_state = {k: v.detach().cpu().clone()
                 for k, v in model.state_dict().items()}

    # ---------------- fine-tune with real task losses ----------------
    print("building train dataset ...")
    ds = build_train_dataset(cfg, max_samples=200)
    print(f"train samples: {len(ds)}")

    model.train()
    # keep BN frozen? SparseDrive trains BN; for short QAT keep eval-mode BN
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = None  # fp32 QAT on P100 (fp16 + fake quant is fragile)

    t0 = time.time()
    losses_log = []
    idx = 0
    order = np.random.permutation(len(ds))
    best = {"l2": l2_ptq, "state": None, "iter": -1}
    eval_every = max(1, args.iters // 10)
    for it in range(args.iters):
        if it % len(order) == 0:
            order = np.random.permutation(len(ds))
            idx = 0
        items = []
        for b in range(args.batch_size):
            items.append(ds[int(order[(idx + b) % len(order)])])
        idx += args.batch_size
        data = collate(items)
        data = {k: to_cuda(v) for k, v in data.items()}
        img = data.pop("img")
        if img.dim() == 4:
            img = img.unsqueeze(0)
        if not torch.is_tensor(data.get("focal")):
            # safety net: focal arrived unstacked -> concat to [B*num_cams]
            data["focal"] = torch.as_tensor(
                np.concatenate(
                    [np.asarray(x).reshape(-1) for x in data["focal"]]),
                dtype=torch.float32).to(img.device)
        losses = model.forward_train(img, **data)
        loss = sum(v for k, v in losses.items() if "loss" in k and v.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 25)
        opt.step()
        losses_log.append(loss.item())
        if (it + 1) % eval_every == 0 or it == args.iters - 1:
            # drift-gated QAT: small-data fine-tunes can move weights away
            # from the eval optimum, so keep the checkpoint with the LOWEST
            # eval drift (never worse than the PTQ state we started from)
            model.eval()
            l2, cos = drift_now()
            if l2 < best["l2"]:
                best = {"l2": l2,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()},
                        "iter": it + 1, "cos": cos}
            print(f"drift gate @{it+1}: rel_l2={l2:.5f} cos={cos:.5f} "
                  f"(best {best['l2']:.5f} @iter {best['iter']})", flush=True)
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        if (it + 1) % 10 == 0:
            print(f"iter {it+1}/{args.iters} loss={np.mean(losses_log[-10:]):.4f} "
                  f"({(time.time()-t0)/(it+1):.2f}s/it)", flush=True)

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        l2_qat, cos_qat = best["l2"], best.get("cos", float("nan"))
        print(f"restoring best drift state @iter {best['iter']}")
    else:
        # fine-tune never beat the PTQ state -> ship the post-calib weights
        # (drift gate guarantees no regression vs PTQ)
        model.load_state_dict(ptq_state)
        l2_qat, cos_qat = l2_ptq, cos_ptq
        print("fine-tune did not improve drift; keeping post-calib state")
    model.eval()
    torch.save(model.state_dict(), args.save_ckpt)
    print("saved", args.save_ckpt)
    # NOTE: ModelOpt 0.11 Quant* classes are dynamically generated and
    # cannot be pickled whole; export_quant_onnx.py rebuilds the quantized
    # model in-process (quantize+calib -> skip list -> load_state_dict).

    rep = {"iters": args.iters, "lr": args.lr,
           "mean_loss_last20": float(np.mean(losses_log[-20:])),
           "drift_fp32_to_ptq": {"rel_l2": l2_ptq, "cos": cos_ptq},
           "drift_fp32_to_qat": {"rel_l2": l2_qat, "cos": cos_qat},
           "skip_json": args.skip_json, "skip_k": args.skip_k,
           "qat_ckpt": args.save_ckpt}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "qat_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("QAT_DONE")


if __name__ == "__main__":
    main()

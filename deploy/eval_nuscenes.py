"""nuScenes detection evaluation (mAP / NDS) for fp32 vs quantized model.

Sentinel on mini (81 frames, catches gross quantization failures) and the
real acceptance run on trainval val (6019 frames) once the data is present.

Usage (from project root):
    python deploy/eval_nuscenes.py --mode fp32 --checkpoint ckpt/sparsedrive_stage2.pth --tag fp32
    python deploy/eval_nuscenes.py --mode quant --checkpoint ckpt/sparsedrive_stage2_qat.pth \
        --skip-json deploy/artifacts/sparsedrive_ptq_sensitivity.json --skip-k 28 --tag int8qat
    # full val:
    python deploy/eval_nuscenes.py --mode quant ... --version v1.0-trainval \
        --ann-file data/infos/nuscenes_infos_val.pkl --max-samples 0
"""

import argparse
import copy
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)
os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")
os.environ.setdefault("SPARSEDRIVE_DFA_PCHUNK", "32")

import projects.mmdet3d_plugin  # noqa: E402,F401
from mmcv import Config  # noqa: E402
from mmcv.parallel import collate, scatter  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402

DET_ONLY_EVAL_MODE = dict(
    with_det=True,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)


def build_eval_dataset(cfg, version, ann_file, data_root, max_samples):
    from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (
        NuScenes3DDataset,  # noqa: F401
    )
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg["test_mode"] = True
    if version:
        ds_cfg["version"] = version
    if data_root:
        ds_cfg["data_root"] = data_root
    if ann_file:
        ds_cfg["ann_file"] = ann_file
    ds_cfg.pop("type", None)
    ds = NuScenes3DDataset(**ds_cfg)
    if max_samples and max_samples < len(ds):
        full = ds

        class _FirstN(torch.utils.data.Dataset):
            def __len__(self):
                return max_samples

            def __getitem__(self, idx):
                return full[idx]

        ds = _FirstN()
    return ds


def build_fp32(cfg_path, ckpt):
    from qat import build_model_det_map
    return build_model_det_map(cfg_path, ckpt)[1]


def _load_weights(model, ckpt):
    """mmcv checkpoints wrap tensors as {'meta':..., 'state_dict': {...}};
    plain torch.save(state_dict) files load as-is.  Packed FlashMHA
    in_proj_* tensors are remapped onto the split q/k/v Linears (the
    surgery removed in_proj, so an unmapped load would leave QKV random).
    Extra keys (e.g. motion_plan_head in the full stage2 ckpt) are ignored."""
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    remapped = {}
    for k, v in sd.items():
        if k.endswith("in_proj_weight"):
            base = k[: -len("in_proj_weight")]
            e = v.shape[1]
            remapped[base + "q_proj.weight"] = v[:e].clone()
            remapped[base + "k_proj.weight"] = v[e:2 * e].clone()
            remapped[base + "v_proj.weight"] = v[2 * e:].clone()
        elif k.endswith("in_proj_bias"):
            base = k[: -len("in_proj_bias")]
            e = v.shape[0] // 3
            remapped[base + "q_proj.bias"] = v[:e].clone()
            remapped[base + "k_proj.bias"] = v[e:2 * e].clone()
            remapped[base + "v_proj.bias"] = v[2 * e:].clone()
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    unexpected = [k for k in unexpected if "in_proj" not in k]
    print(f"loaded weights: {len(remapped)} tensors "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")
    assert not any("proj" in k and "quantizer" not in k for k in missing), \
        f"attention proj keys missing after remap: {missing[:8]}"
    return missing, unexpected


def build_quantized(cfg, cfg_path, ckpt, skip_json, skip_k):
    """Rebuild the quantized structure in-process and restore weights.

    PTQ path (mmcv fp32 ckpt): weights MUST be loaded BEFORE calibration so
    the amax statistics see real activations.  QAT path (plain state_dict
    with quantizer keys): loaded after quantize to restore weights + amax.
    """
    from qat import calib_inputs_from_train, apply_skip_from_report
    from flashmha_qkv import pack_to_linears
    from ptq_sensitivity import set_module_quant

    cfg2 = Config.fromfile(cfg_path)
    from mmdet.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    pack_to_linears(model)
    model.cuda().eval()

    sd = torch.load(ckpt, map_location="cpu")
    is_plain = not (isinstance(sd, dict) and "state_dict" in sd)
    if not is_plain:
        _load_weights(model, ckpt)

    wrapper_c, calib_inputs = calib_inputs_from_train(cfg2, model, 16)
    bn_backup = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if "running_mean" in k or "running_var" in k
        or "num_batches_tracked" in k
    }

    def _calib(m):
        m.eval()
        with torch.no_grad():
            for si in calib_inputs:
                m(*si)

    mtq.quantize(wrapper_c, mtq.INT8_DEFAULT_CFG, _calib)
    if bn_backup:
        model.load_state_dict(bn_backup, strict=False)
    if skip_json and os.path.exists(skip_json):
        if skip_k is not None:
            with open(skip_json, "r", encoding="utf-8") as f:
                ranked = [p for p, _ in json.load(f).get("ranked", [])]
            for name in ranked[:skip_k]:
                try:
                    set_module_quant(wrapper_c, [name], False)
                except Exception:
                    pass
            print(f"skip-list: manual top-{skip_k} kept in FP")
        else:
            apply_skip_from_report(wrapper_c, skip_json)

    if is_plain:
        # QAT state_dict: weights + amax, restored onto the quant structure
        _load_weights(model, ckpt)
    model.cuda().eval()
    return model


def run_inference(model, ds, attn_fp16=False):
    if attn_fp16:
        # mimic the official flash_attn fp16 kernels: the checkpoint was
        # trained with fp16 attention, so fp32 attention is slightly
        # out-of-distribution for the downstream layers
        from projects.mmdet3d_plugin.models.attention import FlashAttention

        def _fp16_forward(self, q, k, v, causal=False, key_padding_mask=None):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.softmax_scale if self.softmax_scale is not None \
                else q.size(-1) ** -0.5
            q16, k16, v16 = q.half(), k.half(), v.half()
            w = torch.matmul(q16, k16.transpose(-2, -1)) * scale
            if key_padding_mask is not None:
                m = key_padding_mask.unsqueeze(1).unsqueeze(1).bool()
                w = w.masked_fill(m, float("-inf"))
            w = torch.softmax(w, dim=-1)
            out = torch.matmul(w, v16)
            return out.transpose(1, 2).contiguous().float(), None

        FlashAttention.forward = _fp16_forward
        print("attention compute dtype: fp16 (official-kernel mimic)")
    model.eval()
    outputs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            data = scatter(collate([item], samples_per_gpu=1), [0])[0]
            result = model(return_loss=False, rescale=True, **data)
            outputs.extend(result)
            if (i + 1) % 20 == 0 or i == len(ds) - 1:
                print(f"infer {i+1}/{len(ds)} "
                      f"({(time.time()-t0)/(i+1):.2f}s/it)", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--mode", choices=["fp32", "quant"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--skip-json", default=None)
    ap.add_argument("--skip-k", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", default=None,
                    help="e.g. v1.0-trainval (default: config's mini)")
    ap.add_argument("--ann-file", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = full set")
    ap.add_argument("--out-dir", default="deploy/artifacts")
    ap.add_argument("--attn-fp16", action="store_true",
                    help="compute attention in fp16 like the official "
                         "flash_attn kernels the checkpoint was trained "
                         "with")
    args = ap.parse_args()
    torch.manual_seed(0)

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

    if args.mode == "fp32":
        model = build_fp32(args.config, args.checkpoint)
    else:
        model = build_quantized(cfg, args.config, args.checkpoint,
                                args.skip_json, args.skip_k)

    ds = build_eval_dataset(cfg, args.version, args.ann_file,
                            args.data_root, args.max_samples)
    print(f"dataset: {len(ds)} samples, version={ds.version}")

    outputs = run_inference(model, ds, attn_fp16=args.attn_fp16)

    ds.work_dir = os.path.join(args.out_dir, f"eval_{args.tag}")
    os.makedirs(ds.work_dir, exist_ok=True)
    ret = ds.evaluate(outputs, eval_mode=DET_ONLY_EVAL_MODE)

    summary = {k: float(v) for k, v in ret.items()
               if isinstance(v, (int, float))}
    summary = {k: v for k, v in summary.items()
               if k.endswith(("mAP", "NDS", "mATE", "mASE", "mAOE",
                              "mAVE", "mAAE"))}
    out_json = os.path.join(args.out_dir, f"eval_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved", out_json)
    print(json.dumps(summary, indent=2))
    print("EVAL_DONE", args.tag)


if __name__ == "__main__":
    main()

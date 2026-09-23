"""Export the QUANTIZED (PTQ/QAT) SparseDrive det+map model to ONNX.

Reuses SparseDriveONNXWrapper so the graph matches the deployment contract
(inputs/outputs, SparseDrive::DeformableAggregation custom node). Fake-quant
modules export as standard QuantizeLinear/DequantizeLinear pairs (opset 13).

After export, run deploy/qdq_onnx_rewrite.py (copied from the SparseDriveV2
deploy toolchain) to normalize the QDQ structure: per-site weight Q folded to
int8 initializers feeding DequantizeLinear, Constant-node scale/zp moved to
shared initializers - the "standard ORT-style" QDQ layout.

Usage (from project root):
    python deploy/export_quant_onnx.py --checkpoint ckpt/sparsedrive_stage2_qat.pth \
        --out work_dirs/sparsedrive_small_stage2/sparsedrive_int8.onnx
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "tools"))
sys.path.append(os.path.join(ROOT, "deploy"))

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401 (registers plugin modules)
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402


def _enable_traceable_fake_quant():
    """ModelOpt's _tensor_quant mutates scale via boolean-mask assignment
    (scale[zero_amax_mask] = 0), which the torch ONNX tracer rejects
    ("Named tensors are not supported with the tracer").  Swap in a
    numerically identical, traceable version (torch.where)."""
    import modelopt.torch.quantization.tensor_quant as tq

    def _tensor_quant_traceable(inputs, amax, num_bits=8, unsigned=False,
                                narrow_range=True):
        input_dtype = inputs.dtype
        if input_dtype == torch.half:
            inputs = inputs.float()
        if amax.dtype == torch.half:
            amax = amax.float()
        max_bound = torch.tensor(
            (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
        if unsigned:
            min_bound = 0
        elif narrow_range:
            min_bound = -max_bound
        else:
            min_bound = -max_bound - 1
        scale = max_bound / amax
        epsilon = 1.0 / (1 << 24)
        zero_amax_mask = amax <= epsilon
        scale = torch.where(zero_amax_mask, torch.zeros_like(scale), scale)
        outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)
        scale = torch.where(zero_amax_mask, torch.ones_like(scale), scale)
        if input_dtype == torch.half:
            outputs = outputs.half()
        return outputs, scale

    tq._tensor_quant = _tensor_quant_traceable


def simplify_onnx(model_path):
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(model_path)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, model_path)
            print("onnxsim ok")
        else:
            print("onnxsim check failed; kept original")
    except ImportError:
        print("onnxsim not installed; skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--checkpoint", default="ckpt/sparsedrive_stage2_qat.pth")
    ap.add_argument("--out", default="work_dirs/sparsedrive_small_stage2/sparsedrive_int8.onnx")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-json", default=None)
    ap.add_argument("--skip-k", type=int, default=None)
    ap.add_argument("--protect-groups", default=None,
                    help="comma list of functional groups kept in FP "
                         "(e.g. det_head_output); overrides skip options")
    ap.add_argument("--attn-fp16", action="store_true",
                    help="emit attention QK^T/softmax/PV in fp16 (matches "
                         "the flash_attn kernels the checkpoint was "
                         "trained with; ~half the attention bandwidth)")
    args = ap.parse_args()
    _enable_traceable_fake_quant()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    if "head" in cfg.model:
        cfg.model.head.task_config = cfg.task_config

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # QAT checkpoints carry q/k/v Linears (not the packed in_proj), so
    # surgery must run BEFORE quantization/load for the keys to line up
    from flashmha_qkv import pack_to_linears
    n_attn = pack_to_linears(model)
    model.cuda().eval()

    # checkpoint routing: mmcv-wrapped dict = fp32/PTQ weights (load BEFORE
    # calibration); plain state_dict = QAT weights+amax (load after quantize)
    from eval_nuscenes import _load_weights, keep_modules_for_groups
    sd = torch.load(args.checkpoint, map_location="cpu")
    is_qat = not (isinstance(sd, dict) and "state_dict" in sd)

    # rebuild the quantized model in-process: ModelOpt 0.11 Quant* classes
    # are dynamically generated (unpicklable), so QAT ships a state_dict
    # that is restored onto an identically-quantized structure here
    from qat import calib_inputs_from_train, apply_skip_from_report
    import modelopt.torch.quantization as mtq
    sys.path.append(os.path.join(ROOT, "deploy"))
    if not is_qat:
        _load_weights(model, args.checkpoint)
    wrapper_c, calib_inputs = calib_inputs_from_train(cfg, model, 16)

    # keep BN in eval during calib (see qat.py: train-mode calib pollutes
    # BN running stats)
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
    if args.protect_groups:
        groups = [g.strip() for g in args.protect_groups.split(",")]
        keep_modules_for_groups(wrapper_c, groups)
    elif args.skip_json and os.path.exists(args.skip_json):
        if args.skip_k is not None:
            import json
            with open(args.skip_json, "r", encoding="utf-8") as f:
                ranked = [p for p, _ in json.load(f).get("ranked", [])]
            from ptq_sensitivity import set_module_quant
            for name in ranked[:args.skip_k]:
                try:
                    set_module_quant(wrapper_c, [name], False)
                except Exception:
                    pass
            print(f"skip-list: manual top-{args.skip_k} kept in FP")
        else:
            apply_skip_from_report(wrapper_c, args.skip_json)

    if is_qat:
        model.load_state_dict(sd, strict=False)
        print(f"loaded QAT state: {len(sd)} tensors")
    print(f"QKV surgery: {n_attn} FlashMHA modules (checkpoint: "
          f"{args.checkpoint})")

    wrapper = SparseDriveONNXWrapper(model)

    bs, nc, H, W, embed = 1, 6, 256, 704, 256
    n_det = model.head.det_head.instance_bank.num_temp_instances
    n_map = model.head.map_head.instance_bank.num_temp_instances
    dev = "cuda"

    input_names = [
        "img", "projection_mat",
        "prev_det_feat", "prev_det_anchor", "prev_det_conf", "prev_det_id", "prev_id_count",
        "prev_map_feat", "prev_map_anchor", "prev_map_conf",
        "instance_t_matrix", "time_interval",
    ]
    output_names = [
        "det_cls", "det_bbox", "det_quality",
        "det_instance_feature", "det_anchor_embed", "det_instance_id",
        "next_det_feat", "next_det_anchor", "next_det_conf", "next_det_instance_id", "next_id_count",
        "map_cls", "map_pts",
        "map_instance_feature", "map_anchor_embed",
        "next_map_feat", "next_map_anchor", "next_map_conf",
        "ego_feature_map",
    ]

    def dummy(history_random):
        if history_random:
            dfeat = torch.randn(bs, n_det, embed, device=dev)
            danc = torch.randn(bs, n_det, 11, device=dev)
            dconf = torch.rand(bs, n_det, device=dev)
            did = torch.randint(0, 100, (bs, n_det), dtype=torch.int32, device=dev)
            didc = torch.tensor([[100]], dtype=torch.int32, device=dev)
            mfeat = torch.randn(bs, n_map, embed, device=dev)
            manc = torch.randn(bs, n_map, 40, device=dev)
            mconf = torch.rand(bs, n_map, device=dev)
        else:
            dfeat = torch.zeros(bs, n_det, embed, device=dev)
            danc = torch.zeros(bs, n_det, 11, device=dev)
            dconf = torch.zeros(bs, n_det, device=dev)
            did = torch.full((bs, n_det), -1, dtype=torch.int32, device=dev)
            didc = torch.zeros((bs, 1), dtype=torch.int32, device=dev)
            mfeat = torch.zeros(bs, n_map, embed, device=dev)
            manc = torch.zeros(bs, n_map, 40, device=dev)
            mconf = torch.zeros(bs, n_map, device=dev)
        return (
            torch.randn(bs, nc, 3, H, W, device=dev),
            torch.randn(bs, nc, 4, 4, device=dev),
            dfeat, danc, dconf, did, didc,
            mfeat, manc, mconf,
            torch.eye(4, device=dev).unsqueeze(0),
            torch.tensor([0.5], dtype=torch.float32, device=dev),
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.attn_fp16:
        # during trace, run QK^T/softmax/PV in fp16: the checkpoint was
        # trained with fp16 flash kernels, and half matmuls halve the
        # attention bandwidth on TRT
        from projects.mmdet3d_plugin.models.attention import FlashAttention

        def _fp16_export_forward(self, q, k, v, causal=False,
                                 key_padding_mask=None):
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

        FlashAttention.forward = _fp16_export_forward
        print("attention export dtype: fp16")

    for first_frame, suffix in [(True, "_first"), (False, "")]:
        out_path = args.out.replace(".onnx", f"{suffix}.onnx")
        print(f"exporting {out_path} ...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy(not first_frame), out_path,
                input_names=input_names, output_names=output_names,
                opset_version=13, do_constant_folding=True,
            )
        if not args.skip_sim:
            simplify_onnx(out_path)

    print("QUANT_ONNX_EXPORT_DONE")


if __name__ == "__main__":
    main()

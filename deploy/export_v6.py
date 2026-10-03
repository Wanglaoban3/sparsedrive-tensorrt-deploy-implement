# -*- coding: utf-8 -*-
"""v6 export: same as v5 (logits-DF) PLUS float-ification patch:
- torch.where(bool, x, y) → float arithmetic (eliminates bool fusion poison)
- torch.full(int32) → float32 (eliminates int32 intermediate)
This unblocks TRT fp16 fusion for the 5 degenerate decoder regions."""
import argparse
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
sys.path.append(os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401
from mmdet.models import build_detector  # noqa: E402

import projects.mmdet3d_plugin.models.blocks as blocks_mod  # noqa: E402
from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402
import export_quant_onnx as eqo  # noqa: E402

# ============================================================
# FLOAT-IFICATION PATCH
# torch.where(bool, x, y) → x * cond_f + y * (1 - cond_f)
# torch.full(dtype=int32) → dtype=float32
# ============================================================
_orig_where = torch.where
_orig_full = torch.full


def _float_where(cond, x, y):
    if cond.dtype == torch.bool:
        cf = cond.to(x.dtype)
        return x * cf + y * (1 - cf)
    return _orig_where(cond, x, y)


def _float_full(*args, **kwargs):
    if kwargs.get('dtype') == torch.int32:
        kwargs['dtype'] = torch.float32
    if len(args) > 2 and args[2] == torch.int32:
        args = args[:2] + (torch.float32,) + args[3:]
    return _orig_full(*args, **kwargs)


def enable_float_patch():
    torch.where = _float_where
    torch.full = _float_full


def disable_float_patch():
    torch.where = _orig_where
    torch.full = _orig_full


# ============================================================
# v5 DFA logits patch (from export_v5.py)
# ============================================================
DFA_STATE = [False]

_dfa_orig = blocks_mod.DeformableFeatureAggregation.forward


def _dfa_fwd_logits(self, instance_feature, anchor, anchor_embed,
                    feature_maps, metas, **kwargs):
    if not DFA_STATE[0]:
        return _dfa_orig(self, instance_feature, anchor, anchor_embed,
                         feature_maps, metas, **kwargs)
    bs, num_anchor = instance_feature.shape[:2]
    key_points = self.kps_generator(anchor, instance_feature)
    feature = instance_feature + anchor_embed
    if self.camera_encoder is not None:
        camera_embed = self.camera_encoder(
            metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1))
        feature = feature[:, :, None] + camera_embed[:, None]
    logits = self.weights_fc(feature).reshape(
        bs, num_anchor, self.num_cams, self.num_levels, self.num_pts,
        self.num_groups)
    points_2d = (self.project_points(
        key_points, metas["projection_mat"], metas.get("image_wh"))
        .permute(0, 2, 3, 1, 4)
        .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2))
    features = blocks_mod.DAF(*feature_maps, points_2d, logits).reshape(
        bs, num_anchor, self.embed_dims)
    output = self.proj_drop(self.output_proj(features))
    if self.residual_mode == "add":
        output = output + instance_feature
    elif self.residual_mode == "cat":
        output = torch.cat([output, instance_feature], dim=-1)
    return output


blocks_mod.DeformableFeatureAggregation.forward = _dfa_fwd_logits

import projects.mmdet3d_plugin.ops.deformable_aggregation as _dfa_op  # noqa

_dfa_pure_orig = _dfa_op.deformable_aggregation_pure


def _dfa_pure_trace(mc_ms_feat, spatial_shape, scale_start_index,
                    sampling_location, weights):
    if DFA_STATE[0]:
        bs, a_cnt = weights.shape[:2]
        return mc_ms_feat.new_zeros((bs, a_cnt, mc_ms_feat.shape[-1]))
    return _dfa_pure_orig(mc_ms_feat, spatial_shape, scale_start_index,
                          sampling_location, weights)


_dfa_op.deformable_aggregation_pure = _dfa_pure_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--checkpoint", default="ckpt/sparsedrive_stage2.pth")
    ap.add_argument("--out",
                    default="work_dirs/sparsedrive_small_stage2/"
                            "sparsedrive_int8_v6.onnx")
    ap.add_argument("--with-ref",
                    default="work_dirs/sparsedrive_small_stage2/"
                            "mtq_v6_ref.npz")
    ap.add_argument("--protect-groups", default="det_head_output")
    args = ap.parse_args()
    eqo._enable_traceable_fake_quant()

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
    from flashmha_qkv import pack_to_linears
    n_attn = pack_to_linears(model)
    model.cuda().eval()

    from eval_nuscenes import _load_weights, keep_modules_for_groups
    sd = torch.load(args.checkpoint, map_location="cpu")
    is_qat = not (isinstance(sd, dict) and "state_dict" in sd)
    from qat import calib_inputs_from_train
    import modelopt.torch.quantization as mtq
    if not is_qat:
        _load_weights(model, args.checkpoint)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    wrapper_c, calib_inputs = calib_inputs_from_train(cfg, model, 16)
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

    # 校准阶段不使用 float patch（保持与 v5 量化状态一致）
    mtq.quantize(wrapper_c, mtq.INT8_DEFAULT_CFG, _calib)
    if bn_backup:
        model.load_state_dict(bn_backup, strict=False)
    model.eval()
    if args.protect_groups:
        groups = [g.strip() for g in args.protect_groups.split(",")]
        keep_modules_for_groups(wrapper_c, groups)
    if is_qat:
        model.load_state_dict(sd, strict=False)
    print(f"QKV surgery: {n_attn} (ckpt: {args.checkpoint})")

    # 参考运行也用 float patch（与导出语义一致）
    enable_float_patch()
    DFA_STATE[0] = True
    try:
        eqo._run_reference(model, args.with_ref, fp16_attn=True)
    finally:
        DFA_STATE[0] = False
        disable_float_patch()

    wrapper = SparseDriveONNXWrapper(model)
    bs, nc, H, W, embed = 1, 6, 256, 704, 256
    n_det = model.head.det_head.instance_bank.num_temp_instances
    n_map = model.head.map_head.instance_bank.num_temp_instances
    dev = "cuda"

    input_names = ["img", "projection_mat", "prev_det_feat",
                   "prev_det_anchor", "prev_det_conf", "prev_det_id",
                   "prev_id_count", "prev_map_feat", "prev_map_anchor",
                   "prev_map_conf", "instance_t_matrix", "time_interval"]
    output_names = ["det_cls", "det_bbox", "det_quality",
                    "det_instance_feature", "det_anchor_embed",
                    "det_instance_id", "next_det_feat", "next_det_anchor",
                    "next_det_conf", "next_det_instance_id", "next_id_count",
                    "map_cls", "map_pts", "map_instance_feature",
                    "map_anchor_embed", "next_map_feat", "next_map_anchor",
                    "next_map_conf", "ego_feature_map"]

    def dummy(hist_rand):
        if hist_rand:
            dfeat = torch.randn(bs, n_det, embed, device=dev)
            danc = torch.randn(bs, n_det, 11, device=dev)
            dconf = torch.rand(bs, n_det, device=dev)
            did = torch.randint(0, 100, (bs, n_det), dtype=torch.int32,
                                device=dev)
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
        return (torch.randn(bs, nc, 3, H, W, device=dev),
                torch.randn(bs, nc, 4, 4, device=dev),
                dfeat, danc, dconf, did, didc,
                mfeat, manc, mconf,
                torch.eye(4, device=dev).unsqueeze(0),
                torch.tensor([0.5], dtype=torch.float32, device=dev))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from projects.mmdet3d_plugin.models.attention import FlashAttention

    def _fp16_attn(self, q, k, v, causal=False, key_padding_mask=None):
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scale = self.softmax_scale if self.softmax_scale is not None \
            else q.size(-1) ** -0.5
        q16, k16, v16 = q.half(), k.half(), v.half()
        w = torch.matmul(q16, k16.transpose(-2, -1)) * scale
        if key_padding_mask is not None:
            m = key_padding_mask.unsqueeze(1).unsqueeze(1).bool()
            w = w.masked_fill(m, float("-inf"))
        w = torch.softmax(w, dim=-1)
        out = torch.matmul(w.half(), v16)
        return out.transpose(1, 2).contiguous().float(), None

    FlashAttention.forward = _fp16_attn
    print("attention fp16")

    model.eval()
    for first, suffix in [(True, "_first"), (False, "")]:
        out_path = args.out.replace(".onnx", f"{suffix}.onnx")
        print(f"exporting {out_path} ...")
        # float patch + DFA logits 都只在此期间生效
        enable_float_patch()
        DFA_STATE[0] = True
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper, dummy(not first), out_path,
                    input_names=input_names, output_names=output_names,
                    opset_version=13, do_constant_folding=True)
        finally:
            DFA_STATE[0] = False
            disable_float_patch()
        eqo.simplify_onnx(out_path)
        print("onnxsim ok")
    print("QUANT_V6_EXPORT_DONE")


if __name__ == "__main__":
    main()

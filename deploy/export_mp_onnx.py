# -*- coding: utf-8 -*-
"""Export the motion+planning head as a SECOND-STAGE ONNX (v5_mp.onnx) that
consumes e_T6 outputs: det/logits+features, map/logits+features and
ego_feature_map. Two graphs are produced (same trick as export_v5):
  v5_mp_first.onnx  (scene-first frame: initializes the history queue)
  v5_mp.onnx        (subsequent frames)
The board chain picks per frame. FP32 graph; board build uses --fp16 only
(no QDQ). Quantization is evaluated separately (B2-style) before any INT8.

Run with sparsedrive_deploy env python.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

import projects.mmdet3d_plugin  # noqa: E402,F401

CKPT = os.path.join(ROOT, "ckpt", "sparsedrive_stage2.pth")
N_DET, N_MAP, Q, DIM = 900, 100, 4, 256

IN_NAMES = [
    "det_cls", "det_bbox", "det_feat", "det_anchor_embed", "det_instance_id",
    "map_cls", "map_feat", "map_anchor_embed", "ego_feature_map",
    "t_matrix",
    "history_instance_feature", "history_anchor", "history_period",
    "prev_instance_id", "prev_confidence",
    "history_ego_feature", "history_ego_anchor", "history_ego_period",
    "prev_ego_status",
]
OUT_NAMES = [
    "motion_cls", "motion_reg", "plan_cls", "plan_reg", "plan_status",
    "next_history_instance_feature", "next_history_anchor",
    "next_history_period", "next_prev_instance_id", "next_prev_confidence",
    "next_history_ego_feature", "next_history_ego_anchor",
    "next_history_ego_period", "next_prev_ego_status",
]


class MPWrapper(nn.Module):
    """Tensor-only forward; is_first_frame is a python attr (two exports)."""

    def __init__(self, model):
        super().__init__()
        self.head = model.head.motion_plan_head
        self.anchor_encoder = model.head.det_head.anchor_encoder
        self.anchor_handler = model.head.det_head.instance_bank.anchor_handler
        self.is_first_frame = False

    def forward(self, det_cls, det_bbox, det_feat, det_anchor_embed,
                det_instance_id, map_cls, map_feat, map_anchor_embed,
                ego_feature_map, t_matrix,
                history_instance_feature, history_anchor, history_period,
                prev_instance_id, prev_confidence,
                history_ego_feature, history_ego_anchor, history_ego_period,
                prev_ego_status):
        m_cls, m_reg, p_cls, p_reg, p_status, ns, _tf, _ta = \
            self.head.forward_onnx(
                det_feat, det_anchor_embed, det_cls.sigmoid(), det_bbox,
                det_instance_id, map_feat, map_anchor_embed, map_cls.sigmoid(),
                ego_feature_map, self.anchor_encoder, self.anchor_handler,
                torch.ones(det_cls.shape[0], dtype=torch.bool,
                           device=det_cls.device),
                self.is_first_frame,
                T_temp2cur=t_matrix,
                history_instance_feature=history_instance_feature,
                history_anchor=history_anchor,
                history_period=history_period,
                prev_instance_id=prev_instance_id,
                prev_confidence=prev_confidence,
                history_ego_feature=history_ego_feature,
                history_ego_anchor=history_ego_anchor,
                history_ego_period=history_ego_period,
                prev_ego_status=prev_ego_status)
        return (
            m_cls[-1], m_reg[-1], p_cls[-1], p_reg[-1], p_status[-1],
            ns["history_instance_feature"], ns["history_anchor"],
            ns["history_period"], ns["prev_instance_id"],
            ns["prev_confidence"], ns["history_ego_feature"],
            ns["history_ego_anchor"], ns["history_ego_period"],
            ns["prev_ego_status"],
        )


def build_model():
    cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = True
    cfg.model.head.task_config = cfg.task_config
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    from flashmha_qkv import pack_to_linears
    n_attn = pack_to_linears(model)
    model.cuda().eval()
    print("packed FlashMHA:", n_attn)
    return model


def dummy(first):
    dev = "cuda"
    hist_f = torch.randn(1, N_DET, Q, DIM, device=dev) if not first else \
        torch.zeros(1, N_DET, Q, DIM, device=dev)
    hist_a = torch.randn(1, N_DET, Q, 11, device=dev) if not first else \
        torch.zeros(1, N_DET, Q, 11, device=dev)
    return (
        torch.randn(1, N_DET, 10, device=dev),
        torch.randn(1, N_DET, 11, device=dev),
        torch.randn(1, N_DET, DIM, device=dev),
        torch.randn(1, N_DET, DIM, device=dev),
        torch.randint(0, 100, (1, N_DET), dtype=torch.int32, device=dev),
        torch.randn(1, N_MAP, 3, device=dev),
        torch.randn(1, N_MAP, DIM, device=dev),
        torch.randn(1, N_MAP, DIM, device=dev),
        torch.randn(1, 256, 8, 22, device=dev),
        torch.eye(4, device=dev).unsqueeze(0),
        hist_f, hist_a,
        torch.randint(1, 5, (1, N_DET), dtype=torch.int32, device=dev),
        torch.randint(0, 100, (1, N_DET), dtype=torch.int32, device=dev),
        torch.rand(1, N_DET, device=dev),
        torch.randn(1, 1, Q, DIM, device=dev),
        torch.randn(1, 1, Q, 11, device=dev),
        torch.randint(1, 5, (1, 1), dtype=torch.int32, device=dev),
        torch.randn(1, 1, 10, device=dev),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        ROOT, "work_dirs", "sparsedrive_small_stage2", "v5_mp.onnx"))
    args = ap.parse_args()

    model = build_model()
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
        w = torch.softmax(w.float(), dim=-1)
        out = torch.matmul(w.half(), v16)
        return out.transpose(1, 2).contiguous().float(), None

    FlashAttention.forward = _fp16_export_forward

    wrapper = MPWrapper(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for first, suffix in [(True, "_first"), (False, "")]:
        wrapper.is_first_frame = first
        out_path = args.out.replace(".onnx", "%s.onnx" % suffix)
        print("exporting", out_path, flush=True)
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy(first), out_path,
                input_names=IN_NAMES, output_names=OUT_NAMES,
                opset_version=13, do_constant_folding=False)
        from export_quant_onnx import simplify_onnx
        simplify_onnx(out_path)
    print("MP_EXPORT_DONE")


if __name__ == "__main__":
    main()

"""导出无量化 FP32 ONNX (first + temporal 两张), 供板上 --fp16 编译
复用项目: eval_nuscenes.build_fp32 + export_quant_onnx.SparseDriveONNXWrapper
"""
import os
import sys

ROOT = r"<REPO>"
os.chdir(ROOT)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
sys.path.append(os.path.join(ROOT, "tools"))
os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")

import torch

from eval_nuscenes import build_fp32
from export_onnx_det_map import SparseDriveONNXWrapper
from export_quant_onnx import simplify_onnx

model = build_fp32("projects/configs/sparsedrive_small_stage2.py",
                   "ckpt/sparsedrive_stage2.pth")
wrapper = SparseDriveONNXWrapper(model)
n_det = model.head.det_head.instance_bank.num_temp_instances
n_map = model.head.map_head.instance_bank.num_temp_instances
print("n_det =", n_det, "n_map =", n_map)

IN_NAMES = ["img", "projection_mat", "prev_det_feat", "prev_det_anchor",
            "prev_det_conf", "prev_det_id", "prev_id_count",
            "prev_map_feat", "prev_map_anchor", "prev_map_conf",
            "instance_t_matrix", "time_interval"]
OUT_NAMES = ["det_cls", "det_bbox", "det_quality",
             "det_instance_feature", "det_anchor_embed", "det_instance_id",
             "next_det_feat", "next_det_anchor", "next_det_conf",
             "next_det_instance_id", "next_id_count",
             "map_cls", "map_pts", "map_instance_feature", "map_anchor_embed",
             "next_map_feat", "next_map_anchor", "next_map_conf",
             "ego_feature_map"]


def dummy(history_random):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    if history_random:
        gg = torch.Generator(device=dev).manual_seed(11)
        dfeat = torch.randn(1, n_det, 256, device=dev, generator=gg)
        danc = torch.randn(1, n_det, 11, device=dev, generator=gg)
        dconf = torch.rand(1, n_det, device=dev, generator=gg)
        did = torch.randint(0, 100, (1, n_det), dtype=torch.int32, device=dev, generator=gg)
        didc = torch.tensor([[100]], dtype=torch.int32, device=dev)
        mfeat = torch.randn(1, n_map, 256, device=dev, generator=gg)
        manc = torch.randn(1, n_map, 40, device=dev, generator=gg)
        mconf = torch.rand(1, n_map, device=dev, generator=gg)
    else:
        dfeat = torch.zeros(1, n_det, 256, device=dev)
        danc = torch.zeros(1, n_det, 11, device=dev)
        dconf = torch.zeros(1, n_det, device=dev)
        did = torch.full((1, n_det), -1, dtype=torch.int32, device=dev)
        didc = torch.zeros((1, 1), dtype=torch.int32, device=dev)
        mfeat = torch.zeros(1, n_map, 256, device=dev)
        manc = torch.zeros(1, n_map, 40, device=dev)
        mconf = torch.zeros(1, n_map, device=dev)
    return (
        torch.randn(1, 6, 3, 256, 704, device=dev, generator=g),
        torch.randn(1, 6, 4, 4, device=dev, generator=g),
        dfeat, danc, dconf, did, didc, mfeat, manc, mconf,
        torch.eye(4, device=dev).unsqueeze(0),
        torch.tensor([0.5], dtype=torch.float32, device=dev),
    )


for first_frame, suffix in [(True, "_first"), (False, "")]:
    out = f"work_dirs/sparsedrive_small_stage2/sparsedrive_fp32{suffix}.onnx"
    print("exporting", out, "...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy(not first_frame), out,
            input_names=IN_NAMES, output_names=OUT_NAMES,
            opset_version=13, do_constant_folding=True)
    print("saved", out)

print("FP32_EXPORT_DONE")

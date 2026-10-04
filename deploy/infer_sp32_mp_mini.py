# -*- coding: utf-8 -*-
"""FP32 PyTorch motion+planning baseline over the 81 mini frames, deployment-
shaped: det+map via SparseDriveONNXWrapper (same as e_T6 outputs), then
MotionPlanningHead.forward_onnx with an EXTERNALLY maintained history queue
(exactly what the future e_mp engine + board chain will do).

Dumps (engine-format flat bins, per frame out_XX/):
  det_cls [900,10] det_bbox [900,11] det_quality [900,3] det_id [900]
  det_feat/det_anchor_embed [900,256] map_cls [100,3]
  map_feat/map_anchor_embed [100,256] ego_feature_map [256,8,22]
  t_matrix [4,4]
  motion_cls   [900,6]        motion_reg [900,6,12,2]
  plan_cls     [1,18]         plan_reg   [1,18,6,2]
  plan_status  [1,1,10]
  history_* / prev_* next-state tensors (for engine parity replay)

Feeding contract: scene reset (dt>2.0 or dt<0) -> is_first_frame=True +
zeroed queue; t_matrix = cur_global_inv @ prev_global (same value feeds
det/map chain and T_temp2cur).

Run with sparsedrive_deploy env python.
"""
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
sys.path.append(os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.parallel.scatter_gather import scatter  # noqa: E402
from mmdet.models import build_detector  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402

import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)
from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402

OUT = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                   "evaldata", "mini_sp32_mp")
CKPT = os.path.join(ROOT, "ckpt", "sparsedrive_stage2.pth")

N_DET, N_MAP, Q, DIM = 900, 100, 4, 256


def zero_state():
    return {
        "history_instance_feature": torch.zeros(1, N_DET, Q, DIM,
                                                device="cuda"),
        "history_anchor": torch.zeros(1, N_DET, Q, 11, device="cuda"),
        "history_period": torch.zeros(1, N_DET, dtype=torch.int32,
                                      device="cuda"),
        "prev_instance_id": torch.full((1, N_DET), -1, dtype=torch.int32,
                                       device="cuda"),
        "prev_confidence": torch.zeros(1, N_DET, device="cuda"),
        "history_ego_feature": torch.zeros(1, 1, Q, DIM, device="cuda"),
        "history_ego_anchor": torch.zeros(1, 1, Q, 11, device="cuda"),
        "history_ego_period": torch.zeros(1, 1, dtype=torch.int32,
                                          device="cuda"),
        "prev_ego_status": torch.zeros(1, 1, 10, device="cuda"),
    }


def main():
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
    model.cuda().eval()
    wrapper = SparseDriveONNXWrapper(model)
    mp_head = model.head.motion_plan_head
    det_head = model.head.det_head
    anchor_encoder = det_head.anchor_encoder
    anchor_handler = det_head.instance_bank.anchor_handler

    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg["test_mode"] = True
    ds_cfg.pop("type", None)
    ds = NuScenes3DDataset(**ds_cfg)
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    n = len(ds)
    print("dataset:", n, "samples", flush=True)

    os.makedirs(OUT, exist_ok=True)
    state = zero_state()
    hist_det = {
        "prev_det_feat": torch.zeros(1, 600, DIM, device="cuda"),
        "prev_det_anchor": torch.zeros(1, 600, 11, device="cuda"),
        "prev_det_conf": torch.zeros(1, 600, device="cuda"),
        "prev_det_id": torch.full((1, 600), -1, dtype=torch.int32,
                                  device="cuda"),
        "prev_id_count": torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
    }
    hist_map = {
        "prev_map_feat": torch.zeros(1, 33, DIM, device="cuda"),
        "prev_map_anchor": torch.zeros(1, 33, 40, device="cuda"),
        "prev_map_conf": torch.zeros(1, 33, device="cuda"),
    }
    prev_global, prev_time = None, None
    flags = []

    def dump(k, name, t):
        arr = t.detach().cpu().numpy()
        if arr.dtype == np.int32:
            arr.tofile(os.path.join(OUT, "out_%02d" % k, name + ".bin"))
        else:
            arr.astype(np.float32).tofile(
                os.path.join(OUT, "out_%02d" % k, name + ".bin"))

    for k, data in enumerate(loader):
        os.makedirs(os.path.join(OUT, "out_%02d" % k), exist_ok=True)
        with torch.no_grad():
            d = scatter(data, [torch.cuda.current_device()])[0]
            metas = d["img_metas"][0]
            curr_time = metas["timestamp"]
            if prev_time is None:
                dt, scene_start = 0.5, True
            else:
                dt = curr_time - prev_time
                scene_start = (dt > 2.0 or dt < 0)
            if scene_start:
                dt = 0.5
                state = zero_state()
                hist_det = {
                    "prev_det_feat": torch.zeros(1, 600, DIM, device="cuda"),
                    "prev_det_anchor": torch.zeros(1, 600, 11, device="cuda"),
                    "prev_det_conf": torch.zeros(1, 600, device="cuda"),
                    "prev_det_id": torch.full((1, 600), -1,
                                              dtype=torch.int32,
                                              device="cuda"),
                    "prev_id_count": torch.zeros(1, 1,
                                                 dtype=torch.int32,
                                                 device="cuda"),
                }
                hist_map = {
                    "prev_map_feat": torch.zeros(1, 33, DIM, device="cuda"),
                    "prev_map_anchor": torch.zeros(1, 33, 40, device="cuda"),
                    "prev_map_conf": torch.zeros(1, 33, device="cuda"),
                }
                prev_global = None
            dt_t = torch.tensor([dt], device="cuda", dtype=torch.float32)
            g = metas["T_global"]
            g_inv = metas["T_global_inv"]
            if prev_global is None:
                t_mat = torch.eye(4, device="cuda").unsqueeze(0)
            else:
                t_mat = torch.from_numpy(
                    np.asarray(g_inv) @ np.asarray(prev_global)
                ).float().cuda().unsqueeze(0)
            prev_global = g
            prev_time = curr_time

            outs = wrapper(d["img"], d["projection_mat"],
                           hist_det["prev_det_feat"], hist_det["prev_det_anchor"],
                           hist_det["prev_det_conf"], hist_det["prev_det_id"],
                           hist_det["prev_id_count"],
                           hist_map["prev_map_feat"], hist_map["prev_map_anchor"],
                           hist_map["prev_map_conf"], t_mat, dt_t)
            hist_det = {
                "prev_det_feat": outs[6], "prev_det_anchor": outs[7],
                "prev_det_conf": outs[8], "prev_det_id": outs[9],
                "prev_id_count": outs[10],
            }
            hist_map = {
                "prev_map_feat": outs[15], "prev_map_anchor": outs[16],
                "prev_map_conf": outs[17],
            }
            # NOTE: wrapper handles det/map recurrent via the args above;
            # maintain them too (same contract as the engine chain)
            det_cls, det_bbox = outs[0], outs[1]
            det_feat, det_anchor_embed = outs[3], outs[4]
            det_id = outs[5]
            map_feat, map_anchor_embed = outs[13], outs[14]
            map_cls = outs[11]
            ego_map = outs[18]
            for t in (det_cls, det_bbox, det_feat, ego_map, map_feat):
                assert torch.isfinite(t).all(), "frame %d non-finite" % k

            m_cls, m_reg, p_cls, p_reg, p_status, ns, temp_f, temp_a = \
                mp_head.forward_onnx(
                    det_feat, det_anchor_embed, det_cls.sigmoid(), det_bbox,
                    det_id, map_feat, map_anchor_embed, map_cls.sigmoid(),
                    ego_map, anchor_encoder, anchor_handler,
                    torch.ones(1, dtype=torch.bool, device="cuda"),
                    scene_start or k == 0,
                    T_temp2cur=t_mat,
                    history_instance_feature=state["history_instance_feature"],
                    history_anchor=state["history_anchor"],
                    history_period=state["history_period"],
                    prev_instance_id=state["prev_instance_id"],
                    prev_confidence=state["prev_confidence"],
                    history_ego_feature=state["history_ego_feature"],
                    history_ego_anchor=state["history_ego_anchor"],
                    history_ego_period=state["history_ego_period"],
                    prev_ego_status=state["prev_ego_status"])
            for nm, t in ns.items():
                assert torch.isfinite(t.float()).all(), \
                    "frame %d non-finite %s" % (k, nm)

            dump(k, "motion_cls", m_cls[-1])
            dump(k, "motion_reg", m_reg[-1])
            dump(k, "plan_cls", p_cls[-1])
            dump(k, "plan_reg", p_reg[-1])
            dump(k, "plan_status", p_status[-1])
            # det side, needed by eval_mp_mini decode (topk + rescore)
            dump(k, "det_cls", det_cls)
            dump(k, "det_bbox", det_bbox)
            dump(k, "det_quality", outs[2])
            dump(k, "det_id", det_id)
            # e_mp engine inputs (parity replay + board chain contract)
            dump(k, "det_feat", det_feat)
            dump(k, "det_anchor_embed", det_anchor_embed)
            dump(k, "map_cls", map_cls)
            dump(k, "map_feat", map_feat)
            dump(k, "map_anchor_embed", map_anchor_embed)
            dump(k, "ego_feature_map", ego_map)
            dump(k, "t_matrix", t_mat)
            for nm, t in ns.items():
                dump(k, nm, t)
            flags.append(1 if (scene_start or k == 0) else 0)

            state = dict(ns)

            if (k + 1) % 10 == 0 or k == n - 1:
                print("frame %2d/%d scene_start=%d plan_reg_max=%.2f"
                      % (k + 1, n, scene_start,
                         p_reg[-1].abs().max().item()), flush=True)

    np.savez(os.path.join(OUT, "mini_meta.npz"),
             tokens=np.array([ds.data_infos[k]["token"] for k in range(n)]),
             first=np.array(flags))
    print("dumped", n, "frames ->", OUT, flush=True)
    print("MP32_DONE")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Motion + planning evaluation from dumped deployment-format tensors.

Inputs : evaldata/<ENG>/out_XX/ flat bins written by infer_sp32_mp_mini.py
         (or the board e_T6 + e_mp chain):
           det_cls det_bbox det_quality det_id          (engine-1 outputs)
           motion_cls motion_reg plan_cls plan_reg plan_status
           history_anchor history_period
           history_ego_anchor history_ego_period
Protocol:
  motion   UniAD NuScenesEval: EPA / minADE / minFDE / miss-rate (car+ped)
           (needs every frame of the val pkl: all 81 belong to mini_val)
  planning UniAD L2(0.5..3.0s) + collision rate via PlanningMetric
           (planning_eval rebuilds its own dataloader from eval_config)
  det side is decoded only to build boxes/trajs for the motion metric.

Run with the sparsedrive_deploy env python.
"""
import argparse
import copy
import io
import json
import os
import sys

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
sys.path.append(os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
from mmcv.utils import build_from_cfg, print_log  # noqa: E402
from mmdet.core.bbox.builder import BBOX_CODERS  # noqa: E402

import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)

SHAPES = {
    "det_cls": ((900, 10), np.float32),
    "det_bbox": ((900, 11), np.float32),
    "det_quality": ((900, 2), np.float32),
    "det_id": ((900,), np.int32),
    "motion_cls": ((900, 6), np.float32),
    "motion_reg": ((900, 6, 12, 2), np.float32),
    "plan_cls": ((1, 18), np.float32),
    "plan_reg": ((1, 18, 6, 2), np.float32),
    "plan_status": ((1, 1, 10), np.float32),
    "history_anchor": ((900, 4, 11), np.float32),
    "history_period": ((900,), np.int32),
    "history_ego_anchor": ((1, 4, 11), np.float32),
    "history_ego_period": ((1,), np.int32),
}


def load_bin(d, k, name):
    shape, dt = SHAPES[name]
    p = os.path.join(d, "out_%02d" % k, name + ".bin")
    a = np.fromfile(p, dtype=dt)
    assert a.size == int(np.prod(shape)), \
        "%s: size %d != %s" % (p, a.size, shape)
    # dumps carry the batch dim; SHAPES lists per-tensor shapes -> add it back
    return torch.from_numpy(a.reshape(shape))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eng", default=os.path.join(
        ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata",
        "mini_sp32_mp"))
    ap.add_argument("--config", default="projects/configs/"
                                        "sparsedrive_small_stage2.py")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    mp_cfg = cfg.model.head.motion_plan_head
    motion_decoder = build_from_cfg(mp_cfg.motion_decoder, BBOX_CODERS)
    planning_decoder = build_from_cfg(mp_cfg.planning_decoder, BBOX_CODERS)
    det_decoder = build_from_cfg(cfg.model.head.det_head.decoder,
                                 BBOX_CODERS)

    # version=v1.0-mini so _evaluate_single_motion picks eval_set mini_val
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg.pop("type", None)
    ds_cfg["test_mode"] = True
    ds_cfg["version"] = "v1.0-mini"
    ds = NuScenes3DDataset(**ds_cfg)
    n = len(ds.data_infos)
    print("dataset:", n, "samples; eng:", args.eng, flush=True)

    meta_p = os.path.join(args.eng, "mini_meta.npz")
    if os.path.exists(meta_p):
        meta = np.load(meta_p, allow_pickle=True)
        got = list(meta["tokens"])
        want = [ds.data_infos[k]["token"] for k in range(n)]
        assert got[:n] == want, "token order mismatch vs pkl"

    results = []
    for k in range(n):
        t = {nm: load_bin(args.eng, k, nm) for nm in SHAPES}
        det_output = {
            "classification": [t["det_cls"]],
            "prediction": [t["det_bbox"]],
            "instance_id": t["det_id"],
            "quality": [t["det_quality"]],
        }
        motion_output = {
            "classification": [t["motion_cls"]],
            "prediction": [t["motion_reg"]],
            "anchor_queue": [x for x in t["history_anchor"].unbind(2)],
            "period": t["history_period"],
        }
        planning_output = {
            "classification": [t["plan_cls"]],
            "prediction": [t["plan_reg"]],
            "status": [t["plan_status"]],
            "anchor_queue": [x for x in t["history_ego_anchor"].unbind(2)],
            "period": t["history_ego_period"],
        }
        data = {
            "gt_ego_fut_cmd": torch.from_numpy(np.asarray(
                ds.data_infos[k]["gt_ego_fut_cmd"],
                dtype=np.float32)).unsqueeze(0),
        }
        det_res = det_decoder.decode(
            det_output["classification"], det_output["prediction"],
            det_output["instance_id"], det_output["quality"])
        motion_res = motion_decoder.decode(
            det_output["classification"], det_output["prediction"],
            det_output["instance_id"], det_output["quality"],
            motion_output)
        plan_res = planning_decoder.decode(
            det_output, motion_output, planning_output, data)
        results.append({"img_bbox": {**det_res[0], **motion_res[0],
                                     **plan_res[0]}})
        if (k + 1) % 20 == 0 or k == n - 1:
            fp = plan_res[0]["final_planning"]
            print("decode %2d/%d final_planning[:, -1] = %s"
                  % (k + 1, n, np.round(fp[-1].numpy(), 2).tolist()),
                  flush=True)

    out_dir = os.path.join(args.eng, "eval_mp")
    os.makedirs(out_dir, exist_ok=True)

    # ---- planning: L2 + collision (UniAD protocol) ----
    from projects.mmdet3d_plugin.datasets.evaluation.planning.\
        planning_eval import planning_eval
    plan_dict = planning_eval(results, ds.eval_config, None)

    # ---- motion: EPA / minADE / minFDE / miss-rate (UniAD protocol) ----
    thresh = cfg.evaluation.get("motion_threshhold", 0.2)
    submissions = ds.format_motion_results(results, thresh=thresh)
    motion_dict = ds._evaluate_single_motion(submissions, out_dir)

    summary = {
        "planning": {k: np.asarray(v).tolist()
                     for k, v in plan_dict.items()},
        "motion": {k: float(v) for k, v in motion_dict.items()
                   if isinstance(v, (int, float, np.floating))},
    }
    with open(os.path.join(out_dir, "mp_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_log("\n===== motion+planning summary (%s) =====" % args.eng,
              logger=None)
    print_log("Car / Ped", logger=None)
    print_log("epa = %.4f / %.4f" % (
        motion_dict["car_EPA"], motion_dict["pedestrian_EPA"]), logger=None)
    print_log("ade = %.4f / %.4f" % (
        motion_dict["car_min_ade_err"],
        motion_dict["pedestrian_min_ade_err"]), logger=None)
    print_log("fde = %.4f / %.4f" % (
        motion_dict["car_min_fde_err"],
        motion_dict["pedestrian_min_fde_err"]), logger=None)
    print_log("mr  = %.4f / %.4f" % (
        motion_dict["car_miss_rate_err"],
        motion_dict["pedestrian_miss_rate_err"]), logger=None)
    print_log("obj_col %.3f%%  obj_box_col %.3f%%  L2 %.4f" % (
        plan_dict["obj_col"] * 100, plan_dict["obj_box_col"] * 100,
        plan_dict["L2"]), logger=None)
    print("MP_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

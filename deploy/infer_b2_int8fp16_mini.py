# -*- coding: utf-8 -*-
"""B2 probe: PyTorch replica of the e_T6 engine numerics over the 81 mini
frames, dumped in the ENGINE dump format so the standalone map/det evals
consume it exactly like a board dump.

Quantization provenance mirrors deploy/export_v5.py line by line (same
checkpoint, seeds, 16-sample train-pipeline calibration, det_head_output
protection).  On top of that it adds the two engine-side graph surgeries:

  * T3 mimic  - quantizers on model.head.{det_head,map_head} disabled
                (the board graph had head QDQ stripped + weights refloated)
  * P1h mimic - det/map heads run under torch.autocast(fp16); FlashAttention
                runs the fp16 QK^T/softmax/PV protocol (checkpoint was
                trained with fp16 flash attention; strategy doc: fp32
                attention is slightly OOD)

BEFORE the long inference a QDQ-scale sentinel compares every calibrated
amax/127 of the rebuilt model against the Quantize/DequantizeLinear scale
initializers inside v5_P1h.onnx (the engine's source graph).  If the
backbone/neck scales don't match, provenance is broken and the run aborts.

Output: work_dirs/sparsedrive_small_stage2/evaldata/mini_b2/
        out_XX/{map_cls,map_pts,det_cls,det_bbox,det_quality}.bin + tokens

Run (from repo root, sparsedrive_deploy env):
    H:/miniconda3/envs/sparsedrive_deploy/python.exe deploy/infer_b2_int8fp16_mini.py
"""
import functools
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "tools"))
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)
os.environ.setdefault("SPARSEDRIVE_DFA_CHUNK", "256")
os.environ.setdefault("SPARSEDRIVE_DFA_PCHUNK", "32")

OUT = os.environ.get("B2_OUT", "") or os.path.join(
    ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata", "mini_b2")
ONNX_REF = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                        "v5_P1h.onnx")
CKPT = os.path.join(ROOT, "ckpt", "sparsedrive_stage2.pth")


def build_quantized_model():
    """export_v5.py quantization path, verbatim."""
    from mmcv import Config
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet.models import build_detector
    from flashmha_qkv import pack_to_linears
    from eval_nuscenes import _load_weights, keep_modules_for_groups
    from qat import calib_inputs_from_train
    from ptq_sensitivity import quantized_modules, set_module_quant
    import modelopt.torch.quantization as mtq

    cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    cfg.model.head.task_config = cfg.task_config

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    n_attn = pack_to_linears(model)
    model.cuda().eval()
    _load_weights(model, CKPT)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    wrapper, calib_inputs = calib_inputs_from_train(cfg, model, 16)
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

    mtq.quantize(wrapper, mtq.INT8_DEFAULT_CFG, _calib)
    if bn_backup:
        model.load_state_dict(bn_backup, strict=False)
    model.eval()
    keep_modules_for_groups(wrapper, ["det_head_output"])

    # T3 mimic: the board graph stripped ALL head-side QDQ (weights refloated)
    head_mods = [n for n, _ in quantized_modules(wrapper)
                 if n.startswith("model.head.det_head.")
                 or n.startswith("model.head.map_head.")]
    set_module_quant(wrapper, head_mods, False)
    print(f"QKV surgery: {n_attn} FlashMHA modules; "
          f"T3 mimic: {len(head_mods)} head modules dequantized")
    return cfg, wrapper


def _quantizer_scales(wrapper):
    """{attr_path: np.ndarray} scale candidates (amax/127) for every enabled
    quantizer; None amax or all-zero scales skipped."""
    from ptq_sensitivity import quantized_modules
    out = {}
    for name, mod in quantized_modules(wrapper):
        for attr in dir(mod):
            if not attr.endswith("_quantizer"):
                continue
            q = getattr(mod, attr)
            amax = getattr(q, "_amax", None)
            if amax is None:
                continue
            s = amax.detach().float().cpu().numpy().reshape(-1) / 127.0
            if s.size == 0 or not np.any(np.abs(s) > 0):
                continue
            out[f"{name}.{attr}"] = s
    return out


def qdq_sentinel(wrapper):
    """Compare rebuilt amax/127 scales against v5_P1h.onnx QDQ scales."""
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(ONNX_REF)
    inits = {i.name: i for i in onnx_model.graph.initializer}
    ref_scalars, ref_vectors = [], []
    seen = set()
    for n in onnx_model.graph.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if len(n.input) < 2 or n.input[1] not in inits:
            continue
        if n.input[1] in seen:
            continue
        seen.add(n.input[1])
        arr = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        arr = arr.reshape(-1)
        (ref_scalars if arr.size == 1 else ref_vectors).append(arr)
    model_scales = _quantizer_scales(wrapper)
    mod_scalars = {k: v for k, v in model_scales.items() if v.size == 1}
    mod_vectors = {k: v for k, v in model_scales.items() if v.size > 1}

    def rel(a, b):
        d = np.abs(a - b)
        denom = np.maximum(np.abs(b), 1e-12)
        return float((d / denom).max())

    unmatched = []
    diffs = []
    for arr in ref_scalars:
        best = min((rel(arr, v), k) for k, v in mod_scalars.items())
        diffs.append(best[0])
        if best[0] > 1e-2:
            unmatched.append(("scalar", float(arr.reshape(-1)[0]), best))
    for arr in ref_vectors:
        cands = [(k, v) for k, v in mod_vectors.items()
                 if v.size == arr.size]
        if not cands:
            diffs.append(float("inf"))
            unmatched.append(("vec%d" % arr.size,
                              float(np.abs(arr).max()),
                              (float("inf"), "no same-size candidate")))
            continue
        best = min((rel(arr, v), k) for k, v in cands)
        diffs.append(best[0])
        if best[0] > 1e-2:
            unmatched.append(("vec%d" % arr.size, float(np.abs(arr).max()),
                              best))
    diffs = np.array(diffs) if diffs else np.array([0.0])
    print(f"[sentinel] onnx scales: {len(ref_scalars)} scalar + "
          f"{len(ref_vectors)} vector; model scales: "
          f"{len(mod_scalars)} scalar + {len(mod_vectors)} vector")
    print(f"[sentinel] match rel-diff: median={np.median(diffs):.2e} "
          f"max={diffs.max():.2e}; unmatched={len(unmatched)}")
    frac_ok = float((diffs <= 1e-2).mean())
    print(f"[sentinel] fraction matched <=1e-2: {frac_ok:.3f}")
    for kind, mag, best in unmatched[:12]:
        print(f"[sentinel]   unmatched {kind} mag={mag:.6g} "
              f"best={best[0]:.3g} @ {best[1]}")
    if frac_ok < 0.9:
        raise SystemExit("SENTINEL FAILED: rebuilt amax does not match the "
                         "engine graph QDQ scales - provenance broken, "
                         "refusing to run B2 inference")
    print("[sentinel] PASS")


def enable_head_fp16(wrapper):
    """P1h mimic: fp16 attention protocol + autocast over the two heads."""
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
        w = torch.softmax(w.float(), dim=-1)
        out = torch.matmul(w.half(), v16)
        return out.transpose(1, 2).contiguous().float(), None

    if os.environ.get("B2_ATTN16", "1") == "1":
        FlashAttention.forward = _fp16_forward
    use_autocast = os.environ.get("B2_AUTOCAST", "1") == "1"

    def _to_fp32(o):
        if torch.is_tensor(o) and o.is_floating_point():
            return o.float()
        return o

    for head in (wrapper.det_head, wrapper.map_head):
        orig = head.forward_onnx

        if use_autocast:
            @functools.wraps(orig)
            def fp16_forward_onnx(*a, _orig=orig, **k):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = _orig(*a, **k)
                if isinstance(out, dict):
                    return {k2: _to_fp32(v) for k2, v in out.items()}
                if isinstance(out, (list, tuple)):
                    return type(out)(_to_fp32(o) for o in out)
                return _to_fp32(out)

            head.forward_onnx = fp16_forward_onnx
    print("head compute dtype: %s + fp16 attention protocol"
          % ("fp16 (autocast)" if use_autocast else "fp32 (autocast off)"))
    print("head compute dtype: fp16 (autocast) + fp16 attention protocol")


def zero_det(nh, dim):
    return {
        "prev_det_feat": torch.zeros(1, nh, 256, device="cuda"),
        "prev_det_anchor": torch.zeros(1, nh, dim, device="cuda"),
        "prev_det_conf": torch.zeros(1, nh, device="cuda"),
        "prev_det_id": torch.full((1, nh), -1, dtype=torch.int32,
                                  device="cuda"),
        "prev_id_count": torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
    }


def zero_map(nh, dim):
    return {
        "prev_map_feat": torch.zeros(1, nh, 256, device="cuda"),
        "prev_map_anchor": torch.zeros(1, nh, dim, device="cuda"),
        "prev_map_conf": torch.zeros(1, nh, device="cuda"),
    }


def main():
    cfg, wrapper = build_quantized_model()
    qdq_sentinel(wrapper)
    enable_head_fp16(wrapper)
    wrapper.eval()

    from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (
        NuScenes3DDataset,
    )
    from mmcv.parallel.scatter_gather import scatter
    import copy
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg["test_mode"] = True
    ds_cfg.pop("type", None)
    ds = NuScenes3DDataset(**ds_cfg)
    from mmdet.datasets import build_dataloader
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    n = len(ds)
    print("dataset:", n, "samples", flush=True)

    det_head = wrapper.model.head.det_head
    map_head = wrapper.model.head.map_head
    nh_det = det_head.instance_bank.num_temp_instances
    dim_det = det_head.instance_bank.anchor.shape[-1]
    nh_map = map_head.instance_bank.num_temp_instances
    dim_map = map_head.instance_bank.anchor.shape[-1]

    hist_det = zero_det(nh_det, dim_det)
    hist_map = zero_map(nh_map, dim_map)
    prev_global, prev_time = None, None

    tokens = []
    os.makedirs(OUT, exist_ok=True)
    for k, data in enumerate(loader):
        with torch.no_grad():
            d = scatter(data, [torch.cuda.current_device()])[0]
            img = d["img"]
            proj_mat = d["projection_mat"]
            metas = d["img_metas"][0]
            curr_time = metas["timestamp"]
            if prev_time is None:
                dt, scene_start = 0.5, True
            else:
                dt = curr_time - prev_time
                scene_start = (dt > 2.0 or dt < 0)
            # B2_NO_RESET=1 复刻板端链盲传: 边界帧 dt=0.5 + identity t_matrix
            # (prep 语义, 链脚本不覆盖这两个 bin), 但历史被 cp 盖掉不清
            if scene_start:
                dt = 0.5
                if os.environ.get("B2_NO_RESET", "0") != "1":
                    hist_det = zero_det(nh_det, dim_det)
                    hist_map = zero_map(nh_map, dim_map)
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

            outs = wrapper(img, proj_mat,
                           hist_det["prev_det_feat"],
                           hist_det["prev_det_anchor"],
                           hist_det["prev_det_conf"],
                           hist_det["prev_det_id"],
                           hist_det["prev_id_count"],
                           hist_map["prev_map_feat"],
                           hist_map["prev_map_anchor"],
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
            map_cls, map_pts = outs[11], outs[12]
            bad = [nm for nm, t in
                   (("det_cls", outs[0]), ("det_bbox", outs[1]),
                    ("map_cls", outs[11]), ("map_pts", outs[12]))
                   if not torch.isfinite(t).all()]
            assert not bad, "frame %d non-finite outputs: %s" % (k, bad)

            od = os.path.join(OUT, "out_%02d" % k)
            os.makedirs(od, exist_ok=True)
            for name, t in (("map_cls", outs[11]), ("map_pts", outs[12]),
                            ("det_cls", outs[0]), ("det_bbox", outs[1]),
                            ("det_quality", outs[2]),
                            ("map_instance_feature", outs[13]),
                            ("map_anchor_embed", outs[14]),
                            ("next_map_feat", outs[15]),
                            ("next_map_anchor", outs[16]),
                            ("next_map_conf", outs[17]),
                            ("ego_feature_map", outs[18])):
                t.detach().cpu().numpy().astype(np.float32).tofile(
                    os.path.join(od, name + ".bin"))
            tokens.append(ds.data_infos[k]["token"])
            if (k + 1) % 10 == 0 or k == n - 1:
                smax = map_cls.sigmoid().max().item()
                print("frame %2d/%d scene_start=%d map_sig_max=%.4f"
                      % (k + 1, n, scene_start, smax), flush=True)

    np.savez(os.path.join(OUT, "mini_meta.npz"), tokens=np.array(tokens))
    print("dumped", n, "frames ->", OUT, flush=True)
    print("B2_INFER_DONE")


if __name__ == "__main__":
    main()

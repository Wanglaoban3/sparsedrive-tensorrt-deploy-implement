import argparse
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model

import projects.mmdet3d_plugin 

def run_input_alignment_debug(model, data_loader):
    model.eval()
    det_head = model.head.det_head
    nh_det = det_head.instance_bank.num_temp_instances 
    
    def get_zero_history():
        return {
            'prev_instance_feature': torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_det, 11), dtype=torch.float32, device='cuda'),
        }

    history = get_zero_history()
    prev_global_mat = None
    prev_time = None

    print("\n" + "=" * 50)
    print(" 🚀 STARTING PURE INPUT TENSOR ALIGNMENT ")
    print("=" * 50)

    for i, data in enumerate(data_loader):
        if i > 2: break
            
        print(f"\n[{' Frame ' + str(i) + ' ':=^40}]")
        with torch.no_grad():
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            img = scattered_data['img']
            kwargs = scattered_data.copy()
            kwargs.pop('img')
            img_metas = kwargs['img_metas'][0]

            curr_time = img_metas['timestamp']
            if prev_time is None:
                dt = 0.5
                is_scene_start = True
            else:
                dt = curr_time - prev_time
                is_scene_start = (dt > 2.0 or dt < 0)

            if is_scene_start:
                dt = 0.5
                history = get_zero_history()
                prev_global_mat = None

            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            prev_time = curr_time

            curr_global = img_metas['T_global']
            curr_global_inv = img_metas['T_global_inv']
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = curr_global_inv @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
            prev_global_mat = curr_global

            features = model.extract_feat(img, metas=kwargs)

            # ---------------------------------------------------------
            # 1. 抓取 Native 准备灌入模型的参数
            # ---------------------------------------------------------
            native_inputs = det_head.forward_debug(features, kwargs)

            # ---------------------------------------------------------
            # 2. 抓取 ONNX 准备灌入模型的参数
            # ---------------------------------------------------------
            onnx_inputs = det_head.forward_onnx_debug(
                feature_maps=features,
                prev_instance_feature=history['prev_instance_feature'],
                prev_anchor=history['prev_anchor'],
                instance_t_matrix=instance_t_matrix,
                time_interval=dt_tensor
            )

            # ---------------------------------------------------------
            # 3. 逐个变量查水表！
            # ---------------------------------------------------------
            keys_to_compare = [
                "time_interval", 
                "instance_feature", 
                "anchor", 
                "anchor_embed", 
                "temp_instance_feature", 
                "temp_anchor", 
                "temp_anchor_embed"
            ]

            for key in keys_to_compare:
                v_nat = native_inputs[key]
                v_onnx = onnx_inputs[key]

                if v_nat is None and v_onnx is None:
                    print(f"  ✅ {key:<22}: Both are None")
                    continue
                if v_nat is not None and v_onnx is not None:
                    err = (v_nat - v_onnx).abs().max().item()
                    if err > 1e-4:
                        print(f"  ❌ {key:<22}: MAX Error = {err:.6e}  <-- 【问题在此!】")
                    else:
                        print(f"  ✅ {key:<22}: MAX Error = {err:.6e}")
                else:
                    print(f"  ❌ {key:<22}: One is None, the other is NOT! (Fatal)")

            # ---------------------------------------------------------
            # 4. 执行真正的 Native 前向推理，刷新 InstanceBank 真实状态
            # ---------------------------------------------------------
            _ = det_head(features, kwargs)
            
            # 【关键】把 Native 产生的最真实的缓存喂给下一帧的 ONNX，切断误差累积！
            if det_head.instance_bank.cached_feature is not None:
                history['prev_instance_feature'] = det_head.instance_bank.cached_feature.clone()
                history['prev_anchor'] = det_head.instance_bank.cached_anchor.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_map'] = False
        cfg.task_config['with_motion_plan'] = False
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir).split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config).split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                plg_lib = importlib.import_module(_module_path)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_origin(
        dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    model = model.cuda()
    run_input_alignment_debug(model, data_loader)

if __name__ == "__main__":
    main()
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
import projects.mmdet3d_plugin 

def run_map_alignment_debug(model, data_loader):
    model.eval()
    
    map_head = model.head.map_head
    nh_map = map_head.instance_bank.num_temp_instances 
    dim_anchor = map_head.instance_bank.anchor.shape[-1]
    
    def get_zero_history():
        return {
            'prev_instance_feature': torch.zeros((1, nh_map, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_map, dim_anchor), dtype=torch.float32, device='cuda'),
        }

    history = get_zero_history()
    prev_global_mat = None
    prev_time = None

    print("\n" + "=" * 50)
    print(" 🗺️  STARTING PURE MAP INPUT ALIGNMENT ")
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

            # ================== 核心防护网 ==================
            # 因为 forward_debug 会调用 get() 原地修改 cached_anchor
            # 我们必须在调用前先给它留个快照！
            bkp_cached_anchor = None
            if map_head.instance_bank.cached_anchor is not None:
                bkp_cached_anchor = map_head.instance_bank.cached_anchor.clone()

            # 1. 提取 Native 输入 (污染了 cached_anchor)
            native_inputs = map_head.forward_debug(features, kwargs)

            # 恢复快照，消除污染！
            if bkp_cached_anchor is not None:
                map_head.instance_bank.cached_anchor = bkp_cached_anchor

            # 2. 提取 ONNX 输入
            try:
                onnx_inputs = map_head.forward_onnx_debug(
                    feature_maps=features,
                    prev_instance_feature=history['prev_instance_feature'],
                    prev_anchor=history['prev_anchor'],
                    instance_t_matrix=instance_t_matrix,
                    time_interval=dt_tensor
                )
            except Exception as e:
                print(f"  ❌ 崩溃了！错误信息: {e}")
                return
            # ================================================

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
                        print(f"  ❌ {key:<22}: MAX Error = {err:.6e}  <-- 【不对齐】")
                    else:
                        print(f"  ✅ {key:<22}: MAX Error = {err:.6e}")
                else:
                    print(f"  ❌ {key:<22}: One is None, the other is NOT! (Fatal)")

            # 我们不要跑全前向，只更新历史，用原生最正确的缓存来喂给下一帧的 ONNX
            # 因为这里我们只测输入！
            _ = map_head(features, kwargs)
            if map_head.instance_bank.cached_feature is not None:
                history['prev_instance_feature'] = map_head.instance_bank.cached_feature.clone()
                history['prev_anchor'] = map_head.instance_bank.cached_anchor.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_motion_plan'] = False
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_origin(
        dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    # 强制注释掉 FP16，彻底排除舍入误差干扰
    # fp16_cfg = cfg.get("fp16", None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)

    model = model.cuda()
    run_map_alignment_debug(model, data_loader)

if __name__ == "__main__":
    main()
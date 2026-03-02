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

def run_unified_alignment(model, data_loader):
    model.eval()
    
    # 🔪 魔法操作：物理级抹杀一切潜在的 FP16 混合精度！
    model.float() 
    
    det_head = model.head.det_head
    map_head = model.head.map_head
    
    nh_det = det_head.instance_bank.num_temp_instances 
    dim_det = det_head.instance_bank.anchor.shape[-1]
    
    nh_map = map_head.instance_bank.num_temp_instances 
    dim_map = map_head.instance_bank.anchor.shape[-1]
    
    def get_zero_history(nh, dim):
        return {
            'prev_instance_feature': torch.zeros((1, nh, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh, dim), dtype=torch.float32, device='cuda'),
            'prev_confidence': torch.zeros((1, nh), dtype=torch.float32, device='cuda'),
        }

    history_det = get_zero_history(nh_det, dim_det)
    history_map = get_zero_history(nh_map, dim_map)
    
    prev_global_mat = None
    prev_time = None

    print("\n" + "🔥" * 35)
    print(" 🚗🗺️  STARTING DET & MAP UNIFIED ALIGNMENT ")
    print("🔥" * 35)

    for i, data in enumerate(data_loader):
        if i > 2: 
            break
            
        print(f"\n[{' Frame ' + str(i) + ' ':=^40}]")
        with torch.no_grad():
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            
            # 🔪 强转图像张量为 FP32
            img = scattered_data['img'].float() 
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
                history_det = get_zero_history(nh_det, dim_det)
                history_map = get_zero_history(nh_map, dim_map)
                prev_global_mat = None

            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            prev_time = curr_time

            # 实例运动补偿矩阵
            curr_global = img_metas['T_global']
            curr_global_inv = img_metas['T_global_inv']
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = curr_global_inv @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
            prev_global_mat = curr_global

            # 提取共同的图像特征
            features = model.extract_feat(img, metas=kwargs)
            # 🔪 物理确保所有特征进入 Head 前是纯净 FP32
            features = [f.float() for f in features] 

            # ==========================================================
            # 🚗 DETECTOR HEAD ALIGNMENT
            # ==========================================================
            try:
                onnx_outs_det = det_head.forward_onnx(
                    feature_maps=features,
                    prev_instance_feature=history_det['prev_instance_feature'],
                    prev_anchor=history_det['prev_anchor'],
                    instance_t_matrix=instance_t_matrix,
                    time_interval=dt_tensor,
                    prev_confidence=history_det['prev_confidence'],
                    metas=kwargs
                )
            except Exception as e:
                print(f"  ❌ Det Head forward_onnx 崩溃了！错误信息: {e}")
                return

            native_outs_det = det_head(features, kwargs)
            
            err_cls_det = (native_outs_det['classification'][-1] - onnx_outs_det['cls_scores']).abs().max().item()
            err_box_det = (native_outs_det['prediction'][-1] - onnx_outs_det['bbox_preds']).abs().max().item()
            
            print(f"  🚗 [Det] 分类误差 cls_score : {err_cls_det:.6e} " + ("✅" if err_cls_det < 1e-4 else "❌"))
            print(f"  🚗 [Det] 回归误差 bbox_preds: {err_box_det:.6e} " + ("✅" if err_box_det < 1e-4 else "❌"))

            if det_head.instance_bank.cached_anchor is not None:
                err_cache_det = (det_head.instance_bank.cached_anchor - onnx_outs_det['next_anchor']).abs().max().item()
                print(f"  🚗 [Det] 下帧缓存 next_anchor: {err_cache_det:.6e} " + ("✅" if err_cache_det < 1e-4 else "❌"))
                
                history_det['prev_instance_feature'] = det_head.instance_bank.cached_feature.clone()
                history_det['prev_anchor'] = det_head.instance_bank.cached_anchor.clone()
                if hasattr(det_head.instance_bank, 'confidence') and det_head.instance_bank.confidence is not None:
                    history_det['prev_confidence'] = det_head.instance_bank.confidence.clone()

            # ==========================================================
            # 🗺️ MAP HEAD ALIGNMENT
            # ==========================================================
            try:
                onnx_outs_map = map_head.forward_onnx(
                    feature_maps=features,
                    prev_instance_feature=history_map['prev_instance_feature'],
                    prev_anchor=history_map['prev_anchor'],
                    instance_t_matrix=instance_t_matrix,
                    time_interval=dt_tensor,
                    prev_confidence=history_map['prev_confidence'],
                    metas=kwargs
                )
            except Exception as e:
                print(f"  ❌ Map Head forward_onnx 崩溃了！错误信息: {e}")
                return

            native_outs_map = map_head(features, kwargs)
            
            err_cls_map = (native_outs_map['classification'][-1] - onnx_outs_map['cls_scores']).abs().max().item()
            err_pts_map = (native_outs_map['prediction'][-1] - onnx_outs_map['bbox_preds']).abs().max().item()
            
            print(f"  🗺️  [Map] 分类误差 cls_score : {err_cls_map:.6e} " + ("✅" if err_cls_map < 1e-4 else "❌"))
            print(f"  🗺️  [Map] 几何误差 pts_preds : {err_pts_map:.6e} " + ("✅" if err_pts_map < 1e-4 else "❌"))

            if map_head.instance_bank.cached_anchor is not None:
                err_cache_map = (map_head.instance_bank.cached_anchor - onnx_outs_map['next_anchor']).abs().max().item()
                print(f"  🗺️  [Map] 下帧缓存 next_anchor: {err_cache_map:.6e} " + ("✅" if err_cache_map < 1e-4 else "❌"))
                
                history_map['prev_instance_feature'] = map_head.instance_bank.cached_feature.clone()
                history_map['prev_anchor'] = map_head.instance_bank.cached_anchor.clone()
                if hasattr(map_head.instance_bank, 'confidence') and map_head.instance_bank.confidence is not None:
                    history_map['prev_confidence'] = map_head.instance_bank.confidence.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # 关闭无关任务，纯净排查
    if hasattr(cfg, 'task_config'):
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
    
    # 这里我们不用 wrap_fp16_model，而在上面的循环中直接使用 .float() 进行全链路清洗！
    model = model.cuda()
    
    run_unified_alignment(model, data_loader)

if __name__ == "__main__":
    main()
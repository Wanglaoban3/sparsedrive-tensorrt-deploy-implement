# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mmdet3d
import projects.mmdet3d_plugin
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model

# ==============================================================================
# 🔄 纯 PyTorch 验证 Det & Map 的 forward_onnx 全链路逻辑
# ==============================================================================
def pytorch_unified_forward_onnx_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    # 获取 Det 和 Map 的独立配置
    det_head = model.module.head.det_head
    map_head = model.module.head.map_head

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

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            img = scattered_data['img']
            kwargs = scattered_data.copy()
            kwargs.pop('img')
            img_metas = kwargs['img_metas'][0]

            curr_time = img_metas['timestamp']
            
            # 🚀 时间跳变精准捕捉“场景切换”
            if prev_time is None:
                dt = 0.5
                is_scene_start = True
            else:
                dt = curr_time - prev_time
                is_scene_start = (dt > 2.0 or dt < 0)

            # 新场景开头，彻底清空双头的历史毒药！
            if is_scene_start:
                dt = 0.5
                history_det = get_zero_history(nh_det, dim_det)
                history_map = get_zero_history(nh_map, dim_map)
                prev_global_mat = None 

            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            prev_time = curr_time

            # 运动补偿结算
            curr_global = img_metas['T_global']
            curr_global_inv = img_metas['T_global_inv']
            
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = curr_global_inv @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
            prev_global_mat = curr_global

            # 统一提取图像特征
            features = model.module.extract_feat(img, metas=kwargs)

            # ==========================================
            # 1. 执行 Det 头 forward_onnx
            # ==========================================
            outs_det = det_head.forward_onnx(
                feature_maps=features,
                prev_instance_feature=history_det['prev_instance_feature'],
                prev_anchor=history_det['prev_anchor'],
                instance_t_matrix=instance_t_matrix,
                time_interval=dt_tensor,
                prev_confidence=history_det['prev_confidence'],
                metas=kwargs
            )
            history_det['prev_instance_feature'] = outs_det['next_instance_feature']
            history_det['prev_anchor'] = outs_det['next_anchor']
            history_det['prev_confidence'] = outs_det['next_confidence']

            model_outs_det = {
                "classification": [outs_det['cls_scores']],
                "prediction": [outs_det['bbox_preds']],
                "quality": [outs_det['quality']],    
                "instance_id": None   
            }
            decoded_det_res = det_head.post_process(model_outs_det)

            # ==========================================
            # 2. 执行 Map 头 forward_onnx
            # ==========================================
            outs_map = map_head.forward_onnx(
                feature_maps=features,
                prev_instance_feature=history_map['prev_instance_feature'],
                prev_anchor=history_map['prev_anchor'],
                instance_t_matrix=instance_t_matrix,
                time_interval=dt_tensor,
                prev_confidence=history_map['prev_confidence'],
                metas=kwargs
            )
            history_map['prev_instance_feature'] = outs_map['next_instance_feature']
            history_map['prev_anchor'] = outs_map['next_anchor']
            history_map['prev_confidence'] = outs_map['next_confidence']

            model_outs_map = {
                "classification": [outs_map['cls_scores']],
                "prediction": [outs_map['bbox_preds']],
                "quality": [outs_map.get('quality', None)],
                "instance_id": None   
            }
            decoded_map_res = map_head.post_process(model_outs_map)

            # ==========================================
            # 3. 💡 合并 Det 和 Map 的结果到一个字典中
            # ==========================================
            merged_res = decoded_det_res[0].copy()
            merged_res.update(decoded_map_res[0])
            
            # 这样 format_results 就只会遍历 img_bbox/pts_bbox
            # Map 评测脚本也会自动从 img_bbox 里面去提取 vectors
            result = {
                'img_bbox': merged_res, 
                'pts_bbox': merged_res,
            }
            results.append(result)

        prog_bar.update()

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Test Unified Det & Map forward_onnx")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    parser.add_argument("--eval", type=str, nargs="+", default=['bbox', 'map'], help='evaluation metrics')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'task_config'):
        # 🟢 开启 Det 和 Map，关闭 Motion Plan 以免报错
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = False
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir

    samples_per_gpu = 1
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    
    dataset.work_dir = cfg.work_dir
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        
    model = model.cuda()
    model = mmcv.parallel.MMDataParallel(model, device_ids=[0])
    model.CLASSES = dataset.CLASSES

    print("\n🔥 Starting Unified Evaluation using PyTorch forward_onnx() ...")
    outputs = pytorch_unified_forward_onnx_test(model, data_loader)
    
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
            
        if 'eval_mode' in eval_kwargs:
            eval_kwargs['eval_mode']['with_tracking'] = False
            eval_kwargs['eval_mode']['with_motion'] = False
            eval_kwargs['eval_mode']['with_planning'] = False
            eval_kwargs['eval_mode']['with_det'] = True
            eval_kwargs['eval_mode']['with_map'] = True
        
        eval_kwargs.update(dict(metric=args.eval))
        print(f"\n📊 Evaluating metrics: {eval_kwargs}")
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

if __name__ == "__main__":
    main()
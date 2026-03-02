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

# 核心导入


# ==============================================================================
# 🔄 纯 PyTorch 验证 forward_onnx 逻辑 (终极修复版)
# ==============================================================================
def pytorch_forward_onnx_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    nh_det = 600
    
    def get_zero_history():
        return {
            'prev_instance_feature': torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_det, 11), dtype=torch.float32, device='cuda'),
            'prev_confidence': torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
        }

    history = get_zero_history()
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
            
            # 🚀 终极修复：通过时间跳变来精准捕捉“场景切换”
            if prev_time is None:
                dt = 0.5
                is_scene_start = True
            else:
                dt = curr_time - prev_time
                is_scene_start = (dt > 2.0 or dt < 0)

            # 如果是新场景的开头，彻底清空历史毒药！
            if is_scene_start:
                dt = 0.5
                history = get_zero_history()
                prev_global_mat = None  # 防止计算出跨场景的疯狂 T_temp2cur

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

            features = model.extract_feat(img, metas=kwargs)

            outs = model.head.det_head.forward_onnx(
                feature_maps=features,
                prev_instance_feature=history['prev_instance_feature'],
                prev_anchor=history['prev_anchor'],
                instance_t_matrix=instance_t_matrix,
                time_interval=dt_tensor,
                prev_confidence=history['prev_confidence'],
                metas=kwargs
            )

            history['prev_instance_feature'] = outs['next_instance_feature']
            history['prev_anchor'] = outs['next_anchor']
            history['prev_confidence'] = outs['next_confidence']

            model_outs = {
                "classification": [outs['cls_scores']],
                "prediction": [outs['bbox_preds']],
                "quality": [None],    
                "instance_id": None   
            }
            
            decoded_res = model.head.det_head.post_process(model_outs)
            result = {'img_bbox': decoded_res[0], 'pts_bbox': decoded_res[0]}
            results.append(result)

        prog_bar.update()

    return results

# ==============================================================================
# 🎮 参数解析与主函数
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test PyTorch forward_onnx")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    parser.add_argument("--eval", type=str, nargs="+", help='evaluation metrics, e.g., "bbox"')
    args = parser.parse_args()
    return args

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
    
    # 强制开启 FP16 保证与原本原生环境验证的对齐
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    print("\n🔥 Starting Evaluation using PyTorch forward_onnx() ...")
    outputs = pytorch_forward_onnx_test(model, data_loader)
    
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
            
        if 'eval_mode' in eval_kwargs:
            eval_kwargs['eval_mode']['with_tracking'] = False
            eval_kwargs['eval_mode']['with_map'] = False
            eval_kwargs['eval_mode']['with_motion'] = False
            eval_kwargs['eval_mode']['with_planning'] = False
        
        eval_kwargs.update(dict(metric=args.eval))
        print(f"\n📊 Evaluating metrics: {eval_kwargs}")
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

if __name__ == "__main__":
    main()
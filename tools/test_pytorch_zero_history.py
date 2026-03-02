# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
from os import path as osp
import sys

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
import projects.mmdet3d_plugin

# ==============================================================================
# 🔄 纯 PyTorch 测试 Loop (完美手工维护历史缓存 - 持续滚动版)
# ==============================================================================
def pytorch_perfect_history_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    det_bank = model.module.head.det_head.instance_bank
    history_cache = {}

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # 🛑 1. 注入：外部干预 InstanceBank 缓存 (移除场景重置，无脑滚动)
            if 'cached_feature' in history_cache:
                det_bank.cached_feature = history_cache['cached_feature'].clone()
                det_bank.cached_anchor = history_cache['cached_anchor'].clone()
                det_bank.metas = history_cache['metas'] 
                
                if 'confidence' in history_cache and history_cache['confidence'] is not None:
                    det_bank.confidence = history_cache['confidence'].clone()
                if 'temp_confidence' in history_cache and history_cache['temp_confidence'] is not None:
                    det_bank.temp_confidence = history_cache['temp_confidence'].clone()
                if 'instance_id' in history_cache and history_cache['instance_id'] is not None:
                    det_bank.instance_id = history_cache['instance_id'].clone()
            else:
                # 只有整个数据集的第 0 帧才会走这里，跟原生 test.py 保持完全一致
                det_bank.reset()

            # 🚀 2. 推理：使用完整的 MMDataParallel 前向传播
            result = model(return_loss=False, rescale=True, **data)
            results.extend(result)

            # ♻️ 3. 提取：帧结束时，扣出模型内更新好的缓存状态，供下一帧注入
            if det_bank.cached_feature is not None:
                history_cache['cached_feature'] = det_bank.cached_feature.clone()
                history_cache['cached_anchor'] = det_bank.cached_anchor.clone()
                history_cache['metas'] = det_bank.metas # 直接引用保存
                
                if hasattr(det_bank, 'confidence') and det_bank.confidence is not None:
                    history_cache['confidence'] = det_bank.confidence.clone()
                if hasattr(det_bank, 'temp_confidence') and det_bank.temp_confidence is not None:
                    history_cache['temp_confidence'] = det_bank.temp_confidence.clone()
                if hasattr(det_bank, 'instance_id') and det_bank.instance_id is not None:
                    history_cache['instance_id'] = det_bank.instance_id.clone()

        prog_bar.update()

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Perfect Manual History Test")
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

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
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
    
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])

    print("\n🔥 Starting PyTorch Evaluation with PERFECT MANUAL HISTORY INJECTION...")
    outputs = pytorch_perfect_history_test(model, data_loader)
    
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
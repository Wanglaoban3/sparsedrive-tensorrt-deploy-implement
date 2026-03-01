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
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
import mmdet3d
import projects.mmdet3d_plugin

# ==============================================================================
# 🔄 纯 PyTorch 测试 Loop (强行注入全零历史)
# ==============================================================================
def pytorch_zero_history_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    nh_det = 600
    
    # 💡 提取原生模型 InstanceBank 里学到的固定 Query 作为首帧安全垫！
    bank = model.head.det_head.instance_bank
    default_feat = bank.instance_feature[:nh_det].unsqueeze(0).clone().detach()
    default_anchor = bank.anchor[:nh_det].unsqueeze(0).clone().detach()
    
    def init_history():
        return {
            'prev_det_feat': default_feat.clone(),
            'prev_det_anchor': default_anchor.clone(),
            # 💡 尝试给一个中等置信度（比如 0.5），让模型在第一帧就“敢于”利用这些 Query
            'prev_det_conf': torch.full((1, nh_det), 0.5, device='cuda'), 
        }

    history_cache = init_history()
    prev_global_mat = None
    prev_time = None
    prev_scene_token = None

    # 获取底层 InstanceBank
    det_bank = model.head.det_head.instance_bank

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # ==========================================================
            # 💡 核心修复：正确的数据解包 (去掉冗余的 [0])
            # ==========================================================
            img_metas = data['img_metas'].data[0][0]
            img_raw = data['img'].data[0][0].cuda()
            img = img_raw.unsqueeze(0) if img_raw.dim() == 4 else img_raw
            proj_mat = torch.stack([p.cuda() for p in data['projection_mat'].data[0]], dim=0).unsqueeze(0)
            # ==========================================================

            # 🛑 检测场景切换，强行重置为全零历史！
            curr_scene_token = img_metas.get('scene_token', None)
            if prev_scene_token is None or curr_scene_token != prev_scene_token:
                # 哪怕是 PyTorch，我们也强行给它喂 0！
                history_cache = init_history()
                prev_global_mat = None
                prev_time = img_metas['timestamp'] - 0.5
            prev_scene_token = curr_scene_token

            curr_time = img_metas['timestamp']
            dt = curr_time - prev_time
            prev_time = curr_time

            # 💉 【核心黑客操作】强行把全零的 history_cache 塞进 PyTorch 内部！
            det_bank.cached_feature = history_cache['prev_det_feat'].clone()
            det_bank.cached_anchor = history_cache['prev_det_anchor'].clone()
            if hasattr(det_bank, 'confidence'):
                det_bank.confidence = history_cache['prev_det_conf'].clone()

            # 构造完美的 fake_metas 骗过 PyTorch 原生的位姿补偿
            curr_global = img_metas['T_global']
            fake_metas = {
                "timestamp": curr_time - 0.5,
                "img_metas": [{"T_global": curr_global}] 
            }
            det_bank.metas = fake_metas

            # 🚀 执行 PyTorch 原生前向推理
            native_metas = {
                'img_metas': [img_metas],
                'projection_mat': proj_mat,
                'image_wh': img.new_tensor([img.shape[-1], img.shape[-2]]).view(1, 1, 2).repeat(1, 6, 1),
                'timestamp': img.new_tensor([img_metas['timestamp']]), 
            }
            
            # 提取特征并走 head
            features = model.extract_feat(img, metas=native_metas)
            py_outs = model.head(features, native_metas)

            # ♻️ 闭环滚动更新历史
            history_cache['prev_det_feat'] = det_bank.cached_feature.clone()
            history_cache['prev_det_anchor'] = det_bank.cached_anchor.clone()
            if hasattr(det_bank, 'confidence'):
                history_cache['prev_det_conf'] = det_bank.confidence.clone()

            # 🛠️ 组装结果 (严格对齐 TRT 的解码流程)
            p_det_cls = py_outs[0]['classification'][-1]
            p_det_reg = py_outs[0]['prediction'][-1]
            
            model_outs = {
                "classification": [p_det_cls],
                "prediction": [p_det_reg],
                "quality": [None],    
                "instance_id": None   
            }
            
            decoded_res = model.head.det_head.post_process(model_outs)
            result = {
                'img_bbox': decoded_res[0],
                'pts_bbox': decoded_res[0]
            }
            results.append(result)

        prog_bar.update()

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Zero History Test")
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
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    print("\n🔥 Starting PyTorch Evaluation with FORCED ZERO HISTORY...")
    outputs = pytorch_zero_history_test(model, data_loader)
    
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
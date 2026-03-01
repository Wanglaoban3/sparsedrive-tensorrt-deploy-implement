# Copyright (c) OpenMMLab. All rights reserved.
import tensorrt as trt

import argparse
import mmcv
import os
from os import path as osp
import sys
import ctypes
from collections import OrderedDict
# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
# 💡 核心修复：显式导入 mmdet3d 和自定义插件，激活注册表！
import mmdet3d
import projects.mmdet3d_plugin
# ==============================================================================
# 🚀 1. TensorRT 推理引擎封装
# ==============================================================================
class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT context.")

        self.inputs, self.outputs, self.bindings = OrderedDict(), OrderedDict(), []
        
        for i in range(self.engine.num_bindings):
            if hasattr(self.engine, 'get_binding_name'):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
            else:
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            shape = [s if s > 0 else 1 for s in shape]
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
            gpu_mem = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
            self.bindings.append(gpu_mem.data_ptr())
            
            if is_input:
                self.inputs[name] = gpu_mem
            else:
                self.outputs[name] = gpu_mem

    def infer(self, feed_dict):
        for name, data in feed_dict.items():
            if name in self.inputs:
                self.inputs[name].copy_(data.to(self.inputs[name].dtype).contiguous())
        self.context.execute_v2(self.bindings)
        return {name: mem.clone().float() for name, mem in self.outputs.items()}

# ==============================================================================
# 🔄 2. TensorRT 单卡测试 Loop (管理时序逻辑与后处理)
# ==============================================================================
def trt_single_gpu_test(model, trt_engine, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    nh_det = 600
    
    def init_history():
        return {
            'prev_det_feat': torch.zeros((1, nh_det, 256), device='cuda'),
            'prev_det_anchor': torch.zeros((1, nh_det, 11), device='cuda'),
            'prev_det_conf': torch.zeros((1, nh_det), device='cuda'),
        }

    history_trt = init_history()
    prev_global_mat = None
    prev_time = None
    prev_scene_token = None

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # ==========================================================
            # 💡 核心修复：还原为你之前跑通的完美数据解包逻辑
            # ==========================================================
            img_metas = data['img_metas'].data[0][0]
            
            img_raw = data['img'].data[0][0].cuda()
            # 确保 img 的维度是 [B, N, C, H, W] 即 [1, 6, 3, 256, 704]
            img = img_raw.unsqueeze(0) if img_raw.dim() == 4 else img_raw
            
            proj_mat = torch.stack([p.cuda() for p in data['projection_mat'].data[0]], dim=0).unsqueeze(0)
            # ==========================================================

            # 🛑 核心时序逻辑：检测场景切换，自动清空历史缓存！
            curr_scene_token = img_metas.get('scene_token', None)
            if prev_scene_token is None or curr_scene_token != prev_scene_token:
                history_trt = init_history()
                prev_global_mat = None
                prev_time = img_metas['timestamp'] - 0.5
            prev_scene_token = curr_scene_token

            # 计算 Ego-Motion
            curr_time = img_metas['timestamp']
            dt = curr_time - prev_time
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

            # 🚀 执行 TensorRT 推理
            feed_dict = {
                'img': img,
                'projection_mat': proj_mat,
                'instance_t_matrix': instance_t_matrix,
                'time_interval': dt_tensor,
                **history_trt
            }
            trt_outs = trt_engine.infer(feed_dict)

            # 更新历史缓存
            history_trt['prev_det_feat'] = trt_outs['next_det_feat']
            history_trt['prev_det_anchor'] = trt_outs['next_det_anchor']
            history_trt['prev_det_conf'] = trt_outs['next_det_conf']

            # 🛠️ 桥接官方后处理 (Post Process)
            # 构造模型输出格式，交由原生 Decoder 解析出 bbox, score, label
            model_outs = {
                "classification": [trt_outs['det_cls']],
                "prediction": [trt_outs['det_bbox']],
                "quality": [None],    # 💡 核心修复：伪造一个单元素的列表，防止切片报错
                "instance_id": None   # 顺手把 instance_id 也补齐，以防万一
            }
            
            # 调用 PyTorch 原生后处理 (将网络裸输出转为 3D BBox 格式)
            decoded_res = model.head.det_head.post_process(model_outs)
            
            # MMDetection3D 标准输出格式
            result = {'img_bbox': decoded_res[0]}    # ✅ 正确：直接存成字典
            results.append(result)

        # 进度条更新
        prog_bar.update()

    return results

# ==============================================================================
# 🎮 3. 命令行参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Model Test")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file (for decoding settings)")
    parser.add_argument("--engine", required=True, help="TensorRT engine file path")
    parser.add_argument("--plugin", default="projects/trt_plugin/build/libSparseDrivePlugin.so", help="Plugin path")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--eval", type=str, nargs="+", help='evaluation metrics, e.g., "bbox"')
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 载入自定义插件
    if os.path.exists(args.plugin):
        ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ Loaded Custom Plugin: {args.plugin}")

    cfg = Config.fromfile(args.config)
    
    # 强制只测试 DET 任务
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_map'] = False
        cfg.task_config['with_motion_plan'] = False
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    # 数据集准备
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir

    # 数据集准备
    samples_per_gpu = 1
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    
    # 强行给 dataset 绑定 work_dir，防止 nuScenes eval kit 找不到保存路径
    dataset.work_dir = cfg.work_dir
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # 模型准备 (只用于提供后处理配置参数，不参与推理)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    # 初始化 TRT 引擎
    print(f"\n🚀 Loading TensorRT Engine: {args.engine}")
    trt_model = TRTInfer(args.engine)

    # 开始推理
    print("\n🔥 Starting TensorRT Evaluation...")
    outputs = trt_single_gpu_test(model, trt_model, data_loader)

    # 评估与保存
    if args.out:
        print(f"\n💾 Writing results to {args.out}")
        mmcv.dump(outputs, args.out)
    
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
            
        # ==========================================================
        # 💡 核心修复：关掉不需要的 Tracking 和 Map 评测，避免 ID 报错
        # ==========================================================
        if 'eval_mode' in eval_kwargs:
            eval_kwargs['eval_mode']['with_tracking'] = False
            eval_kwargs['eval_mode']['with_map'] = False
            eval_kwargs['eval_mode']['with_motion'] = False
            eval_kwargs['eval_mode']['with_planning'] = False
        # ==========================================================
        
        eval_kwargs.update(dict(metric=args.eval))
        print(f"\n📊 Evaluating metrics: {eval_kwargs}")
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

if __name__ == "__main__":
    main()
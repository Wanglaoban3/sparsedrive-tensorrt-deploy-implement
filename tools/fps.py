# Copyright (c) OpenMMLab. All rights reserved.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorrt as trt
import argparse
import mmcv
import os
from os import path as osp
import sys
import ctypes
import time
from collections import OrderedDict

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel.scatter_gather import scatter
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

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
            if hasattr(self.engine, 'get_tensor_name'):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            else:
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
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
                self.inputs[name].copy_(data.to(self.inputs[name].dtype))
        self.context.execute_v2(self.bindings)
        return self.outputs

# ==============================================================================
# ⏱️ 2. 核心测速逻辑 (维度自适应版)
# ==============================================================================
def adapt_dim(data, target_dim):
    """自动截断或填充零向量以匹配引擎维度"""
    curr_dim = data.shape[1]
    if curr_dim > target_dim:
        return data[:, :target_dim]
    elif curr_dim < target_dim:
        pad_shape = list(data.shape)
        pad_shape[1] = target_dim - curr_dim
        padding = torch.zeros(pad_shape, device=data.device, dtype=data.dtype)
        return torch.cat([data, padding], dim=1)
    return data

# ==============================================================================
# ⏱️ 2. 极限测速逻辑 (完全消除循环内非推理开销)
# ==============================================================================
def benchmark_model(model, data_loader, args, trt_engines=None):
    model.eval()
    num_warmup = args.warmup
    num_benchmark = args.benchmark
    total_frames = num_warmup + num_benchmark
    
    # 📦 1. 预加载并对齐单帧数据 (仅执行一次)
    data_iter = iter(data_loader)
    raw_data = next(data_iter)
    scattered_data = scatter(raw_data, [torch.cuda.current_device()])[0]
    
    # 提取静态输入
    img_fixed = scattered_data['img'].cuda()
    proj_mat_fixed = scattered_data['projection_mat'].cuda()
    img_metas = scattered_data['img_metas'][0]
    curr_global_inv = torch.from_numpy(img_metas['T_global_inv']).float().cuda()
    prev_global_mat = torch.from_numpy(img_metas['T_global']).float().cuda()
    instance_t_matrix_fixed = (curr_global_inv @ prev_global_mat).unsqueeze(0)
    dt_tensor_fixed = torch.tensor([0.5], device='cuda', dtype=torch.float32)
    mask_tensor_fixed = torch.tensor([True], device='cuda', dtype=torch.bool)

    if args.mode == 'trt':
        _, det_map_engine_temporal, _, motion_engine_temporal = trt_engines
        
        # 获取引擎所需的维度
        N_DET = motion_engine_temporal.inputs['det_instance_feature'].shape[1]
        N_MAP = motion_engine_temporal.inputs['map_instance_feature'].shape[1]
        Q = motion_engine_temporal.inputs['mo_history_anchor'].shape[2]
        
        # 🎯 [极限优化]：预先创建并对齐所有输入 Tensor
        # 感知引擎输入
        static_history_det = {
            'prev_instance_feature': torch.zeros((1, 900, 256), device='cuda'),
            'prev_anchor': torch.zeros((1, 900, 11), device='cuda'),
            'prev_confidence': torch.zeros((1, 900), device='cuda'),
            'prev_instance_id': torch.full((1, 900), -1, dtype=torch.int32, device='cuda'),
            'prev_id_count': torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
        }
        static_history_map = {
            'prev_instance_feature': torch.zeros((1, 100, 256), device='cuda'), # 假设输出100/33，这里预设好
            'prev_anchor': torch.zeros((1, 100, 20), device='cuda'),
            'prev_confidence': torch.zeros((1, 100), device='cuda'),
        }

        # 🎯 核心：预跑一次获取感知的输出，用于固定 Motion 的输入
        print("🛠️ 正在进行一次预推理以固定中间变量...")
        init_feed_perc = {
            'img': img_fixed, 'projection_mat': proj_mat_fixed, 
            'instance_t_matrix': instance_t_matrix_fixed, 'time_interval': dt_tensor_fixed
        }
        init_feed_perc.update(static_history_det)
        init_feed_perc.update(static_history_map)
        trt_outs_perc = det_map_engine_temporal.infer(init_feed_perc)

        # 🎯 极限优化：手动对齐并固定 Motion 的所有输入 Tensor
        static_feed_mo = {
            'det_instance_feature': adapt_dim(trt_outs_perc['det_instance_feature'], N_DET).clone(),
            'det_anchor_embed': adapt_dim(trt_outs_perc['det_anchor_embed'], N_DET).clone(),
            'det_classification_sigmoid': adapt_dim(trt_outs_perc['det_cls'], N_DET).sigmoid().clone(),
            'det_anchors': adapt_dim(trt_outs_perc['det_bbox'], N_DET).clone(),
            'det_instance_id': adapt_dim(trt_outs_perc['det_instance_id'], N_DET).to(torch.int32).clone(),
            'map_instance_feature': adapt_dim(trt_outs_perc['next_map_feat'], N_MAP).clone(),
            'map_anchor_embed': adapt_dim(trt_outs_perc['map_anchor_embed'], N_MAP).clone(),
            'map_classification_sigmoid': adapt_dim(trt_outs_perc['map_cls'], N_MAP).sigmoid().clone(),
            'ego_feature_map': trt_outs_perc['ego_feature_map'].clone(),
            'instance_t_matrix': instance_t_matrix_fixed,
            'mask': mask_tensor_fixed,
            # Motion 历史状态固定
            "mo_history_instance_feature": torch.zeros((1, N_DET, Q, 256), device='cuda'),
            "mo_history_anchor": torch.zeros((1, N_DET, Q, 11), device='cuda'),
            "mo_history_period": torch.zeros((1, N_DET), dtype=torch.int32, device='cuda'),
            "mo_prev_instance_id": torch.zeros((1, N_DET), dtype=torch.int32, device='cuda'),
            "mo_prev_confidence": torch.zeros((1, N_DET), device='cuda'),
            "mo_history_ego_feature": torch.zeros((1, 1, Q, 256), device='cuda'),
            "mo_history_ego_anchor": torch.zeros((1, 1, Q, 11), device='cuda'),
            "mo_history_ego_period": torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
            "mo_prev_ego_status": torch.zeros((1, 1, 10), device='cuda')
        }

    print(f"\n🔥 启动 [TRT] 极限测速 (完全剥离 Python 与数据处理开销)...")
    latencies = []
    prog_bar = mmcv.ProgressBar(total_frames)

    # ---------------------------------------------------------
    # 🏎️ 极限循环体
    # ---------------------------------------------------------
    for i in range(total_frames):
        if i >= num_warmup:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        with torch.no_grad():
            # 只有两行：感知推理 + 动作推理
            _ = det_map_engine_temporal.infer(init_feed_perc)
            _ = motion_engine_temporal.infer(static_feed_mo)

        if i >= num_warmup:
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start_time) * 1000)
        prog_bar.update()

    # ---------------------------------------------------------
    # 📊 报告
    # ---------------------------------------------------------
    latencies = np.array(latencies)
    avg_ms = np.mean(latencies)
    print("\n\n" + "🚀" * 15)
    print(f" 极 限 测 速 报 告 ")
    print(f" 平均耗时: {avg_ms:.2f} ms")
    print(f" 理论 FPS: {1000.0/avg_ms:.2f}")
    print("🚀" * 15 + "\n")

# ==============================================================================
# 🎮 3. 启动逻辑
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End SparseDrive Benchmark")
    parser.add_argument("config", help="config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--mode", type=str, choices=['pytorch', 'trt'], default='trt')
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--benchmark", type=int, default=200)
    parser.add_argument("--engine_perc_init", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine")
    parser.add_argument("--engine_perc_temp", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine")
    parser.add_argument("--engine_mo_init", default="work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine")
    parser.add_argument("--engine_mo_temp", default="work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine")
    parser.add_argument("--plugin", default="projects/trt_plugin/build/libSparseDrivePlugin.so")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == 'trt': ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'task_config'):
        cfg.task_config.update({'with_det':True, 'with_map':True, 'with_motion_plan':True})
        if 'head' in cfg.model: cfg.model.head.task_config = cfg.task_config

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()

    trt_engines = None
    if args.mode == 'trt':
        trt_engines = (
            TRTInfer(args.engine_perc_init), TRTInfer(args.engine_perc_temp),
            TRTInfer(args.engine_mo_init), TRTInfer(args.engine_mo_temp)
        )
    benchmark_model(model, data_loader, args, trt_engines)

if __name__ == "__main__":
    main()
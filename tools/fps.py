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
                self.inputs[name].copy_(data.to(self.inputs[name].dtype).contiguous())
        self.context.execute_v2(self.bindings)
        return {name: mem.clone() for name, mem in self.outputs.items()}

# ==============================================================================
# ⏱️ 2. 核心测速逻辑 (Pytorch vs TensorRT)
# ==============================================================================
def benchmark_model(model, data_loader, args, trt_engines=None):
    model.eval()
    dataset = data_loader.dataset
    
    num_warmup = args.warmup
    num_benchmark = args.benchmark
    total_frames = num_warmup + num_benchmark
    
    if args.mode == 'trt':
        det_map_engine_init, det_map_engine_temporal, motion_engine_init, motion_engine_temporal = trt_engines
        det_head = model.head.det_head
        map_head = model.head.map_head
        motion_head = model.head.motion_plan_head

        nh_det = det_head.instance_bank.num_temp_instances
        dim_det = det_head.instance_bank.anchor.shape[-1]
        nh_map = map_head.instance_bank.num_temp_instances
        dim_map = map_head.instance_bank.anchor.shape[-1]
        Q = motion_head.instance_queue.queue_length
        
        # ===================================================================
        # 🎯 工业级优化：只分配一次显存！后续靠 .zero_() 复用，拒绝显存碎片化
        # ===================================================================
        history_det = {
            'prev_instance_feature': torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_det, dim_det), dtype=torch.float32, device='cuda'),
            'prev_confidence': torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
            'prev_instance_id': torch.full((1, nh_det), -1, dtype=torch.int32, device='cuda'),
            'prev_id_count': torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
        }

        history_map = {
            'prev_instance_feature': torch.zeros((1, nh_map, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_map, dim_map), dtype=torch.float32, device='cuda'),
            'prev_confidence': torch.zeros((1, nh_map), dtype=torch.float32, device='cuda'),
        }

        history_motion = {
            "mo_history_instance_feature": torch.zeros((1, nh_det, Q, 256), dtype=torch.float32, device='cuda'),
            "mo_history_anchor": torch.zeros((1, nh_det, Q, 11), dtype=torch.float32, device='cuda'),
            "mo_history_period": torch.zeros((1, nh_det), dtype=torch.int32, device='cuda'),
            "mo_prev_instance_id": torch.zeros((1, nh_det), dtype=torch.int32, device='cuda'),
            "mo_prev_confidence": torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
            "mo_history_ego_feature": torch.zeros((1, 1, Q, 256), dtype=torch.float32, device='cuda'),
            "mo_history_ego_anchor": torch.zeros((1, 1, Q, 11), dtype=torch.float32, device='cuda'),
            "mo_history_ego_period": torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
            "mo_prev_ego_status": torch.zeros((1, 1, 10), dtype=torch.float32, device='cuda')
        }

        def reset_history_inplace():
            """原地清零，不产生新的内存分配"""
            for k, v in history_det.items():
                v.fill_(-1) if 'prev_instance_id' in k else v.zero_()
            for k, v in history_map.items():
                v.zero_()
            for k, v in history_motion.items():
                v.zero_()

        prev_global_mat = None
        prev_time = None
    else:
        model = MMDataParallel(model, device_ids=[0])

    print(f"\n🔥 启动 [{args.mode.upper()}] 模式 FPS 测评 (纯网络推理)...")
    print(f"   => 预热帧数 (Warmup): {num_warmup}")
    print(f"   => 测试帧数 (Benchmark): {num_benchmark}")
    
    latencies = []
    prog_bar = mmcv.ProgressBar(total_frames)
    
    data_iter = iter(data_loader)

    for i in range(total_frames):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        # 计时开始
        if i >= num_warmup:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        with torch.no_grad():
            if args.mode == 'pytorch':
                # PyTorch 原生推理 (注意: 这里包含了后处理)
                _ = model(return_loss=False, rescale=True, **data)
            
            elif args.mode == 'trt':
                # TensorRT 级联推理 (剔除后处理)
                scattered_data = scatter(data, [torch.cuda.current_device()])[0]
                img = scattered_data['img']
                proj_mat = scattered_data['projection_mat']
                img_metas = scattered_data['img_metas'][0]

                curr_time = img_metas['timestamp']
                if prev_time is None:
                    dt = 0.5
                    is_scene_start = True
                else:
                    dt = curr_time - prev_time
                    is_scene_start = (dt > 2.0 or dt < 0)

                if is_scene_start:
                    dt = 0.5
                    reset_history_inplace() # 🎯 原地清零
                    prev_global_mat = None 

                dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
                mask_tensor = torch.tensor([not is_scene_start], device='cuda', dtype=torch.bool)
                prev_time = curr_time

                curr_global = img_metas['T_global']
                curr_global_inv = img_metas['T_global_inv']
                if prev_global_mat is None:
                    instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
                else:
                    t_mat = curr_global_inv @ prev_global_mat
                    instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
                prev_global_mat = curr_global

                feed_dict_perception = {
                    'img': img,
                    'projection_mat': proj_mat,
                    'instance_t_matrix': instance_t_matrix,
                    'time_interval': dt_tensor,
                }
                feed_dict_perception.update(history_det)
                feed_dict_perception.update(history_map)

                if is_scene_start:
                    trt_outs_perc = det_map_engine_init.infer(feed_dict_perception)
                else:
                    trt_outs_perc = det_map_engine_temporal.infer(feed_dict_perception)

                # 更新状态但不重新分配内存
                history_det['prev_instance_feature'] = trt_outs_perc['next_det_feat']
                history_det['prev_anchor'] = trt_outs_perc['next_det_anchor']
                history_det['prev_confidence'] = trt_outs_perc['next_det_conf']
                history_det['prev_instance_id'] = trt_outs_perc['next_det_instance_id']
                history_det['prev_id_count'] = trt_outs_perc['next_id_count']
                history_map['prev_instance_feature'] = trt_outs_perc['next_map_feat']
                history_map['prev_anchor'] = trt_outs_perc['next_map_anchor']
                history_map['prev_confidence'] = trt_outs_perc['next_map_conf']

                feed_dict_motion = {
                    'det_instance_feature': trt_outs_perc['det_instance_feature'],
                    'det_anchor_embed': trt_outs_perc['det_anchor_embed'],
                    'det_classification_sigmoid': trt_outs_perc['det_cls'].sigmoid(),
                    'det_anchors': trt_outs_perc['det_bbox'],
                    'det_instance_id': trt_outs_perc['det_instance_id'].to(torch.int32),
                    'map_instance_feature': trt_outs_perc['map_instance_feature'],
                    'map_anchor_embed': trt_outs_perc['map_anchor_embed'],
                    'map_classification_sigmoid': trt_outs_perc['map_cls'].sigmoid(),
                    'ego_feature_map': trt_outs_perc['ego_feature_map'],
                    'instance_t_matrix': instance_t_matrix,
                    'mask': mask_tensor,
                }
                feed_dict_motion.update(history_motion)

                if is_scene_start:
                    trt_outs_mo = motion_engine_init.infer(feed_dict_motion)
                else:
                    trt_outs_mo = motion_engine_temporal.infer(feed_dict_motion)

                history_motion['mo_history_instance_feature'] = trt_outs_mo['next_mo_history_instance_feature']
                history_motion['mo_history_anchor'] = trt_outs_mo['next_mo_history_anchor']
                history_motion['mo_history_period'] = trt_outs_mo['next_mo_history_period']
                history_motion['mo_prev_instance_id'] = trt_outs_mo['next_mo_prev_instance_id']
                history_motion['mo_prev_confidence'] = trt_outs_mo['next_mo_prev_confidence']
                history_motion['mo_history_ego_feature'] = trt_outs_mo['next_mo_history_ego_feature']
                history_motion['mo_history_ego_anchor'] = trt_outs_mo['next_mo_history_ego_anchor']
                history_motion['mo_history_ego_period'] = trt_outs_mo['next_mo_history_ego_period']
                history_motion['mo_prev_ego_status'] = trt_outs_mo['next_mo_prev_ego_status']
                
                # 🎯 剔除了所有 post_process 代码，只测纯网络 Forward 耗时！

        # 计时结束
        if i >= num_warmup:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        prog_bar.update()

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    fps = 1000.0 / mean_latency
    
    print("\n\n" + "="*50)
    print(f"🚀 {args.mode.upper()} 端到端测速报告 🚀")
    print("="*50)
    print(f"总测试帧数 : {num_benchmark} frames")
    print(f"平均耗时   : {mean_latency:.2f} ms")
    print(f"P90 耗时   : {np.percentile(latencies, 90):.2f} ms")
    print(f"P99 耗时   : {np.percentile(latencies, 99):.2f} ms")
    print(f"最终 FPS   : {fps:.2f} frames/sec")
    print("="*50 + "\n")

# ==============================================================================
# 🎮 3. 命令行参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End FPS Benchmark (PyTorch vs TensorRT)")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    
    parser.add_argument("--mode", type=str, choices=['pytorch', 'trt'], default='trt', help="Run benchmark in 'pytorch' or 'trt' mode")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup frames")
    parser.add_argument("--benchmark", type=int, default=200, help="Number of benchmark frames")
    
    parser.add_argument("--engine_perc_init", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine")
    parser.add_argument("--engine_perc_temp", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine")
    parser.add_argument("--engine_mo_init", default="work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine")
    parser.add_argument("--engine_mo_temp", default="work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine")
    
    parser.add_argument("--plugin", default="projects/trt_plugin/build/libSparseDrivePlugin.so", help="Plugin path")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.mode == 'trt' and os.path.exists(args.plugin):
        ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ Loaded Custom Plugin: {args.plugin}")

    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    samples_per_gpu = 1
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    
    data_loader = build_dataloader_origin(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    if cfg.get("fp16", None) is not None and args.mode == 'pytorch':
        wrap_fp16_model(model)

    trt_engines = None
    if args.mode == 'trt':
        print(f"\n🚀 Loading TRT Engines...")
        trt_perc_init = TRTInfer(args.engine_perc_init)
        trt_perc_temp = TRTInfer(args.engine_perc_temp)
        trt_mo_init = TRTInfer(args.engine_mo_init)
        trt_mo_temp = TRTInfer(args.engine_mo_temp)
        trt_engines = (trt_perc_init, trt_perc_temp, trt_mo_init, trt_mo_temp)

    benchmark_model(model, data_loader, args, trt_engines)

if __name__ == "__main__":
    main()
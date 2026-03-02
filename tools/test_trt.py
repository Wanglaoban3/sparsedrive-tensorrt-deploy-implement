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
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

# 💡 核心导入：激活 MMDetection3D 和 SparseDrive 注册表
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
# 🔄 2. TensorRT 双引擎测试 Loop (Init Engine + Temporal Engine)
# ==============================================================================
def trt_dual_engine_test(model, engine_init, engine_temporal, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    nh_det = 600
    
    # 初始化占位字典，用于给 Init Engine 填补输入的坑位（虽然 Init Engine 内部由于 Zero Hack 会忽略它们）
    history_trt = {
        'prev_det_feat': torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda'),
        'prev_det_anchor': torch.zeros((1, nh_det, 11), dtype=torch.float32, device='cuda'),
        'prev_det_conf': torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
    }

    prev_global_mat = None
    prev_time = None

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # 🌟 使用原生 scatter 解包，彻底解决多维 Tensor 对齐偏差
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            img = scattered_data['img']
            proj_mat = scattered_data['projection_mat']
            img_metas = scattered_data['img_metas'][0]

            curr_time = img_metas['timestamp']
            
            # 🛑 核心时序判断：确定是否为第一帧
            is_first_frame = (prev_time is None)
            
            if is_first_frame:
                prev_time = curr_time - 0.5 # 假设 2Hz，往前推 0.5 秒

            dt = curr_time - prev_time
            if dt > 2.0 or dt < 0:
                dt = 0.5 
                
            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            prev_time = curr_time

            # Ego-Motion 结算
            curr_global = img_metas['T_global']
            curr_global_inv = img_metas['T_global_inv']
            
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = curr_global_inv @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
            prev_global_mat = curr_global

            # 构建输入字典
            feed_dict = {
                'img': img,
                'projection_mat': proj_mat,
                'instance_t_matrix': instance_t_matrix,
                'time_interval': dt_tensor,
                **history_trt
            }

            # 🚀 执行 TensorRT 推理：双引擎动态切换！
            if is_first_frame:
                trt_outs = engine_init.infer(feed_dict)
            else:
                trt_outs = engine_temporal.infer(feed_dict)

            # ♻️ 更新历史缓存，供下一帧（ Temporal Engine）使用
            history_trt['prev_det_feat'] = trt_outs['next_det_feat']
            history_trt['prev_det_anchor'] = trt_outs['next_det_anchor']
            history_trt['prev_det_conf'] = trt_outs['next_det_conf']

            # 🛠️ 桥接官方后处理
            model_outs = {
                "classification": [trt_outs['det_cls']],
                "prediction": [trt_outs['det_bbox']],
                "quality": [None],    
                "instance_id": None   
            }
            
            decoded_res = model.head.det_head.post_process(model_outs)
            result = {'img_bbox': decoded_res[0], 'pts_bbox': decoded_res[0]}
            results.append(result)

        prog_bar.update()

    return results

# ==============================================================================
# 🎮 3. 命令行参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Dual-Engine Test")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file (for decoding settings)")
    
    # 修改为接收两个 Engine
    parser.add_argument("--engine_init", required=True, help="Init TensorRT engine (First frame, Zero dummy input)")
    parser.add_argument("--engine_temporal", required=True, help="Temporal TensorRT engine (Sequential frames, Randn dummy input)")
    
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

    # 模型准备 (用于后处理配置解析)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    # 🔥 核心加载：初始化双引擎
    print(f"\n🚀 Loading Init Engine: {args.engine_init}")
    trt_engine_init = TRTInfer(args.engine_init)
    
    print(f"🚀 Loading Temporal Engine: {args.engine_temporal}")
    trt_engine_temporal = TRTInfer(args.engine_temporal)

    # 开始推理
    print("\n🔥 Starting TensorRT Dual-Engine Evaluation...")
    outputs = trt_dual_engine_test(model, trt_engine_init, trt_engine_temporal, data_loader)

    # 评估与保存
    if args.out:
        print(f"\n💾 Writing results to {args.out}")
        mmcv.dump(outputs, args.out)
    
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
            
        # 关闭无关任务评测
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
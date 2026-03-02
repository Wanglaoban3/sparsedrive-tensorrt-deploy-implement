# Copyright (c) OpenMMLab. All rights reserved.
import tensorrt as trt
import argparse
import mmcv
import os
import sys
from collections import OrderedDict
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mmdet3d
import projects.mmdet3d_plugin
import ctypes

class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
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

def compare_tensors_sorted(name, t_pth, t_trt, conf_pth, conf_trt):
    # 根据置信度，把被打乱的顺序重新拉齐
    _, idx_pth = torch.sort(conf_pth[0], dim=-1, descending=True)
    _, idx_trt = torch.sort(conf_trt[0], dim=-1, descending=True)
    
    t_pth_sorted = t_pth[0][idx_pth]
    t_trt_sorted = t_trt[0][idx_trt]
    
    diff = torch.abs(t_pth_sorted - t_trt_sorted)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  [{name: <15} (Sorted)] Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
    return max_diff

def compare_pipeline(model, engine_init, engine_temporal, data_loader):
    model.eval()
    dataset = data_loader.dataset
    nh_det = 600
    
    det_bank = model.head.det_head.instance_bank
    
    # TRT 历史占位
    history_trt = {
        'prev_det_feat': torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda'),
        'prev_det_anchor': torch.zeros((1, nh_det, 11), dtype=torch.float32, device='cuda'),
        'prev_det_conf': torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
    }
    
    prev_global_mat = None
    prev_time = None

    print("\n🔍 开始逐帧对比 PyTorch Native 与 TRT...")
    for i, data in enumerate(data_loader):
        if i >= 3: # 我们只测前 3 帧
            break
            
        print(f"\n====================== Frame {i} ======================")
        with torch.no_grad():
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            img = scattered_data['img']
            proj_mat = scattered_data['projection_mat']
            kwargs = scattered_data.copy()
            kwargs.pop('img')
            img_metas = kwargs['img_metas'][0]

            curr_time = img_metas['timestamp']
            is_first_frame = (prev_time is None)
            if is_first_frame:
                prev_time = curr_time - 0.5 

            dt = curr_time - prev_time
            if dt > 2.0 or dt < 0: dt = 0.5 
            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            
            curr_global = img_metas['T_global']
            curr_global_inv = img_metas['T_global_inv']
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = curr_global_inv @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)

            # ================= 1. PyTorch 原生推理 =================
            if is_first_frame:
                det_bank.reset()
            else:
                # 严格按照原生 test.py 滚动
                det_bank.metas = {
                    "timestamp": prev_time,
                    "img_metas": [{"T_global": prev_global_mat}]
                }
                
            features = model.extract_feat(img, metas=kwargs)
            py_outs = model.head(features, kwargs)
            
            py_cls = py_outs[0]['classification'][-1]
            py_reg = py_outs[0]['prediction'][-1]
            py_next_feat = det_bank.cached_feature
            py_next_anchor = det_bank.cached_anchor
            # =======================================================

            # ================= 2. TensorRT 推理 =================
            feed_dict = {
                'img': img,
                'projection_mat': proj_mat,
                'instance_t_matrix': instance_t_matrix,
                'time_interval': dt_tensor,
                **history_trt
            }
            if is_first_frame:
                trt_outs = engine_init.infer(feed_dict)
            else:
                trt_outs = engine_temporal.infer(feed_dict)
                
            trt_cls = trt_outs['det_cls']
            trt_reg = trt_outs['det_bbox']
            trt_next_feat = trt_outs['next_det_feat']
            trt_next_anchor = trt_outs['next_det_anchor']

            history_trt['prev_det_feat'] = trt_next_feat
            history_trt['prev_det_anchor'] = trt_next_anchor
            history_trt['prev_det_conf'] = trt_outs['next_det_conf']
            # =======================================================
            conf_pth = py_cls.max(dim=-1).values.sigmoid()
            conf_trt = trt_cls.max(dim=-1).values.sigmoid()

            # ================= 3. 张量差异计算 =================
            compare_tensors_sorted("BBox Preds", py_reg, trt_reg, conf_pth, conf_trt)
            compare_tensors_sorted("Next Anchor", py_next_anchor, trt_next_anchor, conf_pth, conf_trt)
            
            prev_global_mat = curr_global
            prev_time = curr_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    parser.add_argument("--engine_init", required=True)
    parser.add_argument("--engine_temporal", required=True)
    parser.add_argument("--plugin", default="projects/trt_plugin/build/libSparseDrivePlugin.so")
    args = parser.parse_args()

    if os.path.exists(args.plugin):
        ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ Loaded Custom Plugin: {args.plugin}")

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_map'] = False
        cfg.task_config['with_motion_plan'] = False
        if 'head' in cfg.model: cfg.model.head.task_config = cfg.task_config
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()

    engine_init = TRTInfer(args.engine_init)
    engine_temporal = TRTInfer(args.engine_temporal)

    compare_pipeline(model, engine_init, engine_temporal, data_loader)
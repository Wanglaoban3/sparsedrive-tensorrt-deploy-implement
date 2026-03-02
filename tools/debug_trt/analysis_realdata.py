import torch
import mmcv
import numpy as np
import tensorrt as trt
from collections import OrderedDict
import sys
import os
import ctypes

# 1. 环境初始化
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel

# 加载插件
plugin_lib_path = "projects/trt_plugin/build/libSparseDrivePlugin.so" 
if os.path.exists(plugin_lib_path):
    ctypes.CDLL(plugin_lib_path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    print(f"✅ Loaded SparseDrive custom plugin.")

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
            else:
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                
            gpu_mem = torch.empty(tuple(shape), dtype=torch.float32, device='cuda')
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
        return {name: mem.clone() for name, mem in self.outputs.items()}

def run_real_temporal_comparison():
    cfg_path = "projects/configs/sparsedrive_small_stage2.py"
    ckpt_path = "ckpt/sparsedrive_stage2.pth"
    engine_path = "work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine"

    cfg = Config.fromfile(cfg_path)
    import projects.mmdet3d_plugin 
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    model = build_detector(cfg.model).cuda()
    load_checkpoint(model, ckpt_path, map_location='cuda')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    trt_model = TRTInfer(engine_path)

    nh_det, nh_map = 600, 33
    trt_history = {
        'prev_det_feat': torch.zeros((1, nh_det, 256), device='cuda'),
        'prev_det_anchor': torch.zeros((1, nh_det, 11), device='cuda'),
        'prev_map_feat': torch.zeros((1, nh_map, 256), device='cuda'),
        'prev_map_anchor': torch.zeros((1, nh_map, 40), device='cuda'),
    }

    loader_iter = iter(loader)
    prev_global_mat = None

    for frame_idx in range(2):
        data = next(loader_iter)
        img_metas = data['img_metas'].data[0][0]
        img_raw = data['img'].data[0][0].cuda()
        img_tensor = img_raw.unsqueeze(0) if img_raw.dim() == 4 else img_raw
        bs, n, c, h, w = img_tensor.shape
        proj_mat = torch.stack([p.cuda() for p in data['projection_mat'].data[0]], dim=0).unsqueeze(0)
        
        # 位姿补偿
        curr_global = img_metas['T_global']
        curr_global_inv = img_metas['T_global_inv']
        if prev_global_mat is None:
            instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
        else:
            t_mat = curr_global_inv @ prev_global_mat
            instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
        prev_global_mat = curr_global

        # --- PyTorch ---
        with torch.no_grad():
            native_metas = {
                'img_metas': [img_metas],
                'projection_mat': proj_mat,
                'image_wh': img_tensor.new_tensor([w, h]).view(1, 1, 2).repeat(bs, n, 1),
                'timestamp': img_tensor.new_tensor([img_metas['timestamp']]), 
            }
            features = model.module.extract_feat(img_tensor, metas=native_metas)
            py_outs = model.module.head(features, native_metas)

        # --- TRT ---
        feed_dict = {'img': img_tensor, 'projection_mat': proj_mat, 'instance_t_matrix': instance_t_matrix, **trt_history}
        trt_outs = trt_model.infer(feed_dict)

        print(f"\n" + "="*20 + f" FRAME {frame_idx} AUDIT " + "="*20)

        # 1. 检测头审计 (DET)
        p_det_cls = py_outs[0]['classification'][-1]
        p_det_reg = py_outs[0]['prediction'][-1]
        t_det_cls = trt_outs['det_cls']
        t_det_reg = trt_outs['det_bbox']
        
        p_det_scores = p_det_cls[0].sigmoid().max(-1).values
        p_top_v, p_top_i = torch.topk(p_det_scores, 10)
        
        print(f"\n[DET HEAD] Top 10 BBox Matching (XYZ, WLH, Yaw)")
        print(f"{'Rank':<5}|{'Score(P/T)':<15}|{'Dist':<8}|{'Size_Err':<10}|{'Yaw_Err':<8}")
        print("-"*55)
        
        for i in range(len(p_top_i)):
            p_idx = p_top_i[i]
            p_box = p_det_reg[0, p_idx]
            # 空间最近邻匹配
            dists = torch.norm(t_det_reg[0, :, :3] - p_box[:3], dim=-1)
            min_dist, m_idx = torch.min(dists, dim=0)
            t_box = t_det_reg[0, m_idx]
            
            # 尺寸误差 (WLH)
            size_err = torch.norm(t_box[3:6] - p_box[3:6]).item()
            # 航向角误差 (Yaw: 使用 sin/cos 还原或直接算差异)
            yaw_err = torch.abs(t_box[6:8] - p_box[6:8]).mean().item()
            
            print(f"#{i+1:<4}|{p_top_v[i]:.3f}/{t_det_cls[0, m_idx].sigmoid().max():.3f}|{min_dist:.3f}m |{size_err:.3f}m |{yaw_err:.4f}")

        # 2. 地图头审计 (MAP)
        # 兼容节点名: map_bbox, bbox_map, map_preds
        
        map_key = 'map_pts'
        if map_key:
            p_map_cls = py_outs[1]['classification'][-1]
            p_map_reg = py_outs[1]['prediction'][-1]
            t_map_cls = trt_outs[next(k for k in trt_outs.keys() if 'map' in k and 'cls' in k)]
            t_map_reg = trt_outs[map_key]
            
            p_map_scores = p_map_cls[0].sigmoid().max(-1).values
            p_map_v, p_map_i = torch.topk(p_map_scores, 5)
            
            print(f"\n[MAP HEAD] Top 5 Polyline Matching")
            print(f"{'Rank':<5}|{'Score(P/T)':<15}|{'Avg_Pt_Dist':<12}")
            print("-"*40)
            
            for i in range(len(p_map_i)):
                p_idx = p_map_i[i]
                p_line = p_map_reg[0, p_idx].view(20, 2)
                t_lines = t_map_reg[0].view(-1, 20, 2)
                
                line_dists = torch.norm(t_lines - p_line, dim=-1).mean(dim=-1)
                min_line_dist, m_l_idx = torch.min(line_dists, dim=0)
                
                print(f"#{i+1:<4}|{p_map_v[i]:.3f}/{t_map_cls[0, m_l_idx].sigmoid().max():.3f}|{min_line_dist:.4f}m")
        else:
            print(f"\n⚠️ Map BBox key not found. Available keys: {list(trt_outs.keys())}")

        # 更新历史
        trt_history['prev_det_feat'] = trt_outs['next_det_feat']
        trt_history['prev_det_anchor'] = trt_outs['next_det_anchor']
        trt_history['prev_map_feat'] = trt_outs['next_map_feat']
        trt_history['prev_map_anchor'] = trt_outs['next_map_anchor']

if __name__ == "__main__":
    run_real_temporal_comparison()
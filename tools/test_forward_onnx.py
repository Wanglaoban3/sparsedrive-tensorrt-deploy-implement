import torch
import torch.nn as nn
import numpy as np
import os
import sys
import ctypes
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector

# 1. 环境初始化
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 加载自定义插件 (解决 DFG 算子识别问题)
plugin_lib_path = "projects/trt_plugin/build/libSparseDrivePlugin.so" 
if os.path.exists(plugin_lib_path):
    ctypes.CDLL(plugin_lib_path)

# ==============================================================================
# 🏗️ Step 1: 沿用你的双头 ONNX Wrapper
# ==============================================================================
class SparseDriveONNXPathWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.det_head = model.head.det_head
        self.map_head = model.head.map_head

    def forward(self, img, projection_mat, 
                prev_det_feat, prev_det_anchor, 
                prev_map_feat, prev_map_anchor,
                instance_t_matrix):
        B, N, C, H, W = img.shape
        # Backbone + Neck
        img_reshaped = img.reshape(B * N, C, H, W)
        x = self.model.img_backbone(img_reshaped)
        if self.model.img_neck is not None: 
            x = self.model.img_neck(x)
        
        # 共享特征图处理 (这是你 Wrapper 里的核心对齐逻辑)
        from projects.mmdet3d_plugin.ops import feature_maps_format
        feature_maps = [f.reshape(B, N, f.shape[1], f.shape[2], f.shape[3]) for f in x]
        formatted_feature_maps = feature_maps_format(feature_maps)
        
        metas = {
            'img_metas': [{'lidar2img': projection_mat[i], 'img_shape': [(H, W)] * N} for i in range(B)],
            'projection_mat': projection_mat, 
            'image_wh': img.new_tensor([W, H]).view(1, 1, 2).repeat(B, N, 1)
        }

        # 运行 forward_onnx
        det_outs = self.det_head.forward_onnx(
            formatted_feature_maps, prev_det_feat, prev_det_anchor, instance_t_matrix, metas)
        map_outs = self.map_head.forward_onnx(
            formatted_feature_maps, prev_map_feat, prev_map_anchor, instance_t_matrix, metas)

        return det_outs, map_outs

# ==============================================================================
# 🏁 Step 2: 运行真实数据比对 (Frame 0 -> Frame 1)
# ==============================================================================
def run_real_data_audit():
    cfg_path = "projects/configs/sparsedrive_small_stage2.py"
    ckpt_path = "ckpt/sparsedrive_stage2.pth"

    print("📦 Loading Model...")
    cfg = Config.fromfile(cfg_path)
    # 注入插件
    import projects.mmdet3d_plugin 
    model = build_detector(cfg.model).cuda()
    load_checkpoint(model, ckpt_path, map_location='cuda')
    
    # 包装模型
    wrapper = SparseDriveONNXPathWrapper(model).eval()

    # 准备真实数据
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    # 初始化历史 (按照你的 config 设置维度)
    history = {
        'prev_det_feat': torch.zeros((1, 600, 256), device='cuda'),
        'prev_det_anchor': torch.zeros((1, 600, 11), device='cuda'),
        'prev_map_feat': torch.zeros((1, 33, 256), device='cuda'),
        'prev_map_anchor': torch.zeros((1, 33, 40), device='cuda'),
    }

    loader_iter = iter(loader)
    prev_global_mat = None

    for frame_idx in range(2):
        print(f"\n" + "="*20 + f" FRAME {frame_idx} AUDIT " + "="*20)
        data = next(loader_iter)
        img_metas = data['img_metas'].data[0][0]
        img_tensor = data['img'].data[0][0].cuda().unsqueeze(0)
        proj_mat = torch.stack([p.cuda() for p in data['projection_mat'].data[0]], dim=0).unsqueeze(0)
        
        # 计算位姿矩阵
        curr_global = img_metas['T_global']
        curr_global_inv = img_metas['T_global_inv']
        if prev_global_mat is None:
            instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
        else:
            t_mat = curr_global_inv @ prev_global_mat
            instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
        prev_global_mat = curr_global

        with torch.no_grad():
            # 1. 运行原生路径 (Native)
            native_metas = {
                'img_metas': [img_metas],
                'projection_mat': proj_mat,
                'image_wh': img_tensor.new_tensor([img_tensor.shape[4], img_tensor.shape[3]]).view(1, 1, 2).repeat(1, 6, 1),
                'timestamp': img_tensor.new_tensor([img_metas['timestamp']]), 
            }
            # 原生 extract_feat + head
            raw_feats = model.extract_feat(img_tensor, metas=native_metas)
            py_outs = model.head(raw_feats, native_metas)

            # 2. 运行 ONNX 路径 (你的 Wrapper 逻辑)
            onnx_det, onnx_map = wrapper(
                img_tensor, proj_mat, 
                history['prev_det_feat'], history_det_anchor := history['prev_det_anchor'],
                history['prev_map_feat'], history['prev_map_anchor'],
                instance_t_matrix
            )

        # --- 精度对账 ---
        # 提取 Native 结果 (最后一层检测结果)
        p_det_cls = py_outs[0]['classification'][-1]
        o_det_cls = onnx_det['cls_scores'][:, :900] # 只比对前 900 个

        cos_sim = torch.nn.functional.cosine_similarity(p_det_cls.flatten(), o_det_cls.flatten(), dim=0)
        print(f"[Det_CLS] Cos_Sim: {cos_sim.item():.8f}")
        p_reg = py_outs[0]['prediction'][-1][0, :, :3].mean(0)
        o_reg = onnx_det['bbox_preds'][0, :900, :3].mean(0) # 只看前900个
        print(f"   Frame {frame_idx} Native XYZ Mean: {p_reg.cpu().numpy()}")
        print(f"   Frame {frame_idx} ONNX   XYZ Mean: {o_reg.cpu().numpy()}")
        
        if frame_idx == 1:
            # 重点看第二帧的回归坐标均值，判断位姿补偿是否起效
            p_reg = py_outs[0]['prediction'][-1][0, :, :3].mean(0)
            o_reg = onnx_det['bbox_preds'][0, :900, :3].mean(0)
            print(f"   Native XYZ Mean: {p_reg.cpu().numpy()}")
            print(f"   ONNX   XYZ Mean: {o_reg.cpu().numpy()}")

        # 更新历史 (时序闭环)
        history['prev_det_feat'] = onnx_det['next_instance_feature']
        history['prev_det_anchor'] = onnx_det['next_anchor']
        history['prev_map_feat'] = onnx_map['next_instance_feature']
        history['prev_map_anchor'] = onnx_map['next_anchor']

if __name__ == "__main__":
    run_real_data_audit()
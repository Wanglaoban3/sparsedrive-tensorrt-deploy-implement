import argparse
import os
import sys
import warnings
import torch
import torch.nn as nn
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

# 1. 确保项目根目录在 path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# 💡 Monkey Patch: 确保 DAF 算子能被 ONNX 导出
# ==============================================================================
from projects.mmdet3d_plugin.ops import feature_maps_format

# (这里保留你环境里可能需要的任何自定义 Patch 代码，如果之前有的话)

class SparseDriveONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 显式提取两个任务头
        self.det_head = model.head.det_head
        self.map_head = model.head.map_head

    def forward(self, img, projection_mat, 
                prev_det_feat, prev_det_anchor, prev_det_conf, # [New] Det History
                prev_map_feat, prev_map_anchor, prev_map_conf, # [New] Map History
                instance_t_matrix, time_interval):             # [New] Ego Motion & DT
        
        # 确保所有输入都在同一设备
        dev = img.device
        projection_mat = projection_mat.to(dev)
        prev_det_feat = prev_det_feat.to(dev)
        prev_det_anchor = prev_det_anchor.to(dev)
        prev_det_conf = prev_det_conf.to(dev)
        prev_map_feat = prev_map_feat.to(dev)
        prev_map_anchor = prev_map_anchor.to(dev)
        prev_map_conf = prev_map_conf.to(dev)
        instance_t_matrix = instance_t_matrix.to(dev)
        time_interval = time_interval.to(dev)

        # 1. 特征提取 (共享 Backbone 和 Neck)
        B, N, C, H, W = img.shape
        img_reshaped = img.reshape(B * N, C, H, W)
        x = self.model.img_backbone(img_reshaped)
        if self.model.img_neck is not None:
            x = self.model.img_neck(x)
            
        feature_maps = []
        for feat in x:
            _, C_feat, H_feat, W_feat = feat.shape
            feature_maps.append(feat.reshape(B, N, C_feat, H_feat, W_feat))
            
        # 2. 格式化特征图 (触发 DAF 插件优化)
        formatted_feature_maps = feature_maps_format(feature_maps)
        
        # 构造 Meta 信息
        img_metas = [{'lidar2img': projection_mat[i], 'img_shape': [(H, W)] * N} for i in range(B)]
        image_wh = img.new_tensor([W, H]).view(1, 1, 2).repeat(B, N, 1)

        metas = {
            'img_metas': img_metas,
            'projection_mat': projection_mat, 
            'timestamp': 0, 
            'image_wh': image_wh 
        }

        # 3. 推理检测头 (Det Head)
        # [Update] 传入 time_interval 和 prev_confidence 以启用 Z-Lock 和 Decay
        det_outs = self.det_head.forward_onnx(
            feature_maps=formatted_feature_maps,
            prev_instance_feature=prev_det_feat,
            prev_anchor=prev_det_anchor,
            instance_t_matrix=instance_t_matrix,
            time_interval=time_interval,
            prev_confidence=prev_det_conf,
            metas=metas 
        )

        # 4. 推理地图头 (Map Head)
        # 虽然 Map 可能不使用 time_interval 进行位移补偿，但 Decay 逻辑可能需要 prev_confidence
        # map_outs = self.map_head.forward_onnx(
        #     feature_maps=formatted_feature_maps,
        #     prev_instance_feature=prev_map_feat,
        #     prev_anchor=prev_map_anchor,
        #     instance_t_matrix=instance_t_matrix,
        #     time_interval=time_interval,
        #     prev_confidence=prev_map_conf,
        #     metas=metas 
        # )

        # 5. 合并导出所有的输出节点 (新增了 next_confidence)
        return (
            det_outs["cls_scores"], det_outs["bbox_preds"], 
            det_outs["next_instance_feature"], det_outs["next_anchor"], det_outs["next_confidence"],
            # map_outs["cls_scores"], map_outs["bbox_preds"],
            # map_outs["next_instance_feature"], map_outs["next_anchor"], map_outs["next_confidence"]
        )

def main():
    parser = argparse.ArgumentParser(description='Export SparseDrive Multi-Head to ONNX')
    parser.add_argument('--config', default="projects/configs/sparsedrive_small_stage2.py")
    parser.add_argument('--checkpoint', default='ckpt/sparsedrive_stage2.pth')
    parser.add_argument('--out', default='work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    
    # 强制开启检测和地图任务
    if hasattr(cfg, 'task_config'):
        print("✅ Enabling Detection and Map tasks for export...")
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = False
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    # 初始化模型
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().eval()

    wrapper = SparseDriveONNXWrapper(model)
    
    # 准备 Dummy Inputs
    batch_size = 1
    num_cams = 6
    H, W = 256, 704 
    embed_dims = 256
    device = 'cuda'

    # 获取各自任务的 History 数量
    num_det_history = model.head.det_head.instance_bank.num_temp_instances # 600
    num_map_history = model.head.map_head.instance_bank.num_temp_instances # 33
    
    dummy_img = torch.randn(batch_size, num_cams, 3, H, W, device=device)
    dummy_proj_mat = torch.randn(batch_size, num_cams, 4, 4, device=device)
    dummy_ego_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # [New] 时间间隔输入 (通常为 0.5s)
    dummy_time_interval = torch.tensor([0.5], dtype=torch.float32, device=device).repeat(batch_size)

    # 检测头输入
    dummy_det_feat = torch.randn(batch_size, num_det_history, embed_dims, device=device)
    dummy_det_anchor = torch.randn(batch_size, num_det_history, 11, device=device)
    dummy_det_conf = torch.randn(batch_size, num_det_history, device=device)

    # 地图头输入 (Map Anchor 是 40 维)
    dummy_map_feat = torch.randn(batch_size, num_map_history, embed_dims, device=device)
    dummy_map_anchor = torch.randn(batch_size, num_map_history, 40, device=device)
    # [New] 地图头上一帧置信度
    dummy_map_conf = torch.rand(batch_size, num_map_history, device=device)

    input_names = [
        'img', 'projection_mat', 
        'prev_det_feat', 'prev_det_anchor', 'prev_det_conf',
        'prev_map_feat', 'prev_map_anchor', 'prev_map_conf',
        'instance_t_matrix', 'time_interval'
    ]
    
    output_names = [
        'det_cls', 'det_bbox', 'next_det_feat', 'next_det_anchor', 'next_det_conf',
        # 'map_cls', 'map_pts',  'next_map_feat', 'next_map_anchor', 'next_map_conf'
    ]

    print(f"🚀 Exporting Multi-Head model to {args.out}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_proj_mat, 
             dummy_det_feat, dummy_det_anchor, dummy_det_conf,
             dummy_map_feat, dummy_map_anchor, dummy_map_conf,
             dummy_ego_mat, dummy_time_interval),
            args.out,
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
            do_constant_folding=False,
        )
    
    print("🎉 Export finished successfully! Supported Inputs: Time Interval & History Confidence.")

if __name__ == '__main__':
    main()
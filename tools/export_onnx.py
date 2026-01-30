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
from projects.mmdet3d_plugin.ops import feature_maps_format
# (此处省略你之前的 clean_feature_maps_format 定义及 Monkey Patch 代码，请保留原样)

class SparseDriveONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 显式提取两个任务头
        self.det_head = model.head.det_head
        self.map_head = model.head.map_head

    def forward(self, img, projection_mat, 
                prev_det_feat, prev_det_anchor, 
                prev_map_feat, prev_map_anchor,
                instance_t_matrix):
        dev = img.device
        projection_mat = projection_mat.to(dev)
        prev_det_feat = prev_det_feat.to(dev)
        prev_det_anchor = prev_det_anchor.to(dev)
        prev_map_feat = prev_map_feat.to(dev)
        prev_map_anchor = prev_map_anchor.to(dev)
        instance_t_matrix = instance_t_matrix.to(dev)
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
        det_outs = self.det_head.forward_onnx(
            feature_maps=formatted_feature_maps,
            prev_instance_feature=prev_det_feat,
            prev_anchor=prev_det_anchor,
            instance_t_matrix=instance_t_matrix,
            metas=metas 
        )

        # 4. 推理地图头 (Map Head)
        map_outs = self.map_head.forward_onnx(
            feature_maps=formatted_feature_maps,
            prev_instance_feature=prev_map_feat,
            prev_anchor=prev_map_anchor,
            instance_t_matrix=instance_t_matrix,
            metas=metas 
        )

        # 5. 合并导出所有的输出节点
        return (
            det_outs["cls_scores"], det_outs["bbox_preds"], 
            det_outs["next_instance_feature"], det_outs["next_anchor"],
            map_outs["cls_scores"], map_outs["bbox_preds"],
            map_outs["next_instance_feature"], map_outs["next_anchor"]
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

    # 检测头输入
    dummy_det_feat = torch.randn(batch_size, num_det_history, embed_dims, device=device)
    dummy_det_anchor = torch.randn(batch_size, num_det_history, 11, device=device)

    # 地图头输入 (Map Anchor 是 40 维)
    dummy_map_feat = torch.randn(batch_size, num_map_history, embed_dims, device=device)
    dummy_map_anchor = torch.randn(batch_size, num_map_history, 40, device=device)

    input_names = [
        'img', 'projection_mat', 
        'prev_det_feat', 'prev_det_anchor', 
        'prev_map_feat', 'prev_map_anchor', 
        'instance_t_matrix'
    ]
    output_names = [
        'det_cls', 'det_bbox', 'next_det_feat', 'next_det_anchor',
        'map_cls', 'map_pts', 'next_map_feat', 'next_map_anchor'
    ]

    print(f"🚀 Exporting Multi-Head model to {args.out}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_proj_mat, 
             dummy_det_feat, dummy_det_anchor, 
             dummy_map_feat, dummy_map_anchor, 
             dummy_ego_mat),
            args.out,
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
            do_constant_folding=False,
            # 指定动态维度 (可选，但在 Orin 部署时建议设为固定以获取最高性能)
            # dynamic_axes={'img': {0: 'batch'}} 
        )
    
    print("🎉 Export finished successfully! Map and Det heads are unified.")

if __name__ == '__main__':
    main()
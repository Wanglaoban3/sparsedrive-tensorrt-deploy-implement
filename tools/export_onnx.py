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

class SparseDriveONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 显式提取两个任务头
        self.det_head = model.head.det_head
        self.map_head = model.head.map_head

    def forward(self, img, projection_mat, 
                prev_det_feat, prev_det_anchor, prev_det_conf, # Det History
                prev_map_feat, prev_map_anchor, prev_map_conf, # Map History
                instance_t_matrix, time_interval):             # Ego Motion & DT
        
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
        det_outs = self.det_head.forward_onnx(
            feature_maps=formatted_feature_maps,
            prev_instance_feature=prev_det_feat,
            prev_anchor=prev_det_anchor,
            instance_t_matrix=instance_t_matrix,
            time_interval=time_interval,
            prev_confidence=prev_det_conf,
            metas=metas 
        )

        # 4. 推理地图头 (Map Head) 【解除注释，已修复】
        map_outs = self.map_head.forward_onnx(
            feature_maps=formatted_feature_maps,
            prev_instance_feature=prev_map_feat,
            prev_anchor=prev_map_anchor,
            instance_t_matrix=instance_t_matrix,
            time_interval=time_interval,
            prev_confidence=prev_map_conf,
            metas=metas 
        )

        # 5. 合并导出所有的输出节点 (Det + Map)
        return (
            det_outs["cls_scores"], det_outs["bbox_preds"], 
            det_outs["next_instance_feature"], det_outs["next_anchor"], det_outs["next_confidence"],
            map_outs["cls_scores"], map_outs["bbox_preds"],
            map_outs["next_instance_feature"], map_outs["next_anchor"], map_outs["next_confidence"]
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
    
    # 准备基础 Dummy Inputs
    batch_size = 1
    num_cams = 6
    H, W = 256, 704 
    embed_dims = 256
    device = 'cuda'

    num_det_history = model.head.det_head.instance_bank.num_temp_instances # 600
    num_map_history = model.head.map_head.instance_bank.num_temp_instances # 33
    
    dummy_img = torch.randn(batch_size, num_cams, 3, H, W, device=device)
    dummy_proj_mat = torch.randn(batch_size, num_cams, 4, 4, device=device)
    dummy_ego_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    dummy_time_interval = torch.tensor([0.5], dtype=torch.float32, device=device).repeat(batch_size)

    input_names = [
        'img', 'projection_mat', 
        'prev_det_feat', 'prev_det_anchor', 'prev_det_conf',
        'prev_map_feat', 'prev_map_anchor', 'prev_map_conf',
        'instance_t_matrix', 'time_interval'
    ]
    
    output_names = [
        'det_cls', 'det_bbox', 'next_det_feat', 'next_det_anchor', 'next_det_conf',
        'map_cls', 'map_pts',  'next_map_feat', 'next_map_anchor', 'next_map_conf'
    ]

    # =========================================================================
    # 1️⃣ 导出第一帧模型 (Init Engine) - 历史输入全部为 Zeros
    # =========================================================================
    out_first = args.out.replace('.onnx', '_first.onnx')
    print(f"\n🚀 [1/2] Exporting FIRST-FRAME Multi-Head model to {out_first}...")
    
    # Zeros 填充，触发 is_first_frame 分支剪枝
    det_feat_zeros = torch.zeros(batch_size, num_det_history, embed_dims, device=device)
    det_anchor_zeros = torch.zeros(batch_size, num_det_history, 11, device=device)
    det_conf_zeros = torch.zeros(batch_size, num_det_history, device=device)

    map_feat_zeros = torch.zeros(batch_size, num_map_history, embed_dims, device=device)
    map_anchor_zeros = torch.zeros(batch_size, num_map_history, 40, device=device)
    map_conf_zeros = torch.zeros(batch_size, num_map_history, device=device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_proj_mat, 
             det_feat_zeros, det_anchor_zeros, det_conf_zeros,
             map_feat_zeros, map_anchor_zeros, map_conf_zeros,
             dummy_ego_mat, dummy_time_interval),
            out_first,
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
            do_constant_folding=False,
        )
    print("🎉 First-Frame Export finished!")

    # =========================================================================
    # 2️⃣ 导出时序推理模型 (Temporal Engine) - 历史输入为 Randn
    # =========================================================================
    print(f"\n🚀 [2/2] Exporting TEMPORAL Multi-Head model to {args.out}...")
    
    # Randn 填充，保留完整的历史运动补偿和注意力特征融合图
    det_feat_rand = torch.randn(batch_size, num_det_history, embed_dims, device=device)
    det_anchor_rand = torch.randn(batch_size, num_det_history, 11, device=device)
    det_conf_rand = torch.rand(batch_size, num_det_history, device=device)

    map_feat_rand = torch.randn(batch_size, num_map_history, embed_dims, device=device)
    map_anchor_rand = torch.randn(batch_size, num_map_history, 40, device=device)
    map_conf_rand = torch.rand(batch_size, num_map_history, device=device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_proj_mat, 
             det_feat_rand, det_anchor_rand, det_conf_rand,
             map_feat_rand, map_anchor_rand, map_conf_rand,
             dummy_ego_mat, dummy_time_interval),
            args.out,
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
            do_constant_folding=False,
        )
    print("🎉 Temporal Export finished!")
    print("\n✅ All Multi-Head ONNX models have been successfully exported!")

if __name__ == '__main__':
    main()
# tools/export_onnx.py

import argparse
import os
import os.path as osp
import sys
import warnings

# ==============================================================================
# [Hotfix] Nuke ALL MMCV Wrappers for ONNX Export
# 必须放在所有 import 之前！这是消除 If 节点的关键！
# ==============================================================================
import torch.nn as nn

# 1. 预加载模块防止覆盖失效
try:
    import mmcv.cnn.bricks.wrappers
except ImportError:
    pass

# 2. 定义通用的 Clean Forward (直接透传给 PyTorch 原生层，绕过 MMCV 的空检查)
def clean_forward_linear(self, x):
    return super(mmcv.cnn.bricks.wrappers.Linear, self).forward(x)

def clean_forward_conv2d(self, x):
    return super(mmcv.cnn.bricks.wrappers.Conv2d, self).forward(x)

def clean_forward_conv_transpose2d(self, x, output_size=None):
    return super(mmcv.cnn.bricks.wrappers.ConvTranspose2d, self).forward(x, output_size)

def clean_forward_max_pool2d(self, x):
    return super(mmcv.cnn.bricks.wrappers.MaxPool2d, self).forward(x)

# 3. 暴力替换所有 Wrapper 的 forward 方法
print("☢️☢️☢️ HOTPATCH: Removing 'If' nodes from ALL MMCV wrappers... ☢️☢️☢️")
if hasattr(mmcv.cnn.bricks.wrappers, 'Linear'):
    mmcv.cnn.bricks.wrappers.Linear.forward = clean_forward_linear
    print(" - Patched mmcv.cnn.Linear")

if hasattr(mmcv.cnn.bricks.wrappers, 'Conv2d'):
    mmcv.cnn.bricks.wrappers.Conv2d.forward = clean_forward_conv2d
    print(" - Patched mmcv.cnn.Conv2d")

if hasattr(mmcv.cnn.bricks.wrappers, 'ConvTranspose2d'):
    mmcv.cnn.bricks.wrappers.ConvTranspose2d.forward = clean_forward_conv_transpose2d
    print(" - Patched mmcv.cnn.ConvTranspose2d")

if hasattr(mmcv.cnn.bricks.wrappers, 'MaxPool2d'):
    mmcv.cnn.bricks.wrappers.MaxPool2d.forward = clean_forward_max_pool2d
    print(" - Patched mmcv.cnn.MaxPool2d")
# ==============================================================================

# 将项目根目录添加到路径，确保加载本地修改过的 blocks.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
# 尝试导入 feature_maps_format，如果 ops 里没有定义，提供一个默认实现
try:
    from projects.mmdet3d_plugin.ops import feature_maps_format
except ImportError:
    def feature_maps_format(x): return x


def parse_args():
    parser = argparse.ArgumentParser(description='Export SparseDrive model to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', type=str, default='sparsedrive.onnx', help='output onnx file name')
    parser.add_argument('--fp16', action='store_true', help='export in fp16 mode')
    return parser.parse_args()


class SparseDriveONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 提取 Head 部分 (兼容 det_head 结构)
        self.head = model.head.det_head if hasattr(model.head, 'det_head') else model.head
        
        self.use_deformable_func = getattr(model, 'use_deformable_func', False)

    def forward(self, img, projection_mat, prev_instance_feature, prev_anchor, instance_t_matrix):
        """
        Args:
            img: [B, N_cam, 3, H, W]
            projection_mat: [B, N_cam, 4, 4] (Lidar2Img 矩阵)
            prev_instance_feature: [B, N_history, C]
            prev_anchor: [B, N_history, 11]
            instance_t_matrix: [B, 4, 4] (Ego 变换)
        """
        B, N, C, H, W = img.shape
        
        # 1. Backbone & Neck
        img_reshaped = img.reshape(B * N, C, H, W)
        x = self.model.img_backbone(img_reshaped)
        if self.model.img_neck is not None:
            x = self.model.img_neck(x)
            
        # 2. 恢复特征图维度 [B, N_cam, C, H_feat, W_feat]
        feature_maps = []
        for feat in x:
            _, C_feat, H_feat, W_feat = feat.shape
            feature_maps.append(feat.reshape(B, N, C_feat, H_feat, W_feat))
            
        # 3. 格式化 Feature Maps (适配 DAF 算子)
        formatted_feature_maps = feature_maps_format(feature_maps)
        
        # 4. 构造 Metas
        img_metas = []
        for i in range(B):
            meta = {
                'lidar2img': projection_mat[i], 
                'img_shape': [(H, W)] * N,
            }
            img_metas.append(meta)
        
        # [关键修复] image_wh 需要是 [B, 2] 形状，而不是 [2]
        # 这样 blocks.py 里的 image_wh[:, :, None, None] 才能正确工作
        image_wh = torch.tensor([W, H], device=img.device).unsqueeze(0).repeat(B, 1)

        metas = {
            'img_metas': img_metas,
            'projection_mat': projection_mat, 
            'timestamp': 0, 
            'image_wh': image_wh 
        }

        # 5. Head Inference
        outs = self.head.forward_onnx(
            feature_maps=formatted_feature_maps,
            prev_instance_feature=prev_instance_feature,
            prev_anchor=prev_anchor,
            instance_t_matrix=instance_t_matrix,
            metas=metas 
        )
        
        return outs

def main():
    # 1. 确保当前目录在 sys.path 最前面
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    args = parse_args()

    # 2. 加载配置
    cfg = Config.fromfile(args.config)
    if args.fp16:
        print("Enable FP16 mode...")
    
    # 导入自定义模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # 3. 构建模型
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()
    
    if args.fp16:
        model.half()

    # 4. 包装模型
    wrapper = SparseDriveONNXWrapper(model)
    if args.fp16:
        wrapper.half()

    # 5. 准备 Dummy Input
    print("Preparing dummy inputs...")
    
    batch_size = 1
    num_cams = 6
    # [请确认] 这里是默认尺寸，如果你的 config 里不是 256x704，请修改这里
    H, W = 256, 704 
    embed_dims = 256
    
    # 获取历史帧数量
    head_module = model.head.det_head if hasattr(model.head, 'det_head') else model.head
    if hasattr(head_module, 'instance_bank'):
        num_history = head_module.instance_bank.num_temp_instances
    else:
        num_history = 600
        print(f"Warning: Could not auto-detect num_history, using default: {num_history}")

    print(f"History Instance Count: {num_history}")
    
    dtype = torch.float16 if args.fp16 else torch.float32
    device = 'cuda'

    # 构造输入 Tensors
    dummy_img = torch.randn(batch_size, num_cams, 3, H, W, device=device, dtype=dtype)
    dummy_proj_mat = torch.randn(batch_size, num_cams, 4, 4, device=device, dtype=dtype)
    dummy_prev_feat = torch.randn(batch_size, num_history, embed_dims, device=device, dtype=dtype)
    dummy_prev_anchor = torch.randn(batch_size, num_history, 11, device=device, dtype=dtype)
    dummy_ego_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    input_names = [
        'img', 
        'projection_mat', 
        'prev_instance_feature', 
        'prev_anchor', 
        'instance_t_matrix'
    ]
    
    output_names = [
        'cls_scores', 
        'bbox_preds', 
        'next_instance_feature', 
        'next_anchor',
        'quality_scores'
    ]

    # 6. 导出 ONNX
    print(f"Exporting to {args.out}...")
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_proj_mat, dummy_prev_feat, dummy_prev_anchor, dummy_ego_mat),
            args.out,
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
            do_constant_folding=True,
            verbose=False,
        )
    
    print("Export finished successfully!")
    print("Input shapes:")
    print(f"  img: {dummy_img.shape}")
    print(f"  projection_mat: {dummy_proj_mat.shape}")
    print(f"  prev_feature: {dummy_prev_feat.shape}")
    print(f"  prev_anchor: {dummy_prev_anchor.shape}")

if __name__ == '__main__':
    main()
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from ..blocks import linear_relu_ln


@POSITIONAL_ENCODING.register_module()
class SparsePoint3DEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
    ):
        super(SparsePoint3DEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.input_dims = num_sample * coords_dim
        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(self.input_dims)

    def forward(self, anchor: torch.Tensor):
        pos_feat = self.pos_fc(anchor)  
        return pos_feat


@PLUGIN_LAYERS.register_module()
class SparsePoint3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed)
        output = output + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)  ## NOTE anchor embed?
        else:
            cls = None
        qt = None
        return output, cls, qt


@PLUGIN_LAYERS.register_module()
class SparsePoint3DKeyPointsGenerator(BaseModule): 
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        
        if self.num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.ground_height = ground_height
        
        # 🎯 [核心修复与优化 1]：在初始化阶段直接算好所有的 Z 坐标！
        # 注册为 buffer，它会永远跟模型一起待在正确的 CUDA 设备上，ONNX 导出再也不会设备冲突。
        z_vals = torch.tensor(fix_height, dtype=torch.float32) + ground_height
        self.register_buffer('z_vals', z_vals)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape
        
        # 1. 提取基础的 X, Y 坐标 [B, Q, num_sample, 2]
        base_xy = anchor.view(bs, num_anchor, self.num_sample, 2)
        
        # 2. 计算学习到的 X, Y 偏移量
        offset = self.learnable_fc(instance_feature).view(
            bs, num_anchor, self.num_sample, len(self.z_vals), self.num_learnable_pts, 2
        )        
        
        # 3. 叠加偏移量得到最终的 X, Y [B, Q, num_sample, len(fix_height), num_learnable_pts, 2]
        xy = base_xy[:, :, :, None, None, :] + offset
        
        # 🎯 [核心优化 2]：利用广播机制，用 0 开销补齐 Z 轴！
        # 将我们提前算好的 1D Z坐标扩充维度，然后像拼图一样贴上去
        z = self.z_vals.view(1, 1, 1, -1, 1, 1).expand_as(xy[..., :1])
        
        # 只需要一次极其干净的 cat！
        key_points = torch.cat([xy, z], dim=-1) # 得到完美的 [..., 3]
        
        # 一次性展平，替代原本的 flatten(2, 4)
        key_points = key_points.view(bs, num_anchor, self.num_pts, 3)

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        for i, t_time in enumerate(temp_timestamps):
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype) # [B, 4, 4]
            
            # 🎯 [核心优化 3]：消灭猫和老鼠式的 torch.cat(ones_like)
            # 我们不需要再把 3D 点拼成齐次坐标 [x,y,z,1] 去做 4x4 矩阵乘法了
            # 把它拆成纯代数的 R * P + T 形式，TensorRT 最爱吃这种计算！
            R = T_cur2temp[:, :3, :3].unsqueeze(1).unsqueeze(1) # 旋转矩阵 [B, 1, 1, 3, 3]
            t = T_cur2temp[:, :3, 3].unsqueeze(1).unsqueeze(1)  # 平移向量 [B, 1, 1, 3]
            
            # 矩阵广播相乘: key_points @ R^T + t
            temp_key_points = torch.matmul(key_points, R.transpose(-1, -2)) + t
            temp_key_points_list.append(temp_key_points)
            
        return key_points, temp_key_points_list

    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            bs, num_anchor, _ = anchor.shape
            dst_anchor = anchor.view(bs, num_anchor, self.num_sample, 2)
            
            T_src2dst = T_src2dst_list[i].to(dtype=anchor.dtype) # [B, 4, 4]
            
            # 取出 2D 平面的旋转和平移
            R_2d = T_src2dst[:, :2, :2].unsqueeze(1).unsqueeze(1) # [B, 1, 1, 2, 2]
            t_2d = T_src2dst[:, :2, 3].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2]
            
            # 同样使用最纯净的代数计算
            dst_anchor = torch.matmul(dst_anchor, R_2d.transpose(-1, -2)) + t_2d
            
            dst_anchor = dst_anchor.view(bs, num_anchor, self.num_sample * 2)
            dst_anchors.append(dst_anchor)
            
        return dst_anchors
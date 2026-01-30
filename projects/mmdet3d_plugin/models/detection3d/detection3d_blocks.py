import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
from mmcv.cnn import Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS, POSITIONAL_ENCODING
from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln

__all__ = ["SparseBox3DRefinementModule", "SparseBox3DKeyPointsGenerator", "SparseBox3DEncoder"]

@POSITIONAL_ENCODING.register_module(force=True)
class SparseBox3DEncoder(BaseModule):
    def __init__(self, embed_dims, vel_dims=3, mode="add", output_fc=True, in_loops=1, out_loops=2):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims, self.vel_dims, self.mode = embed_dims, vel_dims, mode
        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(*linear_relu_ln(output_dims, in_loops, out_loops, input_dims))
        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3]) if vel_dims > 0 else None
        self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1]) if output_fc else None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        output = pos_feat + size_feat + yaw_feat if self.mode == "add" else torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            output = (output + vel_feat) if self.mode == "add" else torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output

@PLUGIN_LAYERS.register_module(force=True)
class SparseBox3DRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, output_dim=11, num_cls=10, normalize_yaw=False, refine_yaw=False, with_cls_branch=True, with_quality_estimation=False):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims, self.output_dim, self.num_cls = embed_dims, output_dim, num_cls
        self.normalize_yaw, self.refine_yaw = normalize_yaw, refine_yaw
        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw: self.refine_state += [SIN_YAW, COS_YAW]
        self.layers = nn.Sequential(*linear_relu_ln(embed_dims, 2, 2), Linear(self.embed_dims, self.output_dim), Scale([1.0] * self.output_dim))
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), Linear(self.embed_dims, self.num_cls))
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), Linear(self.embed_dims, 2))

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(self, instance_feature, anchor, anchor_embed, time_interval=1.0, return_cls=True):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        out_pos_size, anc_pos_size = output[..., :6], anchor[..., :6]
        new_pos_size = out_pos_size + anc_pos_size
        new_yaw = (output[..., 6:8] + anchor[..., 6:8]) if self.refine_yaw else output[..., 6:8]
        if self.normalize_yaw: new_yaw = torch.nn.functional.normalize(new_yaw, dim=-1)
        if self.output_dim > 8:
            out_vel, anc_vel = output[..., 8:], anchor[..., 8:]
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            time_interval = time_interval.reshape(-1, 1, 1)
            new_vel = (out_vel / time_interval) + anc_vel
        else:
            new_vel = output[..., 8:]
        output = torch.cat([new_pos_size, new_yaw, new_vel], dim=-1)
        cls = self.cls_layers(instance_feature) if return_cls else None
        quality = self.quality_layers(feature) if (return_cls and self.with_quality_estimation) else None
        return output, cls, quality

@PLUGIN_LAYERS.register_module(force=True)
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(self, embed_dims=256, num_learnable_pts=0, fix_scale=None):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims, self.num_learnable_pts = embed_dims, num_learnable_pts
        if fix_scale is None: fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(torch.tensor(fix_scale), requires_grad=False)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def forward(self, anchor, instance_feature=None, **kwargs):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (self.learnable_fc(instance_feature).reshape(bs, num_anchor, self.num_learnable_pts, 3).sigmoid() - 0.5)
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)
        cos_y, sin_y = anchor[:, :, COS_YAW], anchor[:, :, SIN_YAW]
        zeros, ones = torch.zeros_like(cos_y), torch.ones_like(cos_y)
        row1 = torch.stack([cos_y, -sin_y, zeros], dim=-1)
        row2 = torch.stack([sin_y, cos_y, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)
        rotation_mat = torch.stack([row1, row2, row3], dim=-2)
        rotated_point = torch.matmul(rotation_mat[:, :, None], key_points[..., None])
        center = anchor[..., [X, Y, Z]][:, :, None, :, None]
        key_points = rotated_point + center
        return key_points[..., 0]

    @staticmethod
    def anchor_projection(anchor, T_src2dst_list, time_intervals=None, **kwargs):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            T_mat = T_src2dst_list[i].to(dtype=anchor.dtype)
            R, t = T_mat[..., :3, :3].unsqueeze(1), T_mat[..., :3, 3].unsqueeze(1)
            vel, time_interval = anchor[..., VX:], (time_intervals[i] if time_intervals is not None else 0.0)
            if not isinstance(time_interval, torch.Tensor):
                time_interval = anchor.new_tensor(time_interval)
            center = anchor[..., [X, Y, Z]] - (vel * time_interval.reshape(-1, 1))
            center = torch.matmul(R, center[..., None])[..., 0] + t
            size = anchor[..., [W, L, H]]
            yaw = torch.matmul(R[..., :2, :2], anchor[..., [COS_YAW, SIN_YAW], None])[..., 0]
            y_cos, y_sin = yaw.split(1, dim=-1)
            yaw = torch.cat([y_sin, y_cos], dim=-1)
            vel = torch.matmul(R[..., :vel.shape[-1], :vel.shape[-1]], vel[..., None])[..., 0]
            dst_anchors.append(torch.cat([center, size, yaw, vel], dim=-1))
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)
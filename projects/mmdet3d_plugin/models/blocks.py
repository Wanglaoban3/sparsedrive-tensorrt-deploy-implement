from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import Linear

# [FIX] 补全丢失的 mmcv 工具函数导入
from mmcv.cnn import build_activation_layer, build_norm_layer, constant_init, xavier_init
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

@ATTENTION.register_module(force=True)
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        
        kps_output = self.kps_generator(anchor, instance_feature)
        if isinstance(kps_output, tuple):
            key_points = kps_output[0]
        else:
            key_points = kps_output
            
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
            
            if features.dtype != instance_feature.dtype:
                features = features.to(instance_feature.dtype)

        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)
        
        output = self.proj_drop(self.output_proj(features))
        
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            # [Fix] Broadcasting
            feature_expanded = feature.unsqueeze(2).expand(-1, -1, self.num_cams, -1)
            feature = feature_expanded + camera_embed.unsqueeze(1)

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]
        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)

        raw_pts = torch.matmul(
            projection_mat[:, :, None, None], 
            pts_extend[:, None, ..., None]
        )

        # 2. [关键修复] 使用 view 替代 [..., 0] 或 squeeze
        # 显式指定我们要把最后的 1 维去掉，变成 [B, Cam, Anc, Pts, 4]
        # -1 会自动推导中间的维度，确保 Batch 维和数据维的绝对安全
        points_2d = raw_pts.view(bs, -1, num_anchor, num_pts, 4)
        
        # 3. 保持原版的 clamp 逻辑以确保精度
        # 注意：不要用 + 1e-5，因为如果 depth 是负数，+1e-5 依然是负数且接近 0，
        # 这就是你之前 mATE 飙升的原因。
        depth = points_2d[..., 2:3]
        points_2d = points_2d[..., 0:2] / torch.clamp(depth, min=1e-5)
        
        # ==================================================================
        # [修复] 健壮的 shape 广播，解决 IndexError/RuntimeError
        # ==================================================================
        if image_wh is not None:
            # 无论输入是 [2], [1, 2] 还是 [B, 2]，统一 reshape 成 [B, 1, 1, 1, 2]
            # 这样就能正确广播到 points_2d 的 [B, N_cam, N_anc, N_pts, 2]
            
            # 如果是 1D [W, H] -> 变成 [1, 1, 1, 1, 2]
            if image_wh.dim() == 1:
                w_h = image_wh.view(1, 1, 1, 1, 2)
                points_2d = points_2d / w_h
                
            # 如果是 2D [B, 2] -> 变成 [B, 1, 1, 1, 2]
            elif image_wh.dim() == 2:
                w_h = image_wh.view(bs, 1, 1, 1, 2)
                points_2d = points_2d / w_h
            
            # 旧逻辑保留做 fallback，或者直接删除
            else:
                # 只有当维度 > 2 且符合旧逻辑预期时才用旧逻辑
                points_2d = points_2d / image_wh[:, :, None, None]
                
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features

@PLUGIN_LAYERS.register_module(force=True)
class DenseDepthNet(BaseModule):
    def __init__(self, embed_dims=256, num_depth_layers=1, equal_focal=100, max_depth=60, loss_weight=1.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight
        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        return 0.0 # Simplified

@FEEDFORWARD_NETWORK.register_module(force=True)
class AsymmetricFFN(BaseModule):
    def __init__(self, in_channels=None, pre_norm=None, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=dict(type="ReLU", inplace=True), ffn_drop=0.0, dropout_layer=None, add_identity=True, init_cfg=None, **kwargs):
        super(AsymmetricFFN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]
        for _ in range(num_fcs - 1):
            layers.append(Sequential(Linear(in_channels, feedforward_channels), self.activate, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = torch.nn.Identity() if in_channels == embed_dims else Linear(self.in_channels, embed_dims)

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
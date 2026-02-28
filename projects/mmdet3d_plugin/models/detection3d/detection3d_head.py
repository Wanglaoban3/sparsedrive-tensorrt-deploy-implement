from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from ..blocks import DeformableFeatureAggregation as DFG

# === ONNX Friendly TopK ===
def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    # 使用 topk 算子
    confidence, indices = torch.topk(confidence, k, dim=1)
    
    # 构造 batch 索引偏移 (Safe for ONNX)
    batch_idx = torch.arange(bs, device=confidence.device).unsqueeze(1) * N
    flat_indices = (indices + batch_idx).reshape(-1)
    
    outputs = []
    for input in inputs:
        # Flatten [B, N, C] -> [B*N, C]
        flat_input = input.flatten(end_dim=1)
        # Gather
        selected = flat_input.index_select(0, flat_indices) # 使用 index_select 更稳健
        # Reshape back [B, K, C]
        outputs.append(selected.reshape(bs, k, -1))
        
    return confidence, outputs

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        gt_id_key: str = "instance_id",
        with_instance_id: bool = True,
        task_prefix: str = 'det',
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_id_key = gt_id_key
        self.with_instance_id = with_instance_id
        self.task_prefix = task_prefix
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if self.gt_id_key in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x[self.gt_id_key]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor_embed = anchor_embed[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
            }
        )

        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if self.with_instance_id:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            reg_target_full = reg_target.clone()
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                prefix=f"{self.task_prefix}_",
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )

    # === 修正后的：ONNX 导出专用前向函数 (Myelin-Safe Z-Lock) ===
    # 核心思路：用 Mask (乘法) 代替 Slice/Concat，避免图碎片化导致 Myelin 崩溃
    def forward_onnx(
        self,
        feature_maps: Union[torch.Tensor, List],
        prev_instance_feature: torch.Tensor,
        prev_anchor: torch.Tensor,
        instance_t_matrix: torch.Tensor,
        time_interval: torch.Tensor = None, 
        prev_confidence: torch.Tensor = None,
        metas: dict = None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # 1. 准备 Query
        instance_feature = self.instance_bank.instance_feature.unsqueeze(0).repeat(batch_size, 1, 1).to(prev_instance_feature.device)
        anchor = self.instance_bank.anchor.unsqueeze(0).repeat(batch_size, 1, 1).to(prev_anchor.device)
        anchor_embed = self.anchor_encoder(anchor)

        # 2. 准备 Key (History)
        is_first_frame = (prev_instance_feature.abs().sum() == 0)
        
        if is_first_frame:
            temp_instance_feature = None
            temp_anchor_embed = None
            current_prev_anchor = None
            cached_feature = None
            cached_anchor = None
        else:
            temp_instance_feature = prev_instance_feature
            cached_feature = prev_instance_feature
            
            # 3. 计算 Key Pos
            if self.task_prefix == 'det':
                # 分离 XYZ (0-2) 和 剩余部分 (3-)
                # 使用 split 而不是切片，或者直接取
                xyz = prev_anchor[..., :3]
                rest = prev_anchor[..., 3:]
                
                # 获取 3D 速度 (位于 rest 的 5-8 位: 因为 prev_anchor 是 XYZ(3)+WHL(3)+Yaw(2)+Vel(3)=11)
                # 注意：rest 索引 0-2=WHL, 3-4=Yaw, 5-7=Vel
                vel = rest[..., 5:8]
                
                # 构造掩码 (Shape: [3])
                # Mask XY: [1, 1, 0] 用于保留 XY 变化
                mask_xy = torch.tensor([1.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype)
                # Mask Z:  [0, 0, 1] 用于锁定 Z 轴
                mask_z  = torch.tensor([0.0, 0.0, 1.0], device=xyz.device, dtype=xyz.dtype)
                
                if time_interval is not None:
                    dt = time_interval.view(batch_size, 1, 1)
                else:
                    dt = 0.5 
                
                # [Fix 1] 仅计算 XY 位移 (通过掩码，保持 3D Tensor 完整性)
                # displacement = vel * dt * [1, 1, 0]
                displacement = vel * dt * mask_xy
                
                # 预测位置
                xyz_pred = xyz + displacement
                
                # [Fix 2] Ego Motion (仅 XY 平移)
                R = instance_t_matrix[:, :3, :3]
                t = instance_t_matrix[:, :3, 3].unsqueeze(1)
                
                # t_flat = t * [1, 1, 0] (消除 t_z)
                t_flat = t * mask_xy
                
                # 坐标变换 (R @ pos + t_flat)
                xyz_new = torch.matmul(xyz_pred, R.transpose(1, 2)) + t_flat
                
                # [Fix 3] Z-Lock (强行写回旧 Z)
                # xyz_final = xyz_new * [1, 1, 0] + xyz * [0, 0, 1]
                # 这样完全避免了 tensor[..., 2] = ... 这种 inplace 操作或 concat
                xyz_final = xyz_new * mask_xy + xyz * mask_z
                
                # 旋转 Yaw (2D)
                prev_yaw = rest[..., 3:5] # Sin, Cos
                cos_sin_vec = torch.stack([prev_yaw[..., 1], prev_yaw[..., 0]], dim=-1)
                R_xy = R[:, :2, :2]
                cos_sin_new = torch.matmul(cos_sin_vec, R_xy.transpose(1, 2))
                cos_sin_new = torch.nn.functional.normalize(cos_sin_new, dim=-1)
                prev_yaw_new = torch.cat([cos_sin_new[..., 1:2], cos_sin_new[..., 0:1]], dim=-1)
                
                # 旋转 Velocity (2D)
                # 同样只旋转前两维，避免污染 Z
                prev_vel_xy = vel[..., :2]
                prev_vel_xy_new = torch.matmul(prev_vel_xy, R_xy.transpose(1, 2))
                
                # 拼装 (WHL 不变, Vel_Z 保持原样)
                # rest split: WHL(3), Yaw(2), Vel(3)
                whl = rest[..., 0:3]
                vel_z = rest[..., 7:8]
                
                # 最终拼接：XYZ(New) + WHL(Old) + Yaw(New) + VelXY(New) + VelZ(Old)
                current_prev_anchor = torch.cat([
                    xyz_final, 
                    whl, 
                    prev_yaw_new, 
                    prev_vel_xy_new, 
                    vel_z
                ], dim=-1)
            
            elif self.task_prefix == 'map':
                current_prev_anchor = prev_anchor 
            else:
                current_prev_anchor = prev_anchor

            cached_anchor = current_prev_anchor
            temp_anchor_embed = self.anchor_encoder(current_prev_anchor)

        # 4. Transformer Loop
        prediction, classification = [], []
        
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None: continue
            elif op == "temp_gnn":
                if temp_instance_feature is None:
                    instance_feature = self.graph_model(
                        i, instance_feature, None, None,
                        query_pos=anchor_embed, key_pos=None
                    )
                else:
                    instance_feature = self.graph_model(
                        i, instance_feature, temp_instance_feature, temp_instance_feature,
                        query_pos=anchor_embed, key_pos=temp_anchor_embed
                    )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, value=instance_feature, query_pos=anchor_embed
                )
            elif op in ["norm", "ffn"]:
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature, anchor, anchor_embed, feature_maps, metas
                )
            elif op == "refine":
                dummy_time = instance_feature.new_tensor([self.instance_bank.default_time_interval] * batch_size)
                anchor, cls, qt = self.layers[i](
                    instance_feature, anchor, anchor_embed,
                    time_interval=dummy_time, return_cls=True
                )
                prediction.append(anchor)
                classification.append(cls)
                
                if len(prediction) == self.num_single_frame_decoder:
                    if cached_feature is not None:
                        num_new = self.instance_bank.num_anchor - self.instance_bank.num_temp_instances
                        curr_conf = cls.max(dim=-1).values.sigmoid()
                        _, (selected_feature, selected_anchor) = topk(curr_conf, num_new, instance_feature, anchor)
                        
                        instance_feature = torch.cat([cached_feature, selected_feature], dim=1)
                        anchor = torch.cat([cached_anchor, selected_anchor], dim=1)
                        anchor_embed = self.anchor_encoder(anchor)
                    else:
                        anchor_embed = self.anchor_encoder(anchor)
                else:
                    anchor_embed = self.anchor_encoder(anchor)

        # 5. TopK & Confidence Decay
        last_cls, last_pred = classification[-1], prediction[-1]
        num_temp = self.instance_bank.num_temp_instances
        confidence = last_cls.max(dim=-1).values.sigmoid() 

        if prev_confidence is not None and not is_first_frame:
            decayed_conf = prev_confidence * self.instance_bank.confidence_decay
            if decayed_conf.shape[1] == num_temp: 
                 confidence[:, :num_temp] = torch.maximum(decayed_conf, confidence[:, :num_temp])
        
        next_conf_vals, (next_instance_feature, next_anchor) = topk(
            confidence, num_temp, instance_feature, last_pred
        )

        return {
            "cls_scores": last_cls,
            "bbox_preds": last_pred,
            "next_instance_feature": next_instance_feature,
            "next_anchor": next_anchor, 
            "next_confidence": next_conf_vals
        }
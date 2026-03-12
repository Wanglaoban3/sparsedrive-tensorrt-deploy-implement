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
import torch.nn.functional as F
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

def trt_friendly_topk(confidence, k, *tensors):
    """
    专为 TensorRT 优化的 TopK 与数据捞取逻辑。
    使用完全动态的张量维度，避免评估时因写死尺寸导致越界或空输出。
    """
    B = confidence.shape[0]
    N = confidence.shape[1]
    
    topk_vals, topk_indices = torch.topk(confidence, k, dim=1)
    
    # 动态构建 1D 扁平索引
    batch_offsets = torch.arange(B, device=confidence.device, dtype=torch.long).unsqueeze(1) * N
    flat_indices = (topk_indices + batch_offsets).view(-1)
    
    res_tensors = []
    for t in tensors:
        shape_rest = t.shape[2:]
        flat_t = t.view(B * N, -1) 
        gathered_t = flat_t[flat_indices] 
        res_tensors.append(gathered_t.view(B, k, *shape_rest))
        
    return topk_vals, res_tensors

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

        if getattr(self, 'debug_mode', False):
            self.debug_native_inputs = {
                "instance_feature": instance_feature.clone() if instance_feature is not None else None,
                "anchor": anchor.clone() if anchor is not None else None,
                "anchor_embed": anchor_embed.clone() if anchor_embed is not None else None,
                "temp_instance_feature": temp_instance_feature.clone() if temp_instance_feature is not None else None,
                "temp_anchor": temp_anchor.clone() if temp_anchor is not None else None,
                "temp_anchor_embed": temp_anchor_embed.clone() if temp_anchor_embed is not None else None,
                "time_interval": time_interval.clone() if time_interval is not None else None,
            }
        # ============================

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

    def forward_onnx(
        self,
        feature_maps: Union[torch.Tensor, List],
        prev_instance_feature: torch.Tensor,
        prev_anchor: torch.Tensor,
        instance_t_matrix: torch.Tensor,
        time_interval: torch.Tensor = None, 
        prev_confidence: torch.Tensor = None,
        # 🎯 新增 Tracking 专用历史变量
        prev_instance_id: torch.Tensor = None,
        prev_id_count: torch.Tensor = None,
        metas: dict = None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        device = feature_maps[0].device
        dtype = feature_maps[0].dtype

        # ==============================================================
        # 1-4 步：绝对锁定，绝不修改经过验证的对齐逻辑！
        # ==============================================================
        
        # 1. 初始化当前帧 Query
        instance_feature = self.instance_bank.instance_feature.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        anchor = self.instance_bank.anchor.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        anchor_embed = self.anchor_encoder(anchor)

        # 2. 时间差处理与场景重置判定
        if time_interval is not None:
            raw_dt = time_interval.view(batch_size)
        else:
            raw_dt = instance_feature.new_tensor([self.instance_bank.default_time_interval] * batch_size)
        
        is_valid_history = (prev_instance_feature.abs().sum() > 0) & (raw_dt.abs() <= self.instance_bank.max_time_interval)
        
        dt_for_refine = torch.where(
            is_valid_history, 
            raw_dt, 
            raw_dt.new_tensor(self.instance_bank.default_time_interval)
        ).view(-1, 1, 1)

        # 3. 历史状态投影
        if not is_valid_history.any():
            temp_instance_feature = None
            temp_anchor_embed = None
            cached_feature = None
            cached_anchor = None
        else:
            temp_instance_feature = prev_instance_feature.to(dtype)
            cached_feature = temp_instance_feature
            
            # ================= 分支路由：Det 头 vs Map 头 =================
            if self.task_prefix == 'det':
                xyz = prev_anchor[..., :3].to(dtype)
                rest = prev_anchor[..., 3:].to(dtype)
                vel = rest[..., 5:8]
                
                dt_view = raw_dt.view(-1, 1, 1)
                displacement = vel * dt_view
                xyz_pred = xyz + displacement
                xyz_pred_h = torch.cat([xyz_pred, torch.ones_like(xyz_pred[..., :1])], dim=-1)
                
                xyz_final = torch.matmul(xyz_pred_h, instance_t_matrix.to(dtype).transpose(1, 2))[..., :3]
                
                R_full = instance_t_matrix[:, :3, :3].to(dtype)
                R_xy = R_full[:, :2, :2]
                
                prev_yaw = rest[..., 3:5]
                swap_mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)
                cos_sin_vec = torch.matmul(prev_yaw, swap_mat)
                cos_sin_new = torch.matmul(cos_sin_vec, R_xy.transpose(1, 2))
                prev_yaw_new = torch.matmul(cos_sin_new, swap_mat)
                
                prev_vel_new = torch.matmul(vel, R_full.transpose(1, 2))
                
                yaw_padded = F.pad(prev_yaw_new, (3, 3))
                vel_padded = F.pad(prev_vel_new, (5, 0))
                mask_keep = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
                rest_new = (rest * mask_keep) + yaw_padded + vel_padded
                
                current_prev_anchor = torch.cat([xyz_final, rest_new], dim=-1)

            elif self.task_prefix == 'map':
                bs_map, num_anchor_map, dims_map = prev_anchor.shape
                
                pts = prev_anchor.to(dtype).reshape(bs_map, -1, 2)
                
                R_2x2 = instance_t_matrix[:, :2, :2].to(dtype).unsqueeze(1) # [bs, 1, 2, 2]
                t_2 = instance_t_matrix[:, :2, 3].to(dtype).unsqueeze(1)    # [bs, 1, 2]
                
                pts_rotated = torch.matmul(R_2x2, pts.unsqueeze(-1)).squeeze(-1)
                pts_final = pts_rotated + t_2
                
                current_prev_anchor = pts_final.reshape(bs_map, num_anchor_map, dims_map)
                
            else:
                current_prev_anchor = prev_anchor.to(dtype)
            # ==============================================================
            
            cached_anchor = current_prev_anchor
            temp_anchor_embed = self.anchor_encoder(current_prev_anchor)
        
        if getattr(self, 'debug_mode', False):
            self.debug_onnx_inputs = {
                "instance_feature": instance_feature.clone() if instance_feature is not None else None,
                "anchor": anchor.clone() if anchor is not None else None,
                "anchor_embed": anchor_embed.clone() if anchor_embed is not None else None,
                "temp_instance_feature": temp_instance_feature.clone() if temp_instance_feature is not None else None,
                "temp_anchor": cached_anchor.clone() if 'cached_anchor' in locals() and cached_anchor is not None else None,
                "temp_anchor_embed": temp_anchor_embed.clone() if temp_anchor_embed is not None else None,
                "time_interval": dt_for_refine.clone() if 'dt_for_refine' in locals() else None,
            }
        # ============================

        # 4. Transformer 层迭代
        prediction, classification, quality = [], [], []
        
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
                anchor, cls, qt = self.layers[i](
                    instance_feature, anchor, anchor_embed,
                    time_interval=dt_for_refine, return_cls=True
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                
                if len(prediction) == self.num_single_frame_decoder:
                    if cached_feature is not None:
                        curr_conf = cls.max(dim=-1).values
                        num_new = self.instance_bank.num_anchor - self.instance_bank.num_temp_instances
                        _, (selected_feature, selected_anchor) = topk(curr_conf, num_new, instance_feature, anchor)
                        
                        instance_feature = torch.where(
                            is_valid_history.view(-1, 1, 1),
                            torch.cat([cached_feature, selected_feature], dim=1),
                            instance_feature
                        )
                        anchor = torch.where(
                            is_valid_history.view(-1, 1, 1),
                            torch.cat([cached_anchor, selected_anchor], dim=1),
                            anchor
                        )
                    anchor_embed = self.anchor_encoder(anchor)
                else:
                    anchor_embed = self.anchor_encoder(anchor)

                if len(prediction) > self.num_single_frame_decoder and temp_anchor_embed is not None:
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]

        # ==============================================================================
        # 5. 生成下一帧缓存 & 外挂 ID 分配系统 (🔥 TRT 终极动态矩阵融合版)
        # ==============================================================================
        last_cls, last_pred = classification[-1], prediction[-1]
        last_qt = quality[-1]
        num_temp = self.instance_bank.num_temp_instances
        
        confidence = last_cls.max(dim=-1).values.sigmoid() 

        if prev_confidence is not None and is_valid_history.any():
            decayed_conf = prev_confidence.to(dtype) * self.instance_bank.confidence_decay
            confidence[:, :num_temp] = torch.maximum(decayed_conf, confidence[:, :num_temp])

        if getattr(self, 'with_instance_id', False):
            # 获取动态的 B 和 N，绝对不强制转 int
            B = confidence.shape[0]
            N = confidence.shape[1]
            
            instance_id = torch.full((B, N), -1, dtype=torch.int32, device=device)
            
            if prev_instance_id is not None and is_valid_history.any():
                num_prev = prev_instance_id.shape[1]
                instance_id[:, :num_prev] = prev_instance_id.to(torch.int32)
                
            score_thresh = getattr(self.decoder, 'score_threshold', None) if hasattr(self, 'decoder') else None
            if score_thresh is not None:
                new_id_mask = (instance_id < 0) & (confidence >= score_thresh)
            else:
                new_id_mask = (instance_id < 0)
                
            if prev_id_count is None:
                prev_id_count = instance_id.new_zeros((B, 1))
                
            # =========================================================================
            # 🔥 神级替换：用“动态上三角矩阵乘法”完美替换串行的 cumsum！
            # 既保留了 ONNX 动态形状，又完美调用了 3090 的高并发 FP32 算力！
            # =========================================================================
            idx = torch.arange(N, device=device)
            
            # 行号 <= 列号，利用广播生成上三角全 1 矩阵
            upper_tri_mat = (idx.unsqueeze(1) <= idx.unsqueeze(0)).to(torch.float32)
            
            # 将掩码转为 float32 用于矩阵相乘 (规避 Int 不能 matmul 的报错)
            mask_float = new_id_mask.to(torch.float32)
            
            # [B, 1, N] @ [N, N] -> [B, 1, N] -> squeeze -> [B, N]
            new_id_offsets = torch.matmul(mask_float.unsqueeze(1), upper_tri_mat).squeeze(1).to(torch.int32)
            
            # 生成新 ID
            new_ids = prev_id_count + new_id_offsets - 1
            
            # 安全填入 (保留原汁原味的 where，TRT 处理纯元素级 where 非常快)
            instance_id = torch.where(new_id_mask, new_ids, instance_id)
            
            # 结算本帧一共发了多少新 ID
            # 最后一个元素天然就是前缀和的总数，直接拿来用，省去一次求和！
            next_id_count = prev_id_count + new_id_offsets[:, -1:]
            
            # 🎯 扁平化极速 Gather
            next_conf_vals, gathered_res = trt_friendly_topk(
                confidence, num_temp, 
                instance_feature, 
                last_pred, 
                instance_id.unsqueeze(-1)
            )
            next_instance_feature, next_anchor, next_instance_id = gathered_res
            next_instance_id = next_instance_id.view(B, num_temp).to(torch.int32)

            return {
                "cls_scores": last_cls,
                "bbox_preds": last_pred,
                "quality": last_qt, 
                "instance_id": instance_id,              
                "instance_feature": instance_feature,    
                "anchor_embed": anchor_embed,            
                "next_instance_feature": next_instance_feature,
                "next_anchor": next_anchor, 
                "next_confidence": next_conf_vals,
                "next_instance_id": next_instance_id,    
                "next_id_count": next_id_count           
            }
        else:
            # Map 等不需要跟踪 ID 的 Head 走这里
            next_conf_vals, gathered_res = trt_friendly_topk(
                confidence, num_temp, 
                instance_feature, 
                last_pred
            )
            next_instance_feature, next_anchor = gathered_res
            
            return {
                "cls_scores": last_cls,
                "bbox_preds": last_pred,
                "quality": last_qt, 
                "instance_feature": instance_feature,    
                "anchor_embed": anchor_embed,            
                "next_instance_feature": next_instance_feature,
                "next_anchor": next_anchor, 
                "next_confidence": next_conf_vals
            }
    
    def forward_debug(self, feature_maps: Union[torch.Tensor, List], metas: dict):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # 走原生的 InstanceBank 提取逻辑
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=None
        )

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        return {
            "instance_feature": instance_feature,
            "anchor": anchor,
            "anchor_embed": anchor_embed,
            "temp_instance_feature": temp_instance_feature,
            "temp_anchor": temp_anchor,
            "temp_anchor_embed": temp_anchor_embed,
            "time_interval": time_interval,
        }

    # ==============================================================================
    # 🔍 用于严格对齐输入的 Debug 函数 (ONNX 纯手工投影逻辑)
    # ==============================================================================
    def forward_onnx_debug(
        self,
        feature_maps: Union[torch.Tensor, List],
        prev_instance_feature: torch.Tensor,
        prev_anchor: torch.Tensor,
        instance_t_matrix: torch.Tensor,
        time_interval: torch.Tensor = None, 
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        dtype = feature_maps[0].dtype
        device = feature_maps[0].device

        # 1. 初始化当前帧 Query
        instance_feature = self.instance_bank.instance_feature.unsqueeze(0).repeat(batch_size, 1, 1).to(device=device, dtype=dtype)
        anchor = self.instance_bank.anchor.unsqueeze(0).repeat(batch_size, 1, 1).to(device=device, dtype=dtype)
        anchor_embed = self.anchor_encoder(anchor)

        # 2. 准备 dt
        if time_interval is not None:
            dt = time_interval.view(batch_size).to(dtype)
        else:
            dt = instance_feature.new_tensor([self.instance_bank.default_time_interval] * batch_size).to(dtype)

        is_first_frame = (prev_instance_feature.abs().sum() == 0)

        # 3. 历史状态投影
        if is_first_frame:
            temp_instance_feature = None
            temp_anchor_embed = None
            current_prev_anchor = None
        else:
            temp_instance_feature = prev_instance_feature.to(dtype)
            
            # ============= 核心分支路由：隔离 Det 和 Map =============
            if self.task_prefix == 'det':
                xyz = prev_anchor[..., :3].to(dtype)
                rest = prev_anchor[..., 3:].to(dtype)
                vel = rest[..., 5:8]
                
                dt_view = dt.view(-1, 1, 1)
                displacement = vel * dt_view
                xyz_pred = xyz + displacement
                xyz_pred_h = torch.cat([xyz_pred, torch.ones_like(xyz_pred[..., :1])], dim=-1)
                
                xyz_final = torch.matmul(xyz_pred_h, instance_t_matrix.to(dtype).transpose(1, 2))[..., :3]
                
                R_full = instance_t_matrix[:, :3, :3].to(dtype)
                R_xy = R_full[:, :2, :2]
                
                prev_yaw = rest[..., 3:5] 
                swap_mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)
                cos_sin_vec = torch.matmul(prev_yaw, swap_mat)
                cos_sin_new = torch.matmul(cos_sin_vec, R_xy.transpose(1, 2))
                prev_yaw_new = torch.matmul(cos_sin_new, swap_mat)
                
                prev_vel_new = torch.matmul(vel, R_full.transpose(1, 2))
                
                import torch.nn.functional as F
                yaw_padded = F.pad(prev_yaw_new, (3, 3))
                vel_padded = F.pad(prev_vel_new, (5, 0))
                mask_keep = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
                rest_new = (rest * mask_keep) + yaw_padded + vel_padded
                
                current_prev_anchor = torch.cat([xyz_final, rest_new], dim=-1)

            elif self.task_prefix == 'map':
                # 【破案修复】: Map 锚点是静态的 2D 坐标 (bs, num_anchor, num_pts * 2)
                bs_map, num_anchor_map, dims_map = prev_anchor.shape
                
                # 重塑为 2D 坐标 (bs, N, 2)，这里的 2 才是灵魂！
                pts = prev_anchor.to(dtype).reshape(bs_map, -1, 2)
                
                # 直接提取 2x2 旋转矩阵和 2D 平移向量
                R_2x2 = instance_t_matrix[:, :2, :2].to(dtype).unsqueeze(1) # [bs, 1, 2, 2]
                t_2 = instance_t_matrix[:, :2, 3].to(dtype).unsqueeze(1)    # [bs, 1, 2]
                
                # 纯正的 2D 矩阵相乘：(bs, 1, 2, 2) @ (bs, N, 2, 1) -> (bs, N, 2, 1) -> (bs, N, 2)
                pts_rotated = torch.matmul(R_2x2, pts.unsqueeze(-1)).squeeze(-1)
                
                # 加上平移向量
                pts_final = pts_rotated + t_2
                
                # 还原为 (bs, 33, 40) 的原始形状
                current_prev_anchor = pts_final.reshape(bs_map, num_anchor_map, dims_map)
                
            else:
                current_prev_anchor = prev_anchor.to(dtype)
            # =========================================================

            temp_anchor_embed = self.anchor_encoder(current_prev_anchor)

        return {
            "instance_feature": instance_feature,
            "anchor": anchor,
            "anchor_embed": anchor_embed,
            "temp_instance_feature": temp_instance_feature,
            "temp_anchor": current_prev_anchor,
            "temp_anchor_embed": temp_anchor_embed,
            "time_interval": dt,
        }
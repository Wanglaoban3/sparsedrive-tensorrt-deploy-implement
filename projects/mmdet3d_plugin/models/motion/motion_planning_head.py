from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

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

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map

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

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        # 1. 取出网络预测的原始 sin 和 cos 值
        raw_sin = boxes[..., SIN_YAW]
        raw_cos = boxes[..., COS_YAW]
        
        # 2. 计算模长 (加 1e-6 防止除以 0)
        norm = torch.sqrt(raw_sin ** 2 + raw_cos ** 2 + 1e-6)
        
        # 3. 直接归一化得到最终的 sin_yaw 和 cos_yaw，彻底抛弃 atan2！
        sin_yaw = raw_sin / norm
        cos_yaw = raw_cos / norm
        
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

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
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) 
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result
    

   # === [Debug 版] 100% 复刻 forward 逻辑，仅加 print 和中间值返回 ===
    def forward_debug(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):
        # 原逻辑开始
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        
        # 【关键】抓取张量以便 Debug 比对，因为下方会把它 flatten 掉
        debug_temp_instance_feature = temp_instance_feature.clone()
        debug_temp_anchor = temp_anchor.clone()

        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature_flat = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed_flat = temp_anchor_embed.flatten(0, 1)
        temp_mask_flat = temp_mask.flatten(0, 1)

        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(self.plan_anchor[None], (bs, 1, 1, 1, 1))

        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        motion_classification, motion_prediction = [], []
        planning_classification, planning_prediction, planning_status = [], [], []
        
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None: continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature_flat,
                    temp_instance_feature_flat,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed_flat,
                    key_padding_mask=temp_mask_flat,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, instance_feature_selected, instance_feature_selected,
                    query_pos=anchor_embed, key_pos=anchor_embed_selected,
                )
            elif op in ["norm", "ffn"]:
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature, key=map_instance_feature_selected,
                    query_pos=anchor_embed, key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) 
                m_cls, m_reg, p_cls, p_reg, p_status = self.layers[i](
                    motion_query, plan_query, instance_feature[:, num_anchor:], anchor_embed[:, num_anchor:]
                )
                motion_classification.append(m_cls)
                motion_prediction.append(m_reg)
                planning_classification.append(p_cls)
                planning_prediction.append(p_reg)
                planning_status.append(p_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], planning_status[-1])

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output, debug_temp_instance_feature, debug_temp_anchor

    def forward_onnx(
        self,
        # === Det & Map 特征输入 ===
        det_instance_feature,       # [B, num_det, dim]
        det_anchor_embed,           # [B, num_det, dim]
        det_classification_sigmoid, # [B, num_det, num_classes] (已过sigmoid)
        det_anchors,                # [B, num_det, 11]
        det_instance_id,            # [B, num_det]
        map_instance_feature,       # [B, num_map, dim]
        map_anchor_embed,           # [B, num_map, dim]
        map_classification_sigmoid, # [B, num_map, num_classes] (已过sigmoid)
        # === Ego 规划所需特征 ===
        ego_feature_map,            # [B, C, H, W]
        # === 工具句柄与配置 ===
        anchor_encoder,
        anchor_handler,
        mask,                       # [B]
        is_first_frame,             
        # === 外部维护的历史张量状态 ===
        T_temp2cur=None,            # [B, 4, 4]
        history_instance_feature=None, 
        history_anchor=None,           
        history_period=None,           
        prev_instance_id=None,         
        prev_confidence=None,          
        history_ego_feature=None,      
        history_ego_anchor=None,       
        history_ego_period=None,       
        prev_ego_status=None           
    ):
        bs, num_anchor, dim = det_instance_feature.shape
        queue_length = self.instance_queue.queue_length
        VY = 6
        
        # 1. Det & Map TopK
        det_confidence = det_classification_sigmoid.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, det_instance_feature, det_anchor_embed
        )
        map_confidence = map_classification_sigmoid.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # 2. Ego 特征初始化
        ego_feature = self.instance_queue.ego_feature_encoder(ego_feature_map)
        ego_feature = ego_feature.unsqueeze(1).squeeze(-1).squeeze(-1) # [B, 1, dim]
        ego_anchor = self.instance_queue.ego_anchor[None].repeat(bs, 1, 1) # [B, 1, 11]

        # 3. 手动实现 instance_queue.get() 逻辑 (完全使用你跑通的代码2)
        if is_first_frame:
            new_history_feature = det_instance_feature.new_zeros((bs, num_anchor, queue_length, dim))
            new_history_feature[:, :, -1, :] = det_instance_feature
            
            new_history_anchor = det_anchors.new_zeros((bs, num_anchor, queue_length, 11))
            new_history_anchor[:, :, -1, :] = det_anchors
            
            new_period = det_instance_id.new_ones((bs, num_anchor)).to(torch.int32)
            
            new_history_ego_feature = ego_feature.new_zeros((bs, 1, queue_length, dim))
            new_history_ego_feature[:, :, -1, :] = ego_feature
            
            new_history_ego_anchor = ego_anchor.new_zeros((bs, 1, queue_length, 11))
            new_history_ego_anchor[:, :, -1, :] = ego_anchor
            
            new_ego_period = det_instance_id.new_ones((bs, 1)).to(torch.int32)
        else:
            mask_float = mask.to(prev_ego_status.dtype)
            prev_ego_status_masked = prev_ego_status * mask_float[:, None, None]
            ego_anchor[..., VY] = prev_ego_status_masked[..., 6]
            
            B, N, Q, _ = history_anchor.shape
            flat_history_anchor = history_anchor.view(B, N * Q, 11)
            flat_history_anchor = anchor_handler.anchor_projection(flat_history_anchor, [T_temp2cur])[0]
            history_anchor = flat_history_anchor.view(B, N, Q, 11)
            
            flat_ego_anchor = history_ego_anchor.view(B, Q, 11)
            flat_ego_anchor = anchor_handler.anchor_projection(flat_ego_anchor, [T_temp2cur])[0]
            history_ego_anchor = flat_ego_anchor.view(B, 1, Q, 11)
            
            # 完全沿用你正确的匹配逻辑
            match = det_instance_id.unsqueeze(-1) == prev_instance_id.unsqueeze(1) 
            match_float = match.to(history_instance_feature.dtype)
            
            if self.instance_queue.tracking_threshold > 0:
                track_mask = prev_confidence > self.instance_queue.tracking_threshold
                track_mask_float = track_mask.to(match_float.dtype)
                match_float = match_float * track_mask_float.unsqueeze(1) 
                
            match_long = match_float.to(history_period.dtype)

            matched_history_feature = (match_float.unsqueeze(-1).unsqueeze(-1) * history_instance_feature.unsqueeze(1)).sum(dim=2) 
            matched_history_anchor = (match_float.unsqueeze(-1).unsqueeze(-1) * history_anchor.unsqueeze(1)).sum(dim=2) 
            matched_period = (match_long * history_period.unsqueeze(1)).sum(dim=2)
            
            new_history_feature = torch.cat([matched_history_feature[:, :, 1:, :], det_instance_feature.unsqueeze(2)], dim=2)
            new_history_anchor = torch.cat([matched_history_anchor[:, :, 1:, :], det_anchors.unsqueeze(2)], dim=2)
            new_period = torch.clamp(matched_period + 1, 0, queue_length)
            
            mask_int = mask.to(history_ego_period.dtype)
            curr_history_ego_period = history_ego_period * mask_int[:, None]
            new_history_ego_feature = torch.cat([history_ego_feature[:, :, 1:, :], ego_feature.unsqueeze(2)], dim=2)
            new_history_ego_anchor = torch.cat([history_ego_anchor[:, :, 1:, :], ego_anchor.unsqueeze(2)], dim=2)
            new_ego_period = torch.clamp(curr_history_ego_period + 1, 0, queue_length)

        temp_instance_feature = torch.cat([new_history_feature, new_history_ego_feature], dim=1) 
        temp_anchor = torch.cat([new_history_anchor, new_history_ego_anchor], dim=1) 
        period = torch.cat([new_period, new_ego_period], dim=1) 
        
        num_agent = temp_anchor.shape[1]
        temp_mask_base = temp_anchor.new_tensor(range(queue_length, 0, -1)).view(1, 1, queue_length)
        temp_mask_base_float = temp_mask_base.to(torch.float32)
        period_float = period.unsqueeze(-1).to(torch.float32)
        diff = temp_mask_base_float - period_float
        temp_mask = diff > 0

        # 4. 后续网络推理逻辑与原版保持一致
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature_flat = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed_flat = temp_anchor_embed.flatten(0, 1)
        temp_mask_flat = temp_mask.flatten(0, 1)

        motion_anchor = self.get_motion_anchor(det_classification_sigmoid, det_anchors)
        plan_anchor = self.plan_anchor[None].repeat(bs, 1, 1, 1, 1)

        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)
        instance_feature = torch.cat([det_instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([det_anchor_embed, ego_anchor_embed], dim=1)

        motion_classification, motion_prediction = [], []
        planning_classification, planning_prediction, planning_status = [], [], []
        
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None: continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature_flat,
                    temp_instance_feature_flat,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed_flat,
                    key_padding_mask=temp_mask_flat,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, instance_feature_selected, instance_feature_selected,
                    query_pos=anchor_embed, key_pos=anchor_embed_selected,
                )
            elif op in ["norm", "ffn"]:
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature, key=map_instance_feature_selected,
                    query_pos=anchor_embed, key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) 
                m_cls, m_reg, p_cls, p_reg, p_status = self.layers[i](
                    motion_query, plan_query, instance_feature[:, num_anchor:], anchor_embed[:, num_anchor:]
                )
                motion_classification.append(m_cls)
                motion_prediction.append(m_reg)
                planning_classification.append(p_cls)
                planning_prediction.append(p_reg)
                planning_status.append(p_status)
        
        new_history_ego_feature_cache = new_history_ego_feature.clone()
        new_history_ego_feature_cache[:, :, -1, :] = instance_feature[:, num_anchor:]

        # 5. 组装缓存供外部流水线维持状态
        next_states = {
            "history_instance_feature": new_history_feature,
            "history_anchor": new_history_anchor,
            "history_period": new_period.to(torch.int32),
            "prev_instance_id": det_instance_id.to(torch.int32),
            "prev_confidence": det_confidence,
            "history_ego_feature": new_history_ego_feature_cache, 
            "history_ego_anchor": new_history_ego_anchor,
            "history_ego_period": new_ego_period.to(torch.int32),
            "prev_ego_status": planning_status[-1] 
        }

        return (
            motion_classification, motion_prediction,
            planning_classification, planning_prediction, planning_status,
            next_states,
            temp_instance_feature, temp_anchor  
        )
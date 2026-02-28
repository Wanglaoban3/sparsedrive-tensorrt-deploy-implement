import torch
import torch.nn as nn
import numpy as np
import os
import sys
import ctypes
import types
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector

# 1. 环境初始化
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 加载插件
plugin_lib_path = "projects/trt_plugin/build/libSparseDrivePlugin.so" 
if os.path.exists(plugin_lib_path):
    ctypes.CDLL(plugin_lib_path)

# ==============================================================================
# 🛠️ Helper Function: Inline TopK
# ==============================================================================
def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    topk_conf, topk_indices = torch.topk(confidence, k, dim=1)
    
    outputs = []
    for input in inputs:
        C = input.shape[-1]
        expanded_indices = topk_indices.unsqueeze(-1).expand(bs, k, C)
        selected = torch.gather(input, 1, expanded_indices).contiguous()
        outputs.append(selected)
        
    return topk_conf, outputs

# ==============================================================================
# 🕵️‍♂️ Debug Probe: Native Forward
# ==============================================================================
def debug_forward_native(self, feature_maps, metas):
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = [feature_maps]
    batch_size = feature_maps[0].shape[0]

    # --- 探针起点 ---
    (
        instance_feature,
        anchor,
        temp_instance_feature,
        temp_anchor, 
        time_interval,
    ) = self.instance_bank.get(
        batch_size, metas, dn_metas=self.sampler.dn_metas if hasattr(self.sampler, 'dn_metas') else None
    )
    
    # 🖨️ 打印 Native 关键信息
    print(f"\n[🔍 NATIVE DEBUG] Frame: {metas['img_metas'][0].get('sample_idx', 'Unknown')}")
    print(f"  > Time Interval (dt): {time_interval[0].item():.6f} s")
    
    if temp_anchor is not None:
        # 打印全局均值 (3,)
        mean_xyz = temp_anchor[..., :3].reshape(-1, 3).mean(0).detach().cpu().numpy()
        print(f"  > [Result] Projected Anchor Mean XYZ: {mean_xyz}")
    
    anchor_embed = self.anchor_encoder(anchor)
    if temp_anchor is not None:
        temp_anchor_embed = self.anchor_encoder(temp_anchor)
    else:
        temp_anchor_embed = None
        
    prediction = []
    classification = []
    quality = []
    
    # Transformer Loop
    for i, op in enumerate(self.operation_order):
        if self.layers[i] is None: continue
        elif op == "temp_gnn":
            if temp_instance_feature is None:
                instance_feature = self.graph_model(i, instance_feature, None, None, query_pos=anchor_embed, key_pos=None)
            else:
                instance_feature = self.graph_model(i, instance_feature, temp_instance_feature, temp_instance_feature, query_pos=anchor_embed, key_pos=temp_anchor_embed)
        elif op == "gnn":
            instance_feature = self.graph_model(i, instance_feature, value=instance_feature, query_pos=anchor_embed)
        elif op in ["norm", "ffn"]:
            instance_feature = self.layers[i](instance_feature)
        elif op == "deformable":
            instance_feature = self.layers[i](instance_feature, anchor, anchor_embed, feature_maps, metas)
        elif op == "refine":
            anchor, cls, qt = self.layers[i](instance_feature, anchor, anchor_embed, time_interval=time_interval, return_cls=True)
            prediction.append(anchor)
            classification.append(cls)
            quality.append(qt)
            if len(prediction) == self.num_single_frame_decoder:
                instance_feature, anchor = self.instance_bank.update(instance_feature, anchor, cls)
            anchor_embed = self.anchor_encoder(anchor)
            if len(prediction) > self.num_single_frame_decoder and temp_anchor_embed is not None:
                temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]

    self.instance_bank.cache(instance_feature, anchor, cls, metas, feature_maps)

    output = {
        "classification": classification,
        "prediction": prediction,
        "quality": quality,
        "instance_feature": instance_feature,
        "anchor_embed": anchor_embed,
    }

    if self.with_instance_id:
        instance_id = self.instance_bank.get_instance_id(cls, anchor, self.decoder.score_threshold)
        output["instance_id"] = instance_id

    return output


# ==============================================================================
# 🕵️‍♂️ Debug Probe: ONNX Forward
# ==============================================================================
def debug_forward_onnx(self, feature_maps, prev_instance_feature, prev_anchor, instance_t_matrix, time_interval=None, prev_confidence=None, metas=None):
    batch_size = prev_instance_feature.shape[0]

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
    else:
        temp_instance_feature = prev_instance_feature
        
        # --- 🖨️ 打印 ONNX 关键计算过程 ---
        print(f"\n[🔍 ONNX DEBUG] Frame: {metas['img_metas'][0].get('sample_idx', 'Unknown') if metas else '?'}")
        mean_in_xyz = prev_anchor[..., :3].reshape(-1, 3).mean(0).detach().cpu().numpy()
        print(f"  > [Input] Prev Anchor Mean XYZ: {mean_in_xyz}")
        
        R = instance_t_matrix[:, :3, :3]
        t = instance_t_matrix[:, :3, 3].unsqueeze(1)
        # [FIX] 修正索引错误
        print(f"  > [Ego] Translation (Z): {t[0, 0, 2].item():.6f}")

        prev_center = prev_anchor[..., :3]
        
        # [DEBUG] 强制使用 2D 速度 (Vz=0)
        prev_vel_2d = prev_anchor[..., 8:10]
        zeros = torch.zeros_like(prev_vel_2d[..., 0:1])
        prev_vel_3d = torch.cat([prev_vel_2d, zeros], dim=-1) 
        
        if time_interval is not None:
            dt = time_interval.view(batch_size, 1, 1)
        else:
            dt = 0.5
        print(f"  > Time Interval (dt): {dt[0,0,0].item():.6f} s")

        displacement = prev_vel_3d * dt
        disp_mean = displacement.reshape(-1, 3).mean(0).detach().cpu().numpy()
        print(f"  > [Calc] Displacement Mean: {disp_mean}")
        
        prev_center_pred = prev_center + displacement
        prev_center_new = torch.matmul(prev_center_pred, R.transpose(1, 2)) + t
        
        res_mean = prev_center_new.reshape(-1, 3).mean(0).detach().cpu().numpy()
        print(f"  > [Result] Projected Anchor Mean XYZ: {res_mean}")
        # ====================

        prev_yaw = prev_anchor[..., 6:8]
        cos_sin_vec = torch.stack([prev_yaw[..., 1], prev_yaw[..., 0]], dim=-1)
        R_xy = R[:, :2, :2]
        cos_sin_new = torch.matmul(cos_sin_vec, R_xy.transpose(1, 2))
        cos_sin_new = torch.nn.functional.normalize(cos_sin_new, dim=-1)
        prev_yaw_new = torch.cat([cos_sin_new[..., 1:2], cos_sin_new[..., 0:1]], dim=-1)
        
        prev_vel_new = torch.matmul(prev_anchor[..., 8:10], R_xy.transpose(1, 2))

        current_prev_anchor = torch.cat([prev_center_new, prev_anchor[..., 3:6], prev_yaw_new, prev_vel_new], dim=-1)
        if prev_anchor.shape[-1] > 10:
             needed = prev_anchor.shape[-1] - current_prev_anchor.shape[-1]
             if needed > 0:
                 current_prev_anchor = torch.cat([current_prev_anchor, prev_anchor[..., -needed:]], dim=-1)

        temp_anchor_embed = self.anchor_encoder(current_prev_anchor)

    # Transformer Loop
    prediction, classification = [], []
    for i, op in enumerate(self.operation_order):
        if self.layers[i] is None: continue
        elif op == "temp_gnn":
            if temp_instance_feature is None:
                instance_feature = self.graph_model(i, instance_feature, None, None, query_pos=anchor_embed, key_pos=None)
            else:
                instance_feature = self.graph_model(i, instance_feature, temp_instance_feature, temp_instance_feature, query_pos=anchor_embed, key_pos=temp_anchor_embed)
        elif op == "gnn":
            instance_feature = self.graph_model(i, instance_feature, value=instance_feature, query_pos=anchor_embed)
        elif op in ["norm", "ffn"]:
            instance_feature = self.layers[i](instance_feature)
        elif op == "deformable":
            instance_feature = self.layers[i](instance_feature, anchor, anchor_embed, feature_maps, metas)
        elif op == "refine":
            dummy_time = instance_feature.new_tensor([self.instance_bank.default_time_interval] * batch_size)
            anchor, cls, qt = self.layers[i](instance_feature, anchor, anchor_embed, time_interval=dummy_time, return_cls=True)
            prediction.append(anchor)
            classification.append(cls)
            if len(prediction) == self.num_single_frame_decoder:
                if not is_first_frame:
                    num_new = self.instance_bank.num_anchor - self.instance_bank.num_temp_instances
                    curr_conf = cls.max(dim=-1).values.sigmoid()
                    _, (selected_feature, selected_anchor) = topk(curr_conf, num_new, instance_feature, anchor)
                    instance_feature = torch.cat([prev_instance_feature, selected_feature], dim=1)
                    anchor = torch.cat([current_prev_anchor, selected_anchor], dim=1)
                anchor_embed = self.anchor_encoder(anchor)
            else:
                anchor_embed = self.anchor_encoder(anchor)

    last_cls, last_pred = classification[-1], prediction[-1]
    num_temp = self.instance_bank.num_temp_instances
    confidence = last_cls.max(dim=-1).values.sigmoid()
    
    if prev_confidence is not None and not is_first_frame:
        decayed_conf = prev_confidence * self.instance_bank.confidence_decay
        if decayed_conf.shape[1] == num_temp: 
                confidence[:, :num_temp] = torch.maximum(decayed_conf, confidence[:, :num_temp])

    next_conf_vals, (next_instance_feature, next_anchor) = topk(confidence, num_temp, instance_feature, last_pred)

    return {
        "cls_scores": last_cls, "bbox_preds": last_pred,
        "next_instance_feature": next_instance_feature, "next_anchor": next_anchor, "next_confidence": next_conf_vals
    }


# ==============================================================================
# 🏃‍♂️ Runner Logic
# ==============================================================================
class SparseDriveONNXPathWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.det_head = model.head.det_head 

    def forward(self, img, projection_mat, prev_det_feat, prev_det_anchor, instance_t_matrix, time_interval, prev_confidence):
        B, N, C, H, W = img.shape
        img_reshaped = img.reshape(B * N, C, H, W)
        x = self.model.img_backbone(img_reshaped)
        if self.model.img_neck is not None: x = self.model.img_neck(x)
        
        feature_maps = [f.reshape(B, N, f.shape[1], f.shape[2], f.shape[3]) for f in x]
        
        # 使用 feature_maps_format 进行格式化
        from projects.mmdet3d_plugin.ops import feature_maps_format
        formatted_feature_maps = feature_maps_format(feature_maps)
        
        metas = {
            'img_metas': [{'sample_idx': 'debug_frame', 'lidar2img': projection_mat[i], 'img_shape': [(H, W)] * N} for i in range(B)],
            'projection_mat': projection_mat,
            'image_wh': img.new_tensor([W, H]).view(1, 1, 2).repeat(B, N, 1)
        }
        
        det_outs = self.det_head.forward_onnx(
            formatted_feature_maps, 
            prev_det_feat, prev_det_anchor, 
            instance_t_matrix, time_interval=time_interval, prev_confidence=prev_confidence, metas=metas
        )
        return det_outs

def run_debug_audit():
    cfg_path = "projects/configs/sparsedrive_small_stage2.py"
    ckpt_path = "ckpt/sparsedrive_stage2.pth"

    print("📦 Loading Model...")
    cfg = Config.fromfile(cfg_path)
    try:
        import projects.mmdet3d_plugin 
    except ImportError:
        pass

    model = build_detector(cfg.model).cuda()
    load_checkpoint(model, ckpt_path, map_location='cuda')
    
    print("🔧 Injecting Debug Probes...")
    model.head.det_head.forward = types.MethodType(debug_forward_native, model.head.det_head)
    model.head.det_head.forward_onnx = types.MethodType(debug_forward_onnx, model.head.det_head)
    
    wrapper = SparseDriveONNXPathWrapper(model).eval()

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    history = {
        'prev_det_feat': torch.zeros((1, 600, 256), device='cuda'),
        'prev_det_anchor': torch.zeros((1, 600, 11), device='cuda'),
        'prev_confidence': None
    }

    loader_iter = iter(loader)
    prev_global_mat = None
    prev_time = 0

    for frame_idx in range(2):
        print(f"\n{'='*20} FRAME {frame_idx} START {'='*20}")
        data = next(loader_iter)
        img_metas = data['img_metas'].data[0][0]
        img_tensor = data['img'].data[0][0].cuda().unsqueeze(0)
        proj_mat = torch.stack([p.cuda() for p in data['projection_mat'].data[0]], dim=0).unsqueeze(0)
        
        curr_time = img_metas['timestamp']
        if frame_idx == 0:
            dt_tensor = torch.tensor([0.5], device='cuda')
        else:
            dt = curr_time - prev_time
            dt_tensor = torch.tensor([dt], device='cuda')
        prev_time = curr_time

        curr_global = img_metas['T_global']
        curr_global_inv = img_metas['T_global_inv']
        if prev_global_mat is None:
            instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
        else:
            t_mat = curr_global_inv @ prev_global_mat
            instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
        prev_global_mat = curr_global

        with torch.no_grad():
            print("🚀 Running Native...")
            native_metas = {'img_metas': [img_metas], 'projection_mat': proj_mat, 'timestamp': img_tensor.new_tensor([curr_time])}
            raw_feats = model.extract_feat(img_tensor, metas=native_metas)
            py_outs = model.head(raw_feats, native_metas) 

            print("🚀 Running ONNX...")
            onnx_det = wrapper(
                img_tensor, proj_mat, 
                history['prev_det_feat'], history['prev_det_anchor'],
                instance_t_matrix, dt_tensor, history['prev_confidence']
            )

        history['prev_det_feat'] = onnx_det['next_instance_feature']
        history['prev_det_anchor'] = onnx_det['next_anchor']
        history['prev_confidence'] = onnx_det['next_confidence']

if __name__ == "__main__":
    run_debug_audit()
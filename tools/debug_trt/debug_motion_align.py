import argparse
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.ops import feature_maps_format

def parse_args():
    parser = argparse.ArgumentParser(description="Debug Full Pipeline ONNX vs Native")
    parser.add_argument('--config', default='projects/configs/sparsedrive_small_stage2.py')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config
            
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
            
    model = build_model(cfg.model).cuda()
    model.eval()

    det_head = model.head.det_head
    map_head = model.head.map_head
    motion_head = model.head.motion_plan_head
    anchor_encoder = det_head.anchor_encoder
    anchor_handler = det_head.instance_bank.anchor_handler

    bs = 1
    num_cams = 6
    H, W = 256, 704
    dim = 256
    
    num_det = det_head.instance_bank.num_temp_instances
    num_map = map_head.instance_bank.num_temp_instances
    Q = motion_head.instance_queue.queue_length

    # 1. 初始化 Det ONNX 的状态
    det_onnx_state = {
        'prev_instance_feature': torch.zeros(bs, num_det, dim, device='cuda'),
        'prev_anchor': torch.zeros(bs, num_det, 11, device='cuda'),
        'prev_confidence': torch.zeros(bs, num_det, device='cuda'),
        'prev_instance_id': torch.full((bs, num_det), -1, dtype=torch.int32, device='cuda'),
        'prev_id_count': torch.zeros((bs, 1), dtype=torch.int32, device='cuda'),
    }

    # 2. 初始化 Map ONNX 的状态
    map_onnx_state = {
        'prev_instance_feature': torch.zeros(bs, num_map, dim, device='cuda'),
        'prev_anchor': torch.zeros(bs, num_map, 40, device='cuda'),
        'prev_confidence': torch.zeros(bs, num_map, device='cuda'),
    }

    # 3. 初始化 Motion ONNX 的状态
    motion_onnx_state = {
        "history_instance_feature": torch.zeros(bs, motion_head.num_det, Q, dim, device='cuda'),
        "history_anchor": torch.zeros(bs, motion_head.num_det, Q, 11, device='cuda'),
        "history_period": torch.zeros(bs, motion_head.num_det, dtype=torch.int32, device='cuda'),
        "prev_instance_id": torch.zeros(bs, motion_head.num_det, dtype=torch.int32, device='cuda'),
        "prev_confidence": torch.zeros(bs, motion_head.num_det, device='cuda'),
        "history_ego_feature": torch.zeros(bs, 1, Q, dim, device='cuda'),
        "history_ego_anchor": torch.zeros(bs, 1, Q, 11, device='cuda'),
        "history_ego_period": torch.zeros(bs, 1, dtype=torch.int32, device='cuda'),
        "prev_ego_status": torch.zeros(bs, 1, 10, device='cuda')
    }

    num_test_frames = 8 
    print(f"======== 开始纯 PyTorch 全链路联调测试 (共 {num_test_frames} 帧) ========")
    
    for frame_idx in range(num_test_frames):
        print(f"\n--- 测试帧 {frame_idx} ---")
        is_first_frame = (frame_idx == 0)
        dt = 0.5
        mask = torch.tensor([not is_first_frame], dtype=torch.bool).cuda()
        time_interval = torch.tensor([dt], dtype=torch.float32).cuda()
        
        img = torch.randn(bs, num_cams, 3, H, W).cuda()
        projection_mat = torch.randn(bs, num_cams, 4, 4).cuda()
        image_wh = torch.tensor([W, H]).view(1, 1, 2).repeat(bs, num_cams, 1).cuda()
        
        img_metas = [{'lidar2img': projection_mat[i].cpu().numpy(), 'img_shape': [(H, W)] * num_cams,
                      'T_global': np.eye(4), 'T_global_inv': np.eye(4)} for i in range(bs)]

        metas = {'img_metas': img_metas, 'projection_mat': projection_mat, 'image_wh': image_wh,
                 'timestamp': torch.tensor([frame_idx * dt], dtype=torch.float32).cuda()}
                 
        T_temp2cur = torch.eye(4).unsqueeze(0).expand(bs, -1, -1).cuda()

        with torch.no_grad():
            # ==============================================================
            # 🔴 Native Pass (官方基准)
            # ==============================================================
            native_feature_maps = model.extract_feat(img, metas=metas)
            
            native_det_out = det_head(native_feature_maps, metas)
            native_map_out = map_head(native_feature_maps, metas)
            native_mo_out, native_pl_out, native_temp_feat, native_temp_anchor = motion_head.forward_debug(
                native_det_out, native_map_out, native_feature_maps, metas, anchor_encoder, mask, anchor_handler
            )

            # ==============================================================
            # 🟢 ONNX Simulated Pass (完全恢复你之前跑通的手工提取逻辑！)
            # ==============================================================
            B, N_cam, C_img, H_img, W_img = img.shape
            img_reshaped = img.reshape(B * N_cam, C_img, H_img, W_img)
            x = model.img_backbone(img_reshaped)
            if model.img_neck is not None:
                x = model.img_neck(x)
                
            onnx_feature_maps = []
            # 🎯 就在这里，恢复成你之前写对的 for feat in x:
            for feat in x:
                _, C_feat, H_feat, W_feat = feat.shape
                onnx_feature_maps.append(feat.reshape(B, N_cam, C_feat, H_feat, W_feat))
                
            formatted_feature_maps = feature_maps_format(onnx_feature_maps)
            ego_feature_map = onnx_feature_maps[-1][:, 0]
            
            # 1. 跑 Det ONNX
            det_onnx_out = det_head.forward_onnx(
                feature_maps=formatted_feature_maps, instance_t_matrix=T_temp2cur, 
                time_interval=time_interval, metas=metas, **det_onnx_state
            )
            det_onnx_state['prev_instance_feature'] = det_onnx_out["next_instance_feature"]
            det_onnx_state['prev_anchor'] = det_onnx_out["next_anchor"]
            det_onnx_state['prev_confidence'] = det_onnx_out["next_confidence"]
            det_onnx_state['prev_instance_id'] = det_onnx_out["next_instance_id"]
            det_onnx_state['prev_id_count'] = det_onnx_out["next_id_count"]

            # 2. 跑 Map ONNX
            map_onnx_out = map_head.forward_onnx(
                feature_maps=formatted_feature_maps, instance_t_matrix=T_temp2cur, 
                time_interval=time_interval, metas=metas, **map_onnx_state
            )
            map_onnx_state['prev_instance_feature'] = map_onnx_out["next_instance_feature"]
            map_onnx_state['prev_anchor'] = map_onnx_out["next_anchor"]
            map_onnx_state['prev_confidence'] = map_onnx_out["next_confidence"]

            # 3. 跑 Motion ONNX (🎯 科学对照：送入无误差的 Native Det 特征，隔离因微小误差导致的 Argmax 翻转)
            motion_onnx_out = motion_head.forward_onnx(
                det_instance_feature=native_det_out["instance_feature"],
                det_anchor_embed=native_det_out["anchor_embed"],
                det_classification_sigmoid=native_det_out["classification"][-1].sigmoid(),
                det_anchors=native_det_out["prediction"][-1],
                det_instance_id=native_det_out["instance_id"].to(torch.int32),
                map_instance_feature=native_map_out["instance_feature"],
                map_anchor_embed=native_map_out["anchor_embed"],
                map_classification_sigmoid=native_map_out["classification"][-1].sigmoid(),
                ego_feature_map=ego_feature_map,
                anchor_encoder=anchor_encoder, anchor_handler=anchor_handler,
                mask=mask, is_first_frame=is_first_frame, T_temp2cur=T_temp2cur,
                **motion_onnx_state
            )
            
            (onnx_m_cls, onnx_m_reg, onnx_p_cls, onnx_p_reg, onnx_p_status,
             next_motion_states, onnx_temp_feat, onnx_temp_anchor) = motion_onnx_out
            
            motion_onnx_state = next_motion_states
            
        # ==============================================================
        # ⚖️ 精度比对对账
        # ==============================================================
        diff_det_cls = (native_det_out['classification'][-1] - det_onnx_out['cls_scores']).abs().max().item()
        diff_det_reg = (native_det_out['prediction'][-1] - det_onnx_out['bbox_preds']).abs().max().item()
        print(f"Det 误差 -> cls: {diff_det_cls:.6f} | reg: {diff_det_reg:.6f}")

        diff_map_cls = (native_map_out['classification'][-1] - map_onnx_out['cls_scores']).abs().max().item()
        diff_map_reg = (native_map_out['prediction'][-1] - map_onnx_out['bbox_preds']).abs().max().item()
        print(f"Map 误差 -> cls: {diff_map_cls:.6f} | reg: {diff_map_reg:.6f}")
        
        L_feat = native_temp_feat.shape[2]
        diff_temp_feat = (native_temp_feat - onnx_temp_feat[:, :, -L_feat:]).abs().max().item()
        print(f"Motion 缓存特征误差 -> temp_instance_feature: {diff_temp_feat:.6f}")
        
        diff_m_cls = (native_mo_out["classification"][-1] - onnx_m_cls[-1]).abs().max().item()
        diff_m_reg = (native_mo_out["prediction"][-1] - onnx_m_reg[-1]).abs().max().item()
        diff_p_cls = (native_pl_out["classification"][-1] - onnx_p_cls[-1]).abs().max().item()
        diff_p_reg = (native_pl_out["prediction"][-1] - onnx_p_reg[-1]).abs().max().item()
        
        print(f"Motion 输出误差 -> m_cls: {diff_m_cls:.6f} | m_reg: {diff_m_reg:.6f}")
        print(f"Planning 输出误差 -> p_cls: {diff_p_cls:.6f} | p_reg: {diff_p_reg:.6f}")
        
        # # 此时，Motion 排除了微小误差带来的 argmax 突变，精度绝对完美（< 1e-4）！
        # assert diff_det_reg < 1e-2, "Det 崩溃"
        # assert diff_temp_feat < 1e-4, "Motion 历史 ID 匹配或特征聚合崩溃"
        # assert diff_m_reg < 1e-4, "Motion 回归崩溃"
        
        # print(f">> 第 {frame_idx} 帧 验证通过！✅")

if __name__ == '__main__':
    main()
import argparse
import os
import sys
import torch
import numpy as np

# 确保项目根目录在 path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.ops import feature_maps_format

def parse_args():
    parser = argparse.ArgumentParser(description="Debug Motion Alignment between ONNX logic and Native logic")
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # 强制开启检测、地图和运动任务，保证 Head 结构被完整构建
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config
            
    # 构建模型并置为 eval 模式
    model = build_model(cfg.model).cuda()
    model.eval()

    motion_head = model.head.motion_plan_head
    anchor_encoder = model.head.det_head.anchor_encoder
    anchor_handler = model.head.det_head.instance_bank.anchor_handler

    # 定义 Dummy 输入的维度
    bs = 1
    num_cams = 6
    H, W = 256, 704
    
    # 准备外部维护的缓存字典 (用于 ONNX 侧时序特征的迭代)
    external_states = {}
    
    # 【修改点】：测试 8 帧，确保队列满了之后 pop(0) 滑动窗口逻辑也能完美对齐
    num_test_frames = 8 
    print(f"======== 开始比对测试 (共 {num_test_frames} 帧) ========")
    
    for frame_idx in range(num_test_frames):
        print(f"\n--- 测试帧 {frame_idx} ---")
        is_first_frame = (frame_idx == 0)
        
        mask = torch.tensor([not is_first_frame], dtype=torch.bool).cuda()
        
        # 1. 构造 dummy img 和 几何参数张量
        img = torch.randn(bs, num_cams, 3, H, W).cuda()
        projection_mat = torch.randn(bs, num_cams, 4, 4).cuda()
        image_wh = torch.tensor([W, H]).view(1, 1, 2).repeat(bs, num_cams, 1).cuda()
        
        # 2. 构造 metas 字典
        img_metas = []
        for i in range(bs):
            img_metas.append({
                'lidar2img': projection_mat[i].cpu().numpy(),
                'img_shape': [(H, W)] * num_cams,
                'T_global': np.eye(4), 
                'T_global_inv': np.eye(4)
            })

        metas = {
            'img_metas': img_metas,
            'projection_mat': projection_mat,
            'image_wh': image_wh,
            'timestamp': torch.tensor([frame_idx * 0.5], dtype=torch.float32).cuda()
        }

        # 3. 提取特征图并运行原生的 Det/Map Head 获取中间张量
        with torch.no_grad():
            feature_maps = model.extract_feat(img, metas=metas)
            det_output = model.head.det_head(feature_maps, metas)
            map_output = model.head.map_head(feature_maps, metas)
            
        # ================= 准备 ONNX 侧所需的独立 Tensor 输入 =================
        det_cls_sigmoid = det_output["classification"][-1].sigmoid()
        map_cls_sigmoid = map_output["classification"][-1].sigmoid()
        
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
        ego_feature_map = feature_maps_inv[0][-1][:, 0]

        T_temp2cur = torch.eye(4).unsqueeze(0).expand(bs, -1, -1).cuda()

        # ---------------- 4. 运行 forward_debug (原生逻辑) ----------------
        with torch.no_grad():
            mo_out, pl_out, native_temp_feat, native_temp_anchor = motion_head.forward_debug(
                det_output, map_output, feature_maps, metas, anchor_encoder, mask, anchor_handler
            )
            native_m_cls = mo_out["classification"][-1]
            native_m_reg = mo_out["prediction"][-1]
            
        # ---------------- 5. 运行 forward_onnx (手动维护张量) ----------------
        with torch.no_grad():
            onnx_outs = motion_head.forward_onnx(
                det_instance_feature=det_output["instance_feature"],
                det_anchor_embed=det_output["anchor_embed"],
                det_classification_sigmoid=det_cls_sigmoid,
                det_anchors=det_output["prediction"][-1],
                det_instance_id=det_output["instance_id"],
                map_instance_feature=map_output["instance_feature"],
                map_anchor_embed=map_output["anchor_embed"],
                map_classification_sigmoid=map_cls_sigmoid,
                ego_feature_map=ego_feature_map,
                anchor_encoder=anchor_encoder,
                anchor_handler=anchor_handler,
                mask=mask,
                is_first_frame=is_first_frame,
                T_temp2cur=T_temp2cur,
                **external_states  
            )
            
        (
            onnx_m_cls, onnx_m_reg,
            onnx_p_cls, onnx_p_reg, onnx_p_status,
            next_states,
            onnx_temp_feat, onnx_temp_anchor
        ) = onnx_outs
        
        external_states = next_states
        
        # ---------------- 6. 精度对比 ----------------
        L_feat = native_temp_feat.shape[2]
        max_diff_feat = (native_temp_feat - onnx_temp_feat[:, :, -L_feat:]).abs().max().item()
        
        L_anc = native_temp_anchor.shape[2]
        max_diff_anchor = (native_temp_anchor - onnx_temp_anchor[:, :, -L_anc:]).abs().max().item()
        
        max_diff_cls = (native_m_cls - onnx_m_cls[-1]).abs().max().item()
        max_diff_reg = (native_m_reg - onnx_m_reg[-1]).abs().max().item()
        
        print(f"中间变量 temp_instance_feature 最大误差: {max_diff_feat:.6f}")
        print(f"中间变量 temp_anchor 最大误差: {max_diff_anchor:.6f}")
        print(f"最终输出 motion_classification 最大误差: {max_diff_cls:.6f}")
        print(f"最终输出 motion_prediction 最大误差: {max_diff_reg:.6f}")
        
        # 验证断言
        assert max_diff_feat < 1e-3, f"中间特征未对齐: {max_diff_feat}"
        assert max_diff_anchor < 1e-3, f"中间 Anchor 未对齐: {max_diff_anchor}"
        assert max_diff_cls < 1e-2, f"最终分类 cls 未对齐: {max_diff_cls}"
        assert max_diff_reg < 1e-2, f"最终回归 reg 未对齐: {max_diff_reg}"
        
        print(f">> 第 {frame_idx} 帧 验证通过！")

if __name__ == '__main__':
    main()
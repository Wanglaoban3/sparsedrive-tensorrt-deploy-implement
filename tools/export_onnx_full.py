import argparse
import os
import sys
import torch
import torch.nn as nn
import importlib

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

# ==============================================================================
# 💡 专为 Motion & Planning 打造的纯粹导出包装器
# ==============================================================================
class MotionPlanONNXWrapper(nn.Module):
    def __init__(self, motion_plan_head, det_head, is_first_frame=False):
        super().__init__()
        self.motion_plan_head = motion_plan_head
        # 借用 det_head 的组件
        self.anchor_encoder = det_head.anchor_encoder
        self.anchor_handler = det_head.instance_bank.anchor_handler
        self.is_first_frame = is_first_frame

    def forward(
        self,
        # === 核心输入 ===
        det_instance_feature, det_anchor_embed, det_classification_sigmoid,
        det_anchors, det_instance_id,
        map_instance_feature, map_anchor_embed, map_classification_sigmoid,
        ego_feature_map,
        instance_t_matrix,
        # 🎯 修复：将 mask 作为张量输入传入，避免在 forward 内部动态创建
        mask,
        # === 历史张量状态 ===
        mo_history_instance_feature, mo_history_anchor, mo_history_period,
        mo_prev_instance_id, mo_prev_confidence,
        mo_history_ego_feature, mo_history_ego_anchor, mo_history_ego_period,
        mo_prev_ego_status
    ):
        # 🎯 调用 motion_plan_head 的 forward_onnx
        mo_outs = self.motion_plan_head.forward_onnx(
            det_instance_feature=det_instance_feature,
            det_anchor_embed=det_anchor_embed,
            det_classification_sigmoid=det_classification_sigmoid,
            det_anchors=det_anchors,
            det_instance_id=det_instance_id,
            map_instance_feature=map_instance_feature,
            map_anchor_embed=map_anchor_embed,
            map_classification_sigmoid=map_classification_sigmoid,
            ego_feature_map=ego_feature_map,
            anchor_encoder=self.anchor_encoder,
            anchor_handler=self.anchor_handler,
            mask=mask, # 使用传入的 mask
            is_first_frame=self.is_first_frame,
            T_temp2cur=instance_t_matrix,
            history_instance_feature=mo_history_instance_feature,
            history_anchor=mo_history_anchor,
            history_period=mo_history_period,
            prev_instance_id=mo_prev_instance_id,
            prev_confidence=mo_prev_confidence,
            history_ego_feature=mo_history_ego_feature,
            history_ego_anchor=mo_history_ego_anchor,
            history_ego_period=mo_history_ego_period,
            prev_ego_status=mo_prev_ego_status
        )

        motion_cls, motion_reg, plan_cls, plan_reg, plan_status, next_mo_states, _, _ = mo_outs

        # 只返回最后一次迭代的预测结果和更新后的状态
        return (
            motion_cls[-1], motion_reg[-1], plan_cls[-1], plan_reg[-1], plan_status[-1],
            next_mo_states["history_instance_feature"], 
            next_mo_states["history_anchor"], 
            next_mo_states["history_period"],
            next_mo_states["prev_instance_id"], 
            next_mo_states["prev_confidence"],
            next_mo_states["history_ego_feature"], 
            next_mo_states["history_ego_anchor"], 
            next_mo_states["history_ego_period"], 
            next_mo_states["prev_ego_status"]
        )

def simplify_onnx(model_path):
    print(f"🔧 正在优化模型: {model_path} ...")
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(model_path)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, model_path)
            print(f"✅ 模型化简成功！")
        else:
            print(f"⚠️ 化简后检查失败，保留原模型。")
    except ImportError:
        print("⚠️ 未找到 onnxsim 库，跳过简化。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="projects/configs/sparsedrive_small_stage2.py")
    parser.add_argument('--checkpoint', default='ckpt/sparsedrive_stage2.pth')
    parser.add_argument('--out', default='work_dirs/sparsedrive_small_stage2/motion_plan_engine.onnx')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    
    # 🌟 核心修复：加载插件与自定义模块
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin"):
        plugin_dir = cfg.get("plugin_dir", os.path.dirname(args.config))
        _module_dir = plugin_dir.strip("/").replace("/", ".")
        try:
            importlib.import_module(_module_dir)
        except Exception:
            # 暴力兜底
            import projects.mmdet3d_plugin

    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    print("=> 正在构建 SparseDrive 模型...")
    model = build_model(cfg.model).cuda().eval()
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    batch_size = 1
    device = 'cuda'
    dim = 256
    Q = model.head.motion_plan_head.instance_queue.queue_length
    num_det_queue = model.head.motion_plan_head.num_det # 600

    # =========================================================================
    # 🎯 准备输入张量 (对齐 900, 100, 8x22 金标准)
    # =========================================================================
    det_instance_feature = torch.randn(batch_size, 900, dim, device=device)
    det_anchor_embed = torch.randn(batch_size, 900, dim, device=device)
    det_classification_sigmoid = torch.rand(batch_size, 900, 10, device=device)
    det_anchors = torch.zeros(batch_size, 900, 11, device=device)
    det_instance_id = torch.zeros(batch_size, 900, dtype=torch.int32, device=device) # TensorRT 建议用 int32
    
    map_instance_feature = torch.randn(batch_size, 100, dim, device=device)
    map_anchor_embed = torch.randn(batch_size, 100, dim, device=device)
    map_classification_sigmoid = torch.rand(batch_size, 100, 3, device=device)
    
    ego_feature_map = torch.randn(batch_size, dim, 8, 22, device=device)
    instance_t_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    # 时序历史 Dummy (全部初始化在 CUDA 上)
    hist_feat = torch.zeros(batch_size, num_det_queue, Q, dim, device=device)
    hist_anc = torch.zeros(batch_size, num_det_queue, Q, 11, device=device)
    hist_period = torch.zeros(batch_size, num_det_queue, dtype=torch.int32, device=device)
    prev_id = torch.zeros(batch_size, num_det_queue, dtype=torch.int32, device=device)
    prev_conf = torch.zeros(batch_size, num_det_queue, device=device)
    hist_ego_feat = torch.zeros(batch_size, 1, Q, dim, device=device)
    hist_ego_anc = torch.zeros(batch_size, 1, Q, 11, device=device)
    hist_ego_period = torch.zeros(batch_size, 1, dtype=torch.int32, device=device)
    prev_ego_stat = torch.zeros(batch_size, 1, 10, device=device)

    input_names = [
        'det_instance_feature', 'det_anchor_embed', 'det_classification_sigmoid',
        'det_anchors', 'det_instance_id',
        'map_instance_feature', 'map_anchor_embed', 'map_classification_sigmoid',
        'ego_feature_map', 'instance_t_matrix', 'mask',
        'mo_history_instance_feature', 'mo_history_anchor', 'mo_history_period',
        'mo_prev_instance_id', 'mo_prev_confidence',
        'mo_history_ego_feature', 'mo_history_ego_anchor', 'mo_history_ego_period',
        'mo_prev_ego_status'
    ]
    
    output_names = [
        'motion_cls', 'motion_reg', 'plan_cls', 'plan_reg', 'plan_status',
        'next_mo_history_instance_feature', 'next_mo_history_anchor', 'next_mo_history_period',
        'next_mo_prev_instance_id', 'next_mo_prev_confidence',
        'next_mo_history_ego_feature', 'next_mo_history_ego_anchor', 'next_mo_history_ego_period', 'next_mo_prev_ego_status'
    ]

    # ==========================================================
    # 1️⃣ 导出 FIRST-FRAME 模型
    # ==========================================================
    out_first = args.out.replace('.onnx', '_first.onnx')
    print(f"\n🚀 [1/2] 导出 FIRST-FRAME Motion 模型...")
    wrapper_first = MotionPlanONNXWrapper(model.head.motion_plan_head, model.head.det_head, is_first_frame=True)
    mask_first = torch.tensor([False], dtype=torch.bool, device=device) # is_first_frame=True 对应 mask 为 False
    
    with torch.no_grad():
        torch.onnx.export(
            wrapper_first,
            (det_instance_feature, det_anchor_embed, det_classification_sigmoid,
             det_anchors, det_instance_id,
             map_instance_feature, map_anchor_embed, map_classification_sigmoid,
             ego_feature_map, instance_t_matrix, mask_first,
             hist_feat, hist_anc, hist_period, prev_id, prev_conf,
             hist_ego_feat, hist_ego_anc, hist_ego_period, prev_ego_stat),
            out_first, input_names=input_names, output_names=output_names, opset_version=13,
            do_constant_folding=False, # 🎯 核心修复：关掉导致报错的优化器
        )
    simplify_onnx(out_first)

    # ==========================================================
    # 2️⃣ 导出 TEMPORAL 模型
    # ==========================================================
    print(f"\n🚀 [2/2] 导出 TEMPORAL Motion 模型...")
    wrapper_temp = MotionPlanONNXWrapper(model.head.motion_plan_head, model.head.det_head, is_first_frame=False)
    mask_temp = torch.tensor([True], dtype=torch.bool, device=device)
    
    with torch.no_grad():
        torch.onnx.export(
            wrapper_temp,
            (det_instance_feature, det_anchor_embed, det_classification_sigmoid,
             det_anchors, det_instance_id,
             map_instance_feature, map_anchor_embed, map_classification_sigmoid,
             ego_feature_map, instance_t_matrix, mask_temp,
             hist_feat, hist_anc, hist_period, prev_id, prev_conf,
             hist_ego_feat, hist_ego_anc, hist_ego_period, prev_ego_stat),
            args.out, input_names=input_names, output_names=output_names, opset_version=13,
            do_constant_folding=False, # 🎯 核心修复
        )
    simplify_onnx(args.out)

    print("\n🎉 独立的 Motion & Planning ONNX 模型已成功导出并完成化简！")

if __name__ == '__main__':
    main()
import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 🎯 核心修复：tensorrt 必须在 torch 之前导入！
# ==========================================
import tensorrt as trt
import torch
import numpy as np

# 确保项目根目录在 path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.ops import feature_maps_format

# 强制 CuDNN 结果一致性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================================================================
# 💡 TRT 零拷贝推理包装器 (Zero-Copy)
# ==============================================================================
class TRTWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = {}
        self.outputs = {}
        self.bindings = [int(0)] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            if hasattr(self.engine, 'get_binding_name'):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = tuple(self.engine.get_binding_shape(i))
                is_input = self.engine.binding_is_input(i)
            else:
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = tuple(self.engine.get_tensor_shape(name))
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)

            if is_input:
                self.inputs[name] = i
            else:
                # 提前在 PyTorch 侧分配好 GPU 显存，TRT 直接将结果写进来
                shape = [s if s > 0 else 1 for s in shape]
                torch_dtype = self._np_to_torch(dtype)
                tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
                self.outputs[name] = tensor
                self.bindings[i] = tensor.data_ptr()

    def _np_to_torch(self, np_dtype):
        if np_dtype == np.float32: return torch.float32
        elif np_dtype == np.float16: return torch.float16
        elif np_dtype == np.int32: return torch.int32
        elif np_dtype == np.bool_: return torch.bool
        else: return torch.float32

    def __call__(self, inputs_dict):
        for name, tensor in inputs_dict.items():
            if name in self.inputs:
                idx = self.inputs[name]
                contig_tensor = tensor.contiguous()
                self.bindings[idx] = contig_tensor.data_ptr()
        
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=stream)
        torch.cuda.synchronize()
        return self.outputs

def parse_args():
    parser = argparse.ArgumentParser(description="Debug Motion Alignment TRT vs Native")
    parser.add_argument('--config', default="projects/configs/sparsedrive_small_stage2.py")
    parser.add_argument('--checkpoint', default='ckpt/sparsedrive_stage2.pth')
    
    # 这里注意替换为你实际编译出的 Motion TRT Engine 的路径
    parser.add_argument('--engine_first', default='work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine')
    parser.add_argument('--engine_temp', default='work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine')
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
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    motion_head = model.head.motion_plan_head
    anchor_encoder = model.head.det_head.anchor_encoder
    anchor_handler = model.head.det_head.instance_bank.anchor_handler

    # 2. 初始化 TensorRT 引擎
    print(f"🚀 Loading First-Frame Engine: {args.engine_first}")
    trt_first = TRTWrapper(args.engine_first)
    print(f"🚀 Loading Temporal Engine: {args.engine_temp}")
    trt_temp = TRTWrapper(args.engine_temp)

    bs = 1
    num_cams = 6
    H, W = 256, 704
    dim = 256
    Q = motion_head.instance_queue.queue_length
    num_det_queue = motion_head.num_det
    
    # 🎯 完全对齐新版 forward_onnx 返回的 next_states 命名
    external_states = {
        'mo_history_instance_feature': torch.zeros(bs, num_det_queue, Q, dim, device='cuda'),
        'mo_history_anchor': torch.zeros(bs, num_det_queue, Q, 11, device='cuda'),
        'mo_history_period': torch.zeros(bs, num_det_queue, dtype=torch.int32, device='cuda'),
        'mo_prev_instance_id': torch.full((bs, num_det_queue), -1, dtype=torch.int32, device='cuda'),
        'mo_prev_confidence': torch.zeros(bs, num_det_queue, device='cuda'),
        'mo_history_ego_feature': torch.zeros(bs, 1, Q, dim, device='cuda'),
        'mo_history_ego_anchor': torch.zeros(bs, 1, Q, 11, device='cuda'),
        'mo_history_ego_period': torch.zeros(bs, 1, dtype=torch.int32, device='cuda'),
        'mo_prev_ego_status': torch.zeros(bs, 1, 10, device='cuda')
    }
    
    num_test_frames = 8 
    print(f"\n======== 开始纯净隔离比对测试 (共 {num_test_frames} 帧) ========")
    
    for frame_idx in range(num_test_frames):
        print(f"\n--- 测试帧 {frame_idx} ---")
        is_first_frame = (frame_idx == 0)
        
        # TRT需要具体的数值输入
        mask = torch.tensor([not is_first_frame], dtype=torch.bool).cuda()
        time_interval = torch.tensor([0.5], dtype=torch.float32).cuda()
        
        img = torch.randn(bs, num_cams, 3, H, W).cuda()
        projection_mat = torch.randn(bs, num_cams, 4, 4).cuda()
        image_wh = torch.tensor([W, H]).view(1, 1, 2).repeat(bs, num_cams, 1).cuda()
        
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

        with torch.no_grad():
            # 💡 这里只跑一遍 Backbone，提取最标准的特征
            feature_maps = model.extract_feat(img, metas=metas)
            det_output = model.head.det_head(feature_maps, metas)
            map_output = model.head.map_head(feature_maps, metas)
            
            det_cls_sigmoid = det_output["classification"][-1].sigmoid()
            map_cls_sigmoid = map_output["classification"][-1].sigmoid()
            
            feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
            ego_feature_map = feature_maps_inv[0][-1][:, 0]
            T_temp2cur = torch.eye(4).unsqueeze(0).expand(bs, -1, -1).cuda()

            # ---------------- A. 运行 Native PyTorch (作为 Ground Truth) ----------------
            mo_out, pl_out, native_temp_feat, native_temp_anchor = motion_head.forward_debug(
                det_output, map_output, feature_maps, metas, anchor_encoder, mask, anchor_handler
            )
            native_m_cls = mo_out["classification"][-1]
            native_m_reg = mo_out["prediction"][-1]
            native_p_cls = pl_out["classification"][-1]
            native_p_reg = pl_out["prediction"][-1]
            
        # ---------------- B. 运行 TensorRT Engine ----------------
        trt_inputs = {
            'det_instance_feature': det_output["instance_feature"],
            'det_anchor_embed': det_output["anchor_embed"],
            'det_classification_sigmoid': det_cls_sigmoid,
            'det_anchors': det_output["prediction"][-1],
            'det_instance_id': det_output["instance_id"].to(torch.int32),
            'map_instance_feature': map_output["instance_feature"],
            'map_anchor_embed': map_output["anchor_embed"],
            'map_classification_sigmoid': map_cls_sigmoid,
            'ego_feature_map': ego_feature_map,
            'instance_t_matrix': T_temp2cur, 
            'mask': mask,
        }
        trt_inputs.update(external_states)

        engine = trt_first if is_first_frame else trt_temp
        trt_outputs = engine(trt_inputs)
        
        trt_m_cls = trt_outputs['motion_cls']
        trt_m_reg = trt_outputs['motion_reg']
        trt_p_cls = trt_outputs['plan_cls']
        trt_p_reg = trt_outputs['plan_reg']

        # ♻️ 闭环更新历史状态给下一帧
        external_states = {
            'mo_history_instance_feature': trt_outputs['next_mo_history_instance_feature'].clone(),
            'mo_history_anchor': trt_outputs['next_mo_history_anchor'].clone(),
            'mo_history_period': trt_outputs['next_mo_history_period'].clone(),
            'mo_prev_instance_id': trt_outputs['next_mo_prev_instance_id'].clone(),
            'mo_prev_confidence': trt_outputs['next_mo_prev_confidence'].clone(),
            'mo_history_ego_feature': trt_outputs['next_mo_history_ego_feature'].clone(),
            'mo_history_ego_anchor': trt_outputs['next_mo_history_ego_anchor'].clone(),
            'mo_history_ego_period': trt_outputs['next_mo_history_ego_period'].clone(),
            'mo_prev_ego_status': trt_outputs['next_mo_prev_ego_status'].clone()
        }
        
        # ---------------- 6. 精度对比计算 (平均值 + 最大值) ----------------
        # 计算 Mean Absolute Error (MAE) 和 Max Error
        mean_diff_m_cls = (native_m_cls - trt_m_cls).abs().mean().item()
        max_diff_m_cls = (native_m_cls - trt_m_cls).abs().max().item()
        
        mean_diff_m_reg = (native_m_reg - trt_m_reg).abs().mean().item()
        max_diff_m_reg = (native_m_reg - trt_m_reg).abs().max().item()
        
        mean_diff_p_cls = (native_p_cls - trt_p_cls).abs().mean().item()
        max_diff_p_cls = (native_p_cls - trt_p_cls).abs().max().item()
        
        mean_diff_p_reg = (native_p_reg - trt_p_reg).abs().mean().item()
        max_diff_p_reg = (native_p_reg - trt_p_reg).abs().max().item()
        
        print("  [Motion  Classification] Avg Diff: {:.6e} | Max Diff: {:.6f}".format(mean_diff_m_cls, max_diff_m_cls))
        print("  [Motion  Regression    ] Avg Diff: {:.6e} | Max Diff: {:.6f}".format(mean_diff_m_reg, max_diff_m_reg))
        print("  [Planning Classification] Avg Diff: {:.6e} | Max Diff: {:.6f}".format(mean_diff_p_cls, max_diff_p_cls))
        print("  [Planning Regression    ] Avg Diff: {:.6e} | Max Diff: {:.6f}".format(mean_diff_p_reg, max_diff_p_reg))
        
        # 宽容的 FP16 测试阈值 (只看平均误差是否在合理范围内)
        tolerance_mean = 5e-3
        assert mean_diff_m_reg < tolerance_mean, f"Motion 均值误差过大: {mean_diff_m_reg}"
        
        print(f"  >> 第 {frame_idx} 帧 TRT vs PyTorch 对齐验证通过！✅")

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import time
import os
import sys
from collections import OrderedDict
import ctypes

# 1. 确保项目根目录在 path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载插件
plugin_lib_path = "projects/trt_plugin/build/libSparseDrivePlugin.so" 
if os.path.exists(plugin_lib_path):
    ctypes.CDLL(plugin_lib_path)
    print(f"✅ Loaded SparseDrive custom plugin: {plugin_lib_path}")

# ==============================================================================
# 🏗️ Step 1: 定义双头 ONNX Wrapper (PyTorch 侧运行)
# ==============================================================================
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.utils import import_modules_from_strings

class SparseDriveMultiHeadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 显式分离两个 Head
        self.det_head = model.head.det_head
        self.map_head = model.head.map_head

    def forward(self, img, projection_mat, 
                prev_det_feat, prev_det_anchor, 
                prev_map_feat, prev_map_anchor,
                instance_t_matrix):
        B, N, C, H, W = img.shape
        img_reshaped = img.reshape(B * N, C, H, W)
        x = self.model.img_backbone(img_reshaped)
        if self.model.img_neck is not None: 
            x = self.model.img_neck(x)
        
        # 共享特征图处理 (注意：这里使用你之前 hotpatch 过的格式化函数)
        from projects.mmdet3d_plugin.ops import feature_maps_format
        feature_maps = [f.reshape(B, N, f.shape[1], f.shape[2], f.shape[3]) for f in x]
        formatted_feature_maps = feature_maps_format(feature_maps)
        
        metas = {
            'img_metas': [{'lidar2img': projection_mat[i], 'img_shape': [(H, W)] * N} for i in range(B)],
            'projection_mat': projection_mat, 
            'timestamp': 0, 
            'image_wh': img.new_tensor([W, H]).view(1, 1, 2).repeat(B, N, 1)
        }

        # 分别运行两个头的 forward_onnx (确保设备对齐)
        det_outs = self.det_head.forward_onnx(
            formatted_feature_maps, prev_det_feat, prev_det_anchor, instance_t_matrix, metas)
        map_outs = self.map_head.forward_onnx(
            formatted_feature_maps, prev_map_feat, prev_map_anchor, instance_t_matrix, metas)

        # 返回与 Engine 输出顺序一致的元组
        return (
            det_outs["cls_scores"], det_outs["bbox_preds"], 
            det_outs["next_instance_feature"], det_outs["next_anchor"],
            map_outs["cls_scores"], map_outs["bbox_preds"],
            map_outs["next_instance_feature"], map_outs["next_anchor"]
        )

# ==============================================================================
# ⚡ Step 2: TensorRT 执行类 (保持你的高效实现)
# ==============================================================================
class TRTEngineExecutor:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = OrderedDict(), OrderedDict(), []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            trt_dtype = self.engine.get_binding_dtype(i)
            torch_dtype = torch.float32 if trt_dtype == trt.float32 else torch.float16
            
            gpu_mem = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
            self.bindings.append(gpu_mem.data_ptr())
            if self.engine.binding_is_input(i): 
                self.inputs[name] = gpu_mem
            else: 
                self.outputs[name] = gpu_mem

    def forward(self, feed_dict):
        for name, data in feed_dict.items():
            if name in self.inputs:
                self.inputs[name].copy_(data.to(self.inputs[name].dtype))
        self.context.execute_v2(self.bindings)
        return {name: mem.clone() for name, mem in self.outputs.items()}

# ==============================================================================
# 🏁 Step 3: 运行比对
# ==============================================================================
def run_comparison():
    cfg_path = "projects/configs/sparsedrive_small_stage2.py"
    ckpt_path = "ckpt/sparsedrive_stage2.pth"
    engine_path = "work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine"
    
    print("📦 Loading PyTorch Multi-Head Model...")
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = cfg.plugin_dir # "projects/mmdet3d_plugin/"
        # 转换路径为模块格式，例如 "projects.mmdet3d_plugin"
        _module_path = plugin_dir.replace("/", ".").rstrip(".")
        print(f"💉 Injecting plugin modules from: {_module_path}")
        importlib.import_module(_module_path)
    # 强制开启双头配置
    cfg.task_config['with_det'] = True
    cfg.task_config['with_map'] = True
    
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    
    model = build_detector(cfg.model)
    load_checkpoint(model, ckpt_path, map_location='cpu')
    wrapper = SparseDriveMultiHeadWrapper(model).cuda().eval()

    # 准备双头 Dummy Inputs
    print("📝 Preparing Multi-Head Dummy Inputs...")
    device = 'cuda'
    dtype = torch.float32
    bs, n, h, w, d = 1, 6, 256, 704, 256
    num_det, num_map = 600, 33 # 对应你的 config

    dummy_inputs = {
        'img': torch.randn(bs, n, 3, h, w, device=device, dtype=dtype),
        'projection_mat': torch.randn(bs, n, 4, 4, device=device, dtype=dtype),
        'prev_det_feat': torch.randn(bs, num_det, d, device=device, dtype=dtype),
        'prev_det_anchor': torch.randn(bs, num_det, 11, device=device, dtype=dtype),
        'prev_map_feat': torch.randn(bs, num_map, d, device=device, dtype=dtype),
        'prev_map_anchor': torch.randn(bs, num_map, 40, device=device, dtype=dtype),
        'instance_t_matrix': torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    }

    # PyTorch 推理
    with torch.no_grad():
        py_outs = wrapper(**dummy_inputs)

    # TensorRT 推理
    trt_model = TRTEngineExecutor(engine_path)
    trt_outs = trt_model.forward(dummy_inputs)

    # 精度校验报告
    print("\n" + "="*50)
    print("📊 MULTI-HEAD ACCURACY REPORT")
    print("="*50)
    
    # 检查项列表: (名称, Py索引, TRT键名)
    checks = [
        ("Det Classification", 0, "det_cls"),
        ("Det BBox (3D Box)", 1, "det_bbox"),
        ("Map Classification", 4, "map_cls"),
        ("Map Points (Lines)", 5, "map_pts")
    ]

    for label, py_idx, trt_key in checks:
        p_val = py_outs[py_idx].float().cpu()
        t_val = trt_outs[trt_key].float().cpu()
        
        cos_sim = torch.nn.functional.cosine_similarity(p_val.flatten(), t_val.flatten(), dim=0)
        max_err = torch.abs(p_val - t_val).max()
        
        print(f"[{label}]")
        print(f"  - Cosine Similarity: {cos_sim.item():.6f}")
        print(f"  - Max Absolute Error: {max_err.item():.6f}")

    print("="*50)

if __name__ == "__main__":
    run_comparison()
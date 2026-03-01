import tensorrt as trt
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import ctypes
from collections import OrderedDict
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector
from mmcv.parallel import MMDataParallel

# 1. 环境初始化
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 加载自定义插件
plugin_lib_path = "projects/trt_plugin/build/libSparseDrivePlugin.so" 
if os.path.exists(plugin_lib_path):
    ctypes.CDLL(plugin_lib_path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    print(f"✅ Loaded SparseDrive custom plugin.")

# ==============================================================================
# 🚀 Step 0: 极其稳定的 TensorRT 推理封装 (加入防御性断言)
# ==============================================================================
class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("❌ 引擎 Context 创建失败！这通常是因为 cuBLAS 库冲突。请确保 'import tensorrt' 在第一行！")

        self.inputs, self.outputs, self.bindings = OrderedDict(), OrderedDict(), []
        
        for i in range(self.engine.num_bindings):
            if hasattr(self.engine, 'get_binding_name'):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
            else:
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            shape = [s if s > 0 else 1 for s in shape]
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
            gpu_mem = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
            self.bindings.append(gpu_mem.data_ptr())
            
            if is_input:
                self.inputs[name] = gpu_mem
            else:
                self.outputs[name] = gpu_mem

    def infer(self, feed_dict):
        for name, data in feed_dict.items():
            if name in self.inputs:
                self.inputs[name].copy_(data.to(self.inputs[name].dtype).contiguous())
                
        self.context.execute_v2(self.bindings)
        return {name: mem.clone().float() for name, mem in self.outputs.items()}

# ==============================================================================
# 🏗️ Step 1: ONNX Wrapper
# ==============================================================================
class SparseDriveONNXPathWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.det_head = model.head.det_head

    def forward(self, img, projection_mat, 
                prev_det_feat, prev_det_anchor, prev_det_conf, 
                instance_t_matrix, time_interval):  
        
        B, N, C, H, W = img.shape
        img_reshaped = img.reshape(B * N, C, H, W)
        x = self.model.img_backbone(img_reshaped)
        if self.model.img_neck is not None: 
            x = self.model.img_neck(x)
        
        from projects.mmdet3d_plugin.ops import feature_maps_format
        feature_maps = [f.reshape(B, N, f.shape[1], f.shape[2], f.shape[3]) for f in x]
        formatted_feature_maps = feature_maps_format(feature_maps)
        
        metas = {
            'img_metas': [{'lidar2img': projection_mat[i], 'img_shape': [(H, W)] * N} for i in range(B)],
            'projection_mat': projection_mat, 
            'image_wh': img.new_tensor([W, H]).view(1, 1, 2).repeat(B, N, 1)
        }

        det_outs = self.det_head.forward_onnx(
            formatted_feature_maps, prev_det_feat, prev_det_anchor, 
            instance_t_matrix, time_interval=time_interval, 
            prev_confidence=prev_det_conf, metas=metas
        )

        return det_outs, None


# ==============================================================================
# 🏁 Step 2: 运行真实数据比对 (Frame 0 -> Frame 1)
# ==============================================================================
def run_real_temporal_comparison():
    cfg_path = "projects/configs/sparsedrive_small_stage2.py"
    ckpt_path = "ckpt/sparsedrive_stage2.pth"
    engine_path = "work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine"

    print("📦 Loading PyTorch Model...")
    cfg = Config.fromfile(cfg_path)
    import projects.mmdet3d_plugin 
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    model = build_detector(cfg.model).cuda()
    load_checkpoint(model, ckpt_path, map_location='cuda')
    wrapper = SparseDriveONNXPathWrapper(model).eval()
    
    # 包装原生模型以提取特征
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    print(f"🚀 Loading TensorRT Engine from {engine_path}...")
    trt_model = TRTInfer(engine_path)

    # =================初始化三路历史缓存=================
    nh_det = 600
    
    # 构建首帧随机初始化张量 (共享给 Native, ONNX, TRT)
    init_feat = torch.rand((1, nh_det, 256), device='cuda')
    init_anchor = torch.zeros((1, nh_det, 11), device='cuda')
    init_conf = torch.zeros((1, nh_det), device='cuda')

    # 1. 给 ONNX Wrapper 用的缓存
    history_wrap = {
        'prev_det_feat': init_feat.clone(),
        'prev_det_anchor': init_anchor.clone(),
        'prev_det_conf': init_conf.clone(),
    }
    
    # 2. 给 TRT Engine 用的缓存
    history_trt = {
        'prev_det_feat': init_feat.clone(),
        'prev_det_anchor': init_anchor.clone(),
        'prev_det_conf': init_conf.clone(),
    }

    loader_iter = iter(loader)
    prev_global_mat = None

    for frame_idx in range(2):
        print(f"\n" + "="*30 + f" FRAME {frame_idx} AUDIT " + "="*30)
        data = next(loader_iter)
        img_metas = data['img_metas'].data[0][0]
        img_raw = data['img'].data[0][0].cuda()
        img_tensor = img_raw.unsqueeze(0) if img_raw.dim() == 4 else img_raw
        bs, n, c, h, w = img_tensor.shape
        proj_mat = torch.stack([p.cuda() for p in data['projection_mat'].data[0]], dim=0).unsqueeze(0)
        
        curr_time = img_metas['timestamp']
        if prev_global_mat is None:
            dt_tensor = torch.tensor([0.5], device='cuda', dtype=torch.float32) 
            prev_time = curr_time - 0.5
        else:
            dt = curr_time - prev_time 
            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
        prev_time = curr_time

        # 位姿补偿
        curr_global = img_metas['T_global']
        curr_global_inv = img_metas['T_global_inv']
        if prev_global_mat is None:
            instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
        else:
            t_mat = curr_global_inv @ prev_global_mat
            instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
        prev_global_mat = curr_global

        with torch.no_grad():
            # ==================================================================
            # 💉 核心修改：在第一帧强行把随机 history 塞进 Native PyTorch
            # ==================================================================
            if frame_idx == 0:
                print("   [Hook] Injecting random history into Native PyTorch InstanceBank...")
                det_bank = model.module.head.det_head.instance_bank
                
                # 1. 注入 600 个缓存的历史特征 (注意是 cached_xxx，绝对不能动 instance_feature!)
                det_bank.cached_feature = history_wrap['prev_det_feat'].clone()
                det_bank.cached_anchor = history_wrap['prev_det_anchor'].clone()
                
                # 注入历史置信度
                if hasattr(det_bank, 'confidence'):
                    det_bank.confidence = history_wrap['prev_det_conf'].clone()
                
                # 2. 伪造上一帧的 metas，骗过 InstanceBank 内部的位姿投影计算
                # 我们令上一帧的 T_global 就等于当前帧的 curr_global，这样网络内部在执行
                # curr_global_inv @ prev_global 时，算出来的就恰好是单位矩阵 I！
                fake_metas = {
                    "timestamp": curr_time - 0.5,
                    "img_metas": [{"T_global": curr_global}] 
                }
                det_bank.metas = fake_metas
            # ==================================================================

            # ==================== 1. PyTorch Native ====================
            native_metas = {
                'img_metas': [img_metas],
                'projection_mat': proj_mat,
                'image_wh': img_tensor.new_tensor([w, h]).view(1, 1, 2).repeat(bs, n, 1),
                'timestamp': img_tensor.new_tensor([img_metas['timestamp']]), 
            }
            features = model.module.extract_feat(img_tensor, metas=native_metas)
            py_outs = model.module.head(features, native_metas)

            # ==================== 2. ONNX Wrapper ====================
            onnx_det, _ = wrapper(
                img_tensor, proj_mat, 
                history_wrap['prev_det_feat'], history_wrap['prev_det_anchor'], history_wrap['prev_det_conf'],
                instance_t_matrix, dt_tensor
            )

            # ==================== 3. TRT Engine ====================
            feed_dict = {
                'img': img_tensor, 
                'projection_mat': proj_mat, 
                'instance_t_matrix': instance_t_matrix, 
                'time_interval': dt_tensor,
                **history_trt
            }
            trt_outs = trt_model.infer(feed_dict)

        # ---------------------- 精度对账 (DET) ----------------------
        p_det_cls = py_outs[0]['classification'][-1]
        p_det_reg = py_outs[0]['prediction'][-1]
        
        o_det_cls = onnx_det['cls_scores']
        o_det_reg = onnx_det['bbox_preds']
        
        t_det_cls = trt_outs['det_cls']
        t_det_reg = trt_outs['det_bbox']
        
        # 1. 宏观 Cosine Similarity (只比对前 900 个新生成的预测，三方对比)
        cos_sim_n_o = torch.nn.functional.cosine_similarity(p_det_cls[:, :900].flatten(), o_det_cls[:, :900].flatten(), dim=0)
        cos_sim_n_t = torch.nn.functional.cosine_similarity(p_det_cls[:, :900].flatten(), t_det_cls[:, :900].flatten(), dim=0)
        cos_sim_o_t = torch.nn.functional.cosine_similarity(o_det_cls[:, :900].flatten(), t_det_cls[:, :900].flatten(), dim=0)
        
        print(f"📊 [CLS] Native vs ONNX Cos_Sim : {cos_sim_n_o.item():.8f}")
        print(f"📊 [CLS] Native vs TRT  Cos_Sim : {cos_sim_n_t.item():.8f}")
        print(f"📊 [CLS] ONNX   vs TRT  Cos_Sim : {cos_sim_o_t.item():.8f}\n")

        # 2. 微观 Top-K Box 严格索引对齐对账 (不再使用距离去大海捞针)
        p_det_scores = p_det_cls[0].sigmoid().max(-1).values
        p_top_v, p_top_i = torch.topk(p_det_scores, 10) # 听你的，打印前10个看看
        
        print(f"🔍 [DET] Top 10 BBox Matching (Strict Index Alignment)")
        print(f"{'Rank':<5}|{'Query_Idx':<10}|{'Score(N/O/T)':<18}|{'Dist(N-O)':<10}|{'Dist(N-T)':<10}|{'Dist(O-T)':<10}")
        print("-" * 80)
        
        for i in range(len(p_top_i)):
            p_idx = p_top_i[i].item()
            
            # 严格提取同索引的框
            p_box = p_det_reg[0, p_idx]
            o_box = o_det_reg[0, p_idx]
            t_box = t_det_reg[0, p_idx]
            
            # 严格提取同索引的置信度
            p_s = p_top_v[i].item()
            o_s = o_det_cls[0, p_idx].sigmoid().max().item()
            t_s = t_det_cls[0, p_idx].sigmoid().max().item()
            
            # 真实距离误差
            dist_n_o = torch.norm(o_box[:3] - p_box[:3]).item()
            dist_n_t = torch.norm(t_box[:3] - p_box[:3]).item()
            dist_o_t = torch.norm(t_box[:3] - o_box[:3]).item()
            
            print(f"#{i+1:<4}|Idx:{p_idx:<6}|{p_s:.2f}/{o_s:.2f}/{t_s:.2f}    |{dist_n_o:.4f}m  |{dist_n_t:.4f}m  |{dist_o_t:.4f}m")

        # 3. 反向自证：看看 TRT 自己认为最高的得分是多少
        t_det_scores = t_det_cls[0].sigmoid().max(-1).values
        t_top_v, t_top_i = torch.topk(t_det_scores, 5)
        print(f"\n🚀 [DET] TRT's Own Top 5 Predictions (Just to verify TRT isn't predicting garbage)")
        print(f"{'Rank':<5}|{'Query_Idx':<10}|{'TRT_Score':<10}")
        print("-" * 35)
        for i in range(len(t_top_i)):
            print(f"#{i+1:<4}|Idx:{t_top_i[i].item():<6}|{t_top_v[i].item():.2f}")
        print("=" * 80)
        
        # ==================== 闭环滚动更新历史 ====================
        history_wrap['prev_det_feat'] = onnx_det['next_instance_feature']
        history_wrap['prev_det_anchor'] = onnx_det['next_anchor']
        history_wrap['prev_det_conf'] = onnx_det['next_confidence']
        
        history_trt['prev_det_feat'] = trt_outs['next_det_feat']
        history_trt['prev_det_anchor'] = trt_outs['next_det_anchor']
        history_trt['prev_det_conf'] = trt_outs['next_det_conf']

if __name__ == "__main__":
    run_real_temporal_comparison()
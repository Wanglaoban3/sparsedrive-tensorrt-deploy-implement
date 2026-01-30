import torch
import time
import numpy as np
import tensorrt as trt
from collections import OrderedDict
import ctypes
import os

# 1. 必须先加载 .so 插件，否则无法识别自定义算子
plugin_lib_path = "projects/trt_plugin/build/libSparseDrivePlugin.so" 
if os.path.exists(plugin_lib_path):
    ctypes.CDLL(plugin_lib_path, mode=ctypes.RTLD_GLOBAL)
    # 这一步是让 TRT 扫描并注册所有插件
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    print(f"✅ Loaded SparseDrive custom plugin.")
else:
    print(f"❌ Plugin not found at {plugin_lib_path}")
    
# 沿用你之前的 TRTInfer 类
class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = OrderedDict(), OrderedDict(), []
        
        for i in range(self.engine.num_bindings):
            if hasattr(self.engine, 'get_binding_name'):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
            else:
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                
            gpu_mem = torch.empty(tuple(shape), dtype=torch.float32, device='cuda')
            self.bindings.append(gpu_mem.data_ptr())
            if is_input:
                self.inputs[name] = gpu_mem
            else:
                self.outputs[name] = gpu_mem

    def infer(self, feed_dict):
        for name, data in feed_dict.items():
            if name in self.inputs:
                self.inputs[name].copy_(data.to(self.inputs[name].dtype))
        
        # 计时核心区域
        self.context.execute_v2(self.bindings)
        
        return self.outputs

def benchmark_sparsedrive():
    engine_path = "work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine"
    num_warmup = 50
    num_iters = 200

    print(f"🚀 Loading Engine: {engine_path}")
    model = TRTInfer(engine_path)

    # 1. 构造随机输入数据 (根据模型定义)
    # 请根据你实际的输入维度调整这些 dummy 数据
    dummy_inputs = {
        'img': torch.randn(1, 6, 3, 256, 704).cuda(), # B, N, C, H, W
        'projection_mat': torch.randn(1, 6, 4, 4).cuda(),
        'instance_t_matrix': torch.eye(4).cuda().unsqueeze(0),
        'prev_det_feat': torch.randn(1, 600, 256).cuda(),
        'prev_det_anchor': torch.randn(1, 600, 11).cuda(),
        'prev_map_feat': torch.randn(1, 33, 256).cuda(),
        'prev_map_anchor': torch.randn(1, 33, 40).cuda(),
    }

    print(f"🔥 Warming up for {num_warmup} iterations...")
    for _ in range(num_warmup):
        _ = model.infer(dummy_inputs)
    torch.cuda.synchronize()

    print(f"⏱️  Profiling for {num_iters} iterations...")
    latencies = []
    
    for i in range(num_iters):
        # 只针对推理核心部分计时
        start_time = time.perf_counter()
        
        # 执行推理 (注意：这里我们不拷贝数据回 CPU，模拟真实端侧流程)
        model.context.execute_v2(model.bindings)
        
        torch.cuda.synchronize() # 确保 GPU 计算完成
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # 转换为 ms

    # 统计结果
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency

    print("\n" + "="*30 + " PERFORMANCE REPORT " + "="*30)
    print(f"  - Device:           {torch.cuda.get_device_name(0)}")
    print(f"  - Precision:        FP16 (Global)")
    print(f"  - Average Latency:  {avg_latency:.2f} ms")
    print(f"  - Median Latency:   {median_latency:.2f} ms")
    print(f"  - Std Deviation:    {std_latency:.2f} ms")
    print(f"  - Min / Max:        {np.min(latencies):.2f} ms / {np.max(latencies):.2f} ms")
    print(f"  - FPS:              {fps:.1f} frames/sec")
    print("="*80)

if __name__ == "__main__":
    benchmark_sparsedrive()
import torch
import numpy as np
import tensorrt as trt
from collections import OrderedDict
import time
import ctypes
import os

# 1. 定义 Profiler 类，用于抓取 TRT 内部层级耗时
class SimpleProfiler(trt.IProfiler):
    def __init__(self):
        super(SimpleProfiler, self).__init__()
        self.layers = OrderedDict()

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []
        self.layers[layer_name].append(ms)

    def print_stats(self, top_n=20):
        print(f"\n" + "📊" * 5 + " LAYER-WISE LATENCY ANALYSIS " + "📊" * 5)
        print(f"{'Layer Name (Top ' + str(top_n) + ')':<70} | {'Avg Time (ms)':<15}")
        print("-" * 90)
        
        # 计算平均耗时并排序
        stats = []
        for name, times in self.layers.items():
            stats.append((name, np.mean(times)))
        
        stats.sort(key=lambda x: x[1], reverse=True)
        
        for name, avg_time in stats[:top_n]:
            # 缩短超长的层名，方便阅读
            display_name = (name[:67] + '...') if len(name) > 70 else name
            print(f"{display_name:<70} | {avg_time:<15.4f}")
        
        total_time = sum([s[1] for s in stats])
        print("-" * 90)
        print(f"{'TOTAL MEASURED ENGINE TIME':<70} | {total_time:<15.4f} ms")
        print("📊" * 15 + "\n")

# 2. 核心推理与性能分析类
class TRTProfiler:
    def __init__(self, engine_path, plugin_path):
        # 加载插件
        if os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
            trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
            print(f"✅ Plugin registered from {plugin_path}")

        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.profiler = SimpleProfiler()
        self.context.profiler = self.profiler # 挂载分析器

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

    def profile(self, feed_dict, num_iters=50):
        # 数据拷贝到输入端
        for name, data in feed_dict.items():
            if name in self.inputs:
                self.inputs[name].copy_(data.to(self.inputs[name].dtype))
        
        print(f"🔥 Starting Profiling for {num_iters} iterations...")
        # 为了让平均值更准，建议忽略前几次的启动波动
        for i in range(num_iters):
            self.context.execute_v2(self.bindings)
            torch.cuda.synchronize()
        
        self.profiler.print_stats()

def run_performance_audit():
    engine_path = "work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine"
    plugin_path = "projects/trt_plugin/build/libSparseDrivePlugin.so"

    # 1. 构造测试数据 (必须与模型定义对齐)
    dummy_inputs = {
        'img': torch.randn(1, 6, 3, 256, 704).cuda(),
        'projection_mat': torch.randn(1, 6, 4, 4).cuda(),
        'instance_t_matrix': torch.eye(4).cuda().unsqueeze(0),
        'prev_det_feat': torch.randn(1, 600, 256).cuda(),
        'prev_det_anchor': torch.randn(1, 600, 11).cuda(),
        'prev_map_feat': torch.randn(1, 33, 256).cuda(),
        'prev_map_anchor': torch.randn(1, 33, 40).cuda(),
    }

    # 2. 运行性能审计
    tester = TRTProfiler(engine_path, plugin_path)
    tester.profile(dummy_inputs)

    # 3. 内存占用审计
    print("🧠" * 5 + " MEMORY EFFICIENCY AUDIT " + "🧠" * 5)
    allocated = torch.cuda.memory_allocated() / (1024**2)
    max_mem = torch.cuda.max_memory_allocated() / (1024**2)
    
    # 获取显卡显存信息
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    
    print(f"  - Device Total Mem:    {total_mem:.2f} MB")
    print(f"  - Model Allocated:     {allocated:.2f} MB")
    print(f"  - Peak Memory Usage:   {max_mem:.2f} MB")
    print(f"  - Memory Utilization:  {(max_mem/total_mem)*100:.2%}")
    print("🧠" * 15)

if __name__ == "__main__":
    run_performance_audit()
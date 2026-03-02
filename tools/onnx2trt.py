import tensorrt as trt
import os
import argparse
import ctypes
import sys

def build_engine(onnx_file_path, engine_file_path, plugin_path, fp16=False, verbose=False):
    # 1. 基础检查
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found at {onnx_file_path}")
        return
    if not os.path.exists(plugin_path):
        print(f"Error: Plugin library not found at {plugin_path}")
        return

    # 2. 加载插件
    print(f"Loading plugin from {plugin_path}...")
    try:
        ctypes.CDLL(plugin_path)
    except OSError as e:
        print(f"Error loading plugin library: {e}")
        return

    # 3. 初始化 Builder
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    
    # 显式 Batch 标志
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    if hasattr(config, 'builder_optimization_level'):
        config.builder_optimization_level = 1

    # =========================================================================
    # 🛡️🛡️🛡️ 核心修复：白名单策略禁用 Myelin 🛡️🛡️🛡️
    # =========================================================================
    print(f"Detected TensorRT Version: {trt.__version__}")
    print("Applying Tactic Source Allow-list (Safe Mode)...")
    try:
        safe_sources = 0
        if "CUBLAS" in trt.TacticSource.__members__:
            safe_sources |= 1 << int(trt.TacticSource.CUBLAS)
        if "CUBLAS_LT" in trt.TacticSource.__members__:
            safe_sources |= 1 << int(trt.TacticSource.CUBLAS_LT)
        if "CUDNN" in trt.TacticSource.__members__:
            safe_sources |= 1 << int(trt.TacticSource.CUDNN)
        if "EDGE_MASK_CONVOLUTIONS" in trt.TacticSource.__members__:
             safe_sources |= 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)

        config.set_tactic_sources(safe_sources)
    except Exception as e:
        print(f"Warning: Failed to set tactic sources: {e}")
    # =========================================================================

    # 5. 配置显存 (8GB)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33) 
    except AttributeError:
        config.max_workspace_size = 1 << 33
    
    # 7. 解析 ONNX
    parser = trt.OnnxParser(network, logger)
    print(f"Parsing ONNX model from {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # =========================================================================
    # 🛡️ 智能混合精度防溢出：仅针对 Softmax 和 Exp 开启 FP32 Fallback
    # ⚠️ 彻底删除了无差别的“精度擦除”代码，完美保护 Int32 形状推导逻辑！
    # =========================================================================
    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 with Strict Mixed Precision Fallback...")
        config.set_flag(trt.BuilderFlag.FP16)
        
        # 强制 TensorRT 听从我们指定的节点精度
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        # 仅针对导致溢出的指数爆炸核心元凶
        dangerous_keywords = ['Softmax', 'Exp']

        fallback_count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            
            if any(k in layer_name for k in dangerous_keywords):
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                fallback_count += 1
                
        print(f"✅ Successfully forced {fallback_count} dangerous floating-point layers to FP32.")
    # =========================================================================

    # 8. 构建
    print("Building TensorRT engine... (Myelin should be inactive)")
    try:
        # TRT 8.5+ 推荐用法
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("Error: Build serialized network failed.")
            return
        engine_bytes = plan
    except AttributeError:
        # 旧版兼容
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Error: Build engine failed.")
            return
        engine_bytes = engine.serialize()

    # 9. 保存
    print(f"Saving engine to {engine_file_path}...")
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print("🎉 Done! Engine built successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx")
    parser.add_argument("--save", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine")
    parser.add_argument("--plugin", default="./projects/trt_plugin/build/libSparseDrivePlugin.so")
    parser.add_argument("--fp16", default=True) # 默认开启FP16
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    build_engine(args.onnx, args.save, args.plugin, args.fp16, args.verbose)
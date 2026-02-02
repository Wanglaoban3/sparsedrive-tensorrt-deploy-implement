import tensorrt as trt
import os
import argparse
import ctypes
import sys

def build_engine(onnx_file_path, engine_file_path, plugin_path, fp16=False, verbose=False):
    # 1. 检查文件是否存在
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found at {onnx_file_path}")
        return
    if not os.path.exists(plugin_path):
        print(f"Error: Plugin library not found at {plugin_path}")
        print("Please compile the plugin first.")
        return

    # 2. 加载自定义插件库 (.so)
    # 这一步非常关键！加载库会自动触发 REGISTER_TENSORRT_PLUGIN 宏
    print(f"Loading plugin from {plugin_path}...")
    try:
        ctypes.CDLL(plugin_path)
    except OSError as e:
        print(f"Error loading plugin library: {e}")
        return

    # 3. 初始化 TensorRT Logger
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    
    # 初始化标准插件库 (可选，如果模型用了标准插件如 InstanceNorm 等)
    trt.init_libnvinfer_plugins(logger, "")

    # 4. 创建 Builder 和 Network
    builder = trt.Builder(logger)
    
    # EXPLICIT_BATCH 标志是解析 ONNX 必须的
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    parser = trt.OnnxParser(network, logger)

    # 5. 配置内存池 (Workspace)
    # TensorRT 8.x+ 使用 set_memory_pool_limit
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33) # 1GB
    except AttributeError:
        # 旧版本 TensorRT (7.x)
        config.max_workspace_size = 1 << 33

    # 6. 配置 FP16
    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 precision.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: FP16 requested but not supported by platform. Falling back to FP32.")

    # 7. 解析 ONNX 模型
    print(f"Parsing ONNX model from {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 检查输入输出维度 (调试用)
    print("Network inputs:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"  Input {i}: {tensor.name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
    
    print("Network outputs:")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"  Output {i}: {tensor.name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")

    # 8. 构建并序列化 Engine
    print("Building TensorRT engine... This may take a while.")
    # create_tensorrt_engine 可能会被废弃，尝试使用 build_serialized_network
    try:
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("Error: Build serialized network failed.")
            return
        engine_bytes = plan
    except AttributeError:
        # 兼容旧版本
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Error: Build engine failed.")
            return
        engine_bytes = engine.serialize()

    # 9. 保存到文件
    print(f"Saving engine to {engine_file_path}...")
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT Engine from ONNX with Custom Plugin")
    parser.add_argument("--onnx", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx", help="Path to input ONNX model")
    parser.add_argument("--save", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine", help="Path to output TensorRT engine")
    parser.add_argument("--plugin", default="./projects/trt_plugin/build/libSparseDrivePlugin.so", help="Path to compiled plugin .so library")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    build_engine(args.onnx, args.save, args.plugin, args.fp16, args.verbose)
import tensorrt as trt
import os
import argparse
import ctypes
import sys

def build_engine(
    onnx_file_path,
    engine_file_path,
    plugin_path,
    fp16=False,
    verbose=False,
    workspace_mb=2048,
    builder_optimization_level=0,
    fp32_fallback_keywords=None,
):
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
        config.builder_optimization_level = int(builder_optimization_level)
        print(f"Builder optimization level: {config.builder_optimization_level}")

    print(f"Detected TensorRT Version: {trt.__version__}")
    print("Enabling tactic sources and allowing Myelin for graph fusion...")
    try:
        # 获取 TensorRT 默认开启的所有 Tactic（默认包含 Myelin, CUBLAS, CUDNN 等）
        tactic_sources = config.get_tactic_sources()
        
        # 如果你之前遇到过特定引擎崩溃（比如 JIT_CONVOLUTIONS），仅在这里使用黑名单剔除
        # 对应你之前 onnx2trt.sh 里的 --tacticSources=-JIT_CONVOLUTIONS
        if "JIT_CONVOLUTIONS" in trt.TacticSource.__members__:
            tactic_sources &= ~(1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
            print("Disabled JIT_CONVOLUTIONS via blocklist.")
            
        config.set_tactic_sources(tactic_sources)
    except Exception as e:
        print(f"Warning: Failed to set tactic sources: {e}")

    # 5. Configure workspace. Keep this modest on small-VRAM builder GPUs.
    workspace_bytes = int(workspace_mb) << 20
    print(f"Workspace limit: {workspace_mb} MB")
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except AttributeError:
        config.max_workspace_size = workspace_bytes
    
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
    # Mixed precision policy: keep broad Softmax subgraphs in FP16 by default,
    # and only pin explicitly requested risky layers to FP32.
    # =========================================================================
    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 with targeted precision constraints...")
        config.set_flag(trt.BuilderFlag.FP16)

        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        fallback_keywords = fp32_fallback_keywords or ["Exp"]

        fallback_count = 0
        dfa_fp16_count = 0
        
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            layer_type = str(layer.type)
            
            if any(k in layer_name or k in layer_type for k in fallback_keywords):
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                fallback_count += 1
                
            elif layer.type == trt.LayerType.PLUGIN and 'DeformableAggregation' in layer_name:
                layer.precision = trt.DataType.HALF
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.HALF)
                dfa_fp16_count += 1
                
        print(f"FP32 fallback keywords: {fallback_keywords}")
        print(f"Forced {fallback_count} selected layers to FP32.")
        print(f"Forced {dfa_fp16_count} DeformableAggregation layers to FP16.")
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
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 build")
    parser.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT workspace limit in MB")
    parser.add_argument("--builder-optimization-level", type=int, default=0, choices=range(0, 6))
    parser.add_argument(
        "--fp32-fallback-keyword",
        action="append",
        dest="fp32_fallback_keywords",
        default=["Exp"],
        help="Layer name/type keyword to force to FP32. Repeat to add more.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    build_engine(
        args.onnx,
        args.save,
        args.plugin,
        fp16=args.fp16,
        verbose=args.verbose,
        workspace_mb=args.workspace_mb,
        builder_optimization_level=args.builder_optimization_level,
        fp32_fallback_keywords=args.fp32_fallback_keywords,
    )

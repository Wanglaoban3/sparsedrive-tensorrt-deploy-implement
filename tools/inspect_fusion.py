import tensorrt as trt
import json
import ctypes
import os

def inspect_engine(engine_path, plugin_path):
    logger = trt.Logger(trt.Logger.INFO)
    if os.path.exists(plugin_path):
        ctypes.CDLL(plugin_path)
    trt.init_libnvinfer_plugins(logger, "")

    print(f"🔍 Loading engine: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("❌ Failed to load engine.")
        return

    inspector = engine.create_engine_inspector()
    # 这里的详细程度依赖于你 build 时设置的 profiling_verbosity
    info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    data = json.loads(info_str)
    layers = data.get('Layers', [])
    
    print(f"📊 Total layers: {len(layers)}")
    print("\n" + "="*100)
    print(f"{'Layer Name':<50} | {'Tactic/Implementation'}")
    print("="*100)
    
    # 打印前 40 层，看看核心计算层都在用什么 Tactic
    for i, layer in enumerate(layers[:40]):
        if isinstance(layer, str):
            layer_data = json.loads(layer)
        else:
            layer_data = layer

        name = layer_data.get('Name', 'Unknown')
        # Tactic 字段在 DETAILED 模式下会包含具体的 Kernel 名字
        tactic = str(layer_data.get('Tactic', ''))
        
        # 截断超长名字
        display_name = (name[:47] + '...') if len(name) > 50 else name
        print(f"{display_name:<50} | {tactic}")

    print("="*100)
    del inspector
    del engine

if __name__ == "__main__":
    engine_file = "work_dirs/sparsedrive_small_stage2/sparsedrive.engine"
    plugin_so = "projects/trt_plugin/build/libSparseDrivePlugin.so"
    inspect_engine(engine_file, plugin_so)
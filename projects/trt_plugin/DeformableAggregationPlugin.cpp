#include "DeformableAggregationPlugin.h"
#include <cuda_runtime.h>
#include <iostream>

// 声明 CUDA Kernel 启动函数
extern "C" void DeformableAggregationLauncher(
    const void* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const void* sample_location, const void* weights, void* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared, 
    bool is_fp16,
    cudaStream_t stream
);

namespace nvinfer1 {

// 注册插件
REGISTER_TENSORRT_PLUGIN(DeformableAggregationPluginCreator);

DeformableAggregationPlugin::DeformableAggregationPlugin(const void* data, size_t length) {}

DimsExprs DeformableAggregationPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    // Input 0: mc_ms_feat [B, N_feat, C]
    // Input 3: sample_location [B, N_anchors, N_pts, N_cam, 2]
    // Output: [B, N_anchors, C]
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0]; // Batch
    output.d[1] = inputs[3].d[1]; // Num Anchors
    output.d[2] = inputs[0].d[2]; // Channels (C)
    return output;
}

// ⬇️ 修复：这里应该是 supportsFormatCombination，你之前写成了 enqueue
bool DeformableAggregationPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    if (pos == 1 || pos == 2) {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    // 允许 FLOAT 和 HALF (FP16)，并要求所有浮点张量精度一致
    return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) && 
           inOut[pos].format == TensorFormat::kLINEAR &&
           inOut[pos].type == inOut[0].type;
}

void DeformableAggregationPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

// 真正的 enqueue 方法
int DeformableAggregationPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    // mc_ms_feat: [B, num_feat, num_embeds] 
    int dims0 = inputDesc[0].dims.nbDims;
    int num_embeds = inputDesc[0].dims.d[dims0 - 1];
    int num_feat   = inputDesc[0].dims.d[dims0 - 2];
    int batch_size = (dims0 >= 3) ? inputDesc[0].dims.d[dims0 - 3] : 1;
    
    // spatial_shape: [num_scale, 2] 
    int dims1 = inputDesc[1].dims.nbDims;
    int num_scale = inputDesc[1].dims.d[dims1 - 2];

    // sample_location: [B, num_anchors, num_pts, num_cams, 2]
    int dims3 = inputDesc[3].dims.nbDims;
    int num_cams    = inputDesc[3].dims.d[dims3 - 2]; 
    int num_pts     = inputDesc[3].dims.d[dims3 - 3];
    int num_anchors = inputDesc[3].dims.d[dims3 - 4];

    // weights: [B, num_anchors, num_pts, num_cams, num_scale, num_groups]
    int dims4 = inputDesc[4].dims.nbDims;
    int num_groups = inputDesc[4].dims.d[dims4 - 1];
    if (num_groups <= 0) num_groups = 1;

    int start_index_vol = 1;
    for(int i = 0; i < inputDesc[2].dims.nbDims; ++i) {
        start_index_vol *= inputDesc[2].dims.d[i];
    }
    bool is_cam_shared = (start_index_vol == num_scale);

    // 探测当前处于 FP16 还是 FP32 模式
    bool is_fp16 = (inputDesc[0].type == DataType::kHALF);
    size_t type_size = is_fp16 ? 2 : 4; 
    size_t output_size = batch_size * num_anchors * num_embeds * type_size;
    
    cudaMemsetAsync(outputs[0], 0, output_size, stream);

    DeformableAggregationLauncher(
        inputs[0], (const int*)inputs[1], (const int*)inputs[2],
        inputs[3], inputs[4],
        outputs[0],
        batch_size, num_cams, num_feat, num_embeds,
        num_scale, num_anchors, num_pts, num_groups,
        is_cam_shared, 
        is_fp16,
        stream
    );

    return 0;
}

// Creator 实现
DeformableAggregationPluginCreator::DeformableAggregationPluginCreator() {
    mFC.nbFields = 0;
    mFC.fields = nullptr;
    mNamespace = ""; 
}

IPluginV2* DeformableAggregationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    return new DeformableAggregationPlugin();
}

IPluginV2* DeformableAggregationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new DeformableAggregationPlugin(serialData, serialLength);
}

} // namespace nvinfer1
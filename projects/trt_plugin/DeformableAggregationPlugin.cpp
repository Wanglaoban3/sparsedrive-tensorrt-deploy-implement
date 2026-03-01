#include "DeformableAggregationPlugin.h"
#include <cuda_runtime.h>
#include <iostream>

// 声明 CUDA Kernel 启动函数
extern "C" void DeformableAggregationLauncher(
    const float* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const float* sample_location, const float* weights, float* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared, // <--- 新增标志位
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

bool DeformableAggregationPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    // 0: feat(float), 1: shape(int), 2: start(int), 3: loc(float), 4: weight(float), 5: output(float)
    if (pos == 1 || pos == 2) {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}

void DeformableAggregationPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

int DeformableAggregationPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    // --- 核心修复 1: 从后往前动态解析维度，彻底免疫 TRT 的 Squeeze 操作 ---
    
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

    // --- 核心修复 2: 探测 spatial_shape 是否在相机维度上被常量折叠 ---
    int start_index_vol = 1;
    for(int i = 0; i < inputDesc[2].dims.nbDims; ++i) {
        start_index_vol *= inputDesc[2].dims.d[i];
    }
    // 如果总元素个数等于 num_scale，说明6个相机合并共用了一份 offset
    bool is_cam_shared = (start_index_vol == num_scale);

    size_t output_size = batch_size * num_anchors * num_embeds * sizeof(float);
    cudaMemsetAsync(outputs[0], 0, output_size, stream);

    DeformableAggregationLauncher(
        (const float*)inputs[0], (const int*)inputs[1], (const int*)inputs[2],
        (const float*)inputs[3], (const float*)inputs[4],
        (float*)outputs[0],
        batch_size, num_cams, num_feat, num_embeds,
        num_scale, num_anchors, num_pts, num_groups,
        is_cam_shared, // 传给 CUDA Kernel
        stream
    );

    return 0;
}

// Creator 实现
DeformableAggregationPluginCreator::DeformableAggregationPluginCreator() {
    mFC.nbFields = 0;
    mFC.fields = nullptr;
    // [修改点] 将 "SparseDrive" 改为 ""
    mNamespace = ""; 
}

IPluginV2* DeformableAggregationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    return new DeformableAggregationPlugin();
}

IPluginV2* DeformableAggregationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new DeformableAggregationPlugin(serialData, serialLength);
}

} // namespace nvinfer1
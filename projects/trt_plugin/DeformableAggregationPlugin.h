#pragma once
#include "NvInfer.h"
#include <vector>
#include <string>

namespace nvinfer1 {

class DeformableAggregationPlugin : public IPluginV2DynamicExt {
public:
    DeformableAggregationPlugin() = default;
    DeformableAggregationPlugin(const void* data, size_t length); // 反序列化构造

    // IPluginV2DynamicExt 方法
    int getNbOutputs() const noexcept override { return 1; }
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override { return 0; }
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 方法
    const char* getPluginType() const noexcept override { return "DeformableAggregation"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2DynamicExt* clone() const noexcept override { return new DeformableAggregationPlugin(*this); }
    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override { return inputTypes[0]; }
    
    // 序列化
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {}
    void terminate() noexcept override {}
    int initialize() noexcept override { return 0; }

private:
    std::string mNamespace;
};

class DeformableAggregationPluginCreator : public IPluginCreator {
public:
    DeformableAggregationPluginCreator();
    const char* getPluginName() const noexcept override { return "DeformableAggregation"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
};

} // namespace nvinfer1
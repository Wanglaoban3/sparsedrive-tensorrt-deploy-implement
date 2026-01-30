#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// 核心计算函数 (Device)
__device__ float bilinear_sampling(
    const float *bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
    // 强制限制范围，防止越界导致的基址错误
    float h_fixed = fmaxf(0.0f, fminf(h_im, (float)height - 1.0001f));
    float w_fixed = fmaxf(0.0f, fminf(w_im, (float)width - 1.0001f));

    int h_low = (int)floorf(h_fixed);
    int w_low = (int)floorf(w_fixed);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h_fixed - (float)h_low;
    float lw = w_fixed - (float)w_low;
    
    // 使用更高精度的权重计算
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    int stride = num_embeds;
    int w_stride = width * stride;
    
    const float* ptr_low = bottom_data + h_low * w_stride + w_low * stride + base_ptr;
    const float* ptr_high = ptr_low + w_stride;

    // 显式取值，减少指针运算开销
    float v1 = ptr_low[0];          // (low, low)
    float v2 = ptr_low[stride];     // (low, high)
    float v3 = ptr_high[0];         // (high, low)
    float v4 = ptr_high[stride];    // (high, high)

    return hh * (hw * v1 + lw * v2) + lh * (hw * v3 + lw * v4);
}

__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    // 解析索引
    const float weight = *(weights + idx / (num_embeds / num_groups));
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;

    int real_anchor_index = batch_index * num_anchors + anchor_index;
    
    // 定位采样点
    const int loc_offset = ((real_anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];

    if (loc_w <= 0 || loc_w >= 1 || loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    // 获取特征图偏移
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    atomicAdd(
        output + real_anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}

// 供 C++ Plugin 调用的入口函数
extern "C" void DeformableAggregationLauncher(
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    float* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    cudaStream_t stream
) {
    // ... (函数体内容保持不变) ...
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    dim3 threads(128);
    dim3 blocks((num_kernels + threads.x - 1) / threads.x);

    deformable_aggregation_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds,
        num_scale, num_anchors, num_pts, num_groups
    );
}
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// 🚀 核心修复 1：完全对齐 PyTorch 的 Zero-Padding 双线性插值
__device__ float bilinear_sampling(
    const float *bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
    // 允许出现负数，使用 floorf 正确向下取整
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h_im - (float)h_low;
    const float lw = w_im - (float)w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;

    int stride = num_embeds;
    int w_stride = width * stride;

    // 默认赋值为 0，实现完美的 Zero-Padding
    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;

    // 只有严格在有效图像范围内的点才去读取显存，否则保持为 0
    if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width)
        v1 = bottom_data[base_ptr + h_low * w_stride + w_low * stride];
        
    if (h_low >= 0 && w_high >= 0 && h_low < height && w_high < width)
        v2 = bottom_data[base_ptr + h_low * w_stride + w_high * stride];

    if (h_high >= 0 && w_low >= 0 && h_high < height && w_low < width)
        v3 = bottom_data[base_ptr + h_high * w_stride + w_low * stride];

    if (h_high >= 0 && w_high >= 0 && h_high < height && w_high < width)
        v4 = bottom_data[base_ptr + h_high * w_stride + w_high * stride];

    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
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
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

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
    
    const int loc_offset = ((real_anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];

    // 🚀 核心修复 2：放宽越界判断，不要用 <= 0 误杀合法的边缘点 (0.0)，超出 -0.5 的由插值函数里的 0 自动处理
    if (loc_w <= -0.5f || loc_w >= 1.5f || loc_h <= -0.5f || loc_h >= 1.5f) return;
    
    int shape_index = is_cam_shared ? scale_index : (cam_index * num_scale + scale_index);
    const int current_start_index = scale_start_index[shape_index];
    
    int cam_offset = is_cam_shared ? (cam_index * (num_feat / num_cams)) : 0;
    
    const int max_feat_size = batch_size * num_feat * num_embeds;
    const int value_offset = (batch_index * num_feat + cam_offset + current_start_index) * num_embeds + channel_index;

    if (value_offset < 0 || value_offset >= max_feat_size) return;

    int shape_idx_2d = shape_index << 1;
    const int h = spatial_shape[shape_idx_2d];
    const int w = spatial_shape[shape_idx_2d + 1];

    if (h <= 0 || w <= 0 || h > num_feat || w > num_feat) return;

    long long max_access_offset = (long long)value_offset + (long long)h * w * num_embeds;
    if (max_access_offset > max_feat_size) return;

    const float h_im = loc_h * h - 0.5f;
    const float w_im = loc_w * w - 0.5f;

    atomicAdd(
        output + real_anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}

extern "C" void DeformableAggregationLauncher(
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    float* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared,
    cudaStream_t stream
) {
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    if (num_kernels <= 0) return;
    
    dim3 threads(128);
    dim3 blocks((num_kernels + threads.x - 1) / threads.x);

    deformable_aggregation_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds,
        num_scale, num_anchors, num_pts, num_groups,
        is_cam_shared
    );
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>

template <typename T>
__device__ float bilinear_sampling_fp32(
    const T *bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
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

    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;

    if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width)
        v1 = static_cast<float>(bottom_data[base_ptr + h_low * w_stride + w_low * stride]);
        
    if (h_low >= 0 && w_high >= 0 && h_low < height && w_high < width)
        v2 = static_cast<float>(bottom_data[base_ptr + h_low * w_stride + w_high * stride]);

    if (h_high >= 0 && w_low >= 0 && h_high < height && w_low < width)
        v3 = static_cast<float>(bottom_data[base_ptr + h_high * w_stride + w_low * stride]);

    if (h_high >= 0 && w_high >= 0 && h_high < height && w_high < width)
        v4 = static_cast<float>(bottom_data[base_ptr + h_high * w_stride + w_high * stride]);

    return (hh * hw) * v1 + (hh * lw) * v2 + (lh * hw) * v3 + (lh * lw) * v4;
}

template <typename T>
__global__ void deformable_aggregation_forward_kernel(
    const int num_outputs,
    const T* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const T* sample_location,
    const T* weights,
    T* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared
) {
    // 采用 Thread-per-Output，彻底消灭 atomicAdd
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs) return;

    int channel_index = idx % num_embeds;
    int anchor_index = (idx / num_embeds) % num_anchors;
    int batch_index = idx / (num_embeds * num_anchors);

    int real_anchor_index = batch_index * num_anchors + anchor_index;
    int group_index = channel_index / (num_embeds / num_groups);

    float out_val = 0.0f;

    for (int pts_index = 0; pts_index < num_pts; ++pts_index) {
        for (int cam_index = 0; cam_index < num_cams; ++cam_index) {
            
            int loc_offset = ((real_anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
            float loc_w = static_cast<float>(sample_location[loc_offset]);
            float loc_h = static_cast<float>(sample_location[loc_offset + 1]);

            if (loc_w <= -0.5f || loc_w >= 1.5f || loc_h <= -0.5f || loc_h >= 1.5f) continue;

            int cam_offset = is_cam_shared ? (cam_index * (num_feat / num_cams)) : 0;

            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                
                int shape_index = is_cam_shared ? scale_index : (cam_index * num_scale + scale_index);
                int current_start_index = scale_start_index[shape_index];
                
                int value_offset = (batch_index * num_feat + cam_offset + current_start_index) * num_embeds + channel_index;
                int shape_idx_2d = shape_index << 1;
                int h = spatial_shape[shape_idx_2d];
                int w = spatial_shape[shape_idx_2d + 1];

                if (h <= 0 || w <= 0 || h > num_feat || w > num_feat) continue;

                int weight_offset = ((((batch_index * num_anchors + anchor_index) * num_pts + pts_index) * num_cams + cam_index) * num_scale + scale_index) * num_groups + group_index;
                float weight = static_cast<float>(weights[weight_offset]);

                float h_im = loc_h * h - 0.5f;
                float w_im = loc_w * w - 0.5f;

                out_val += bilinear_sampling_fp32<T>(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight;
            }
        }
    }
    // 全局只执行一次显存写入
    output[idx] = static_cast<T>(out_val);
}

extern "C" void DeformableAggregationLauncher(
    const void* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const void* sample_location, const void* weights, void* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared,
    bool is_fp16,
    cudaStream_t stream
) {
    const int num_outputs = batch_size * num_anchors * num_embeds;
    if (num_outputs <= 0) return;
    
    dim3 threads(256);
    dim3 blocks((num_outputs + threads.x - 1) / threads.x);

    if (is_fp16) {
        deformable_aggregation_forward_kernel<__half><<<blocks, threads, 0, stream>>>(
            num_outputs,
            (const __half*)mc_ms_feat, spatial_shape, scale_start_index,
            (const __half*)sample_location, (const __half*)weights,
            (__half*)output,
            batch_size, num_cams, num_feat, num_embeds,
            num_scale, num_anchors, num_pts, num_groups,
            is_cam_shared
        );
    } else {
        deformable_aggregation_forward_kernel<float><<<blocks, threads, 0, stream>>>(
            num_outputs,
            (const float*)mc_ms_feat, spatial_shape, scale_start_index,
            (const float*)sample_location, (const float*)weights,
            (float*)output,
            batch_size, num_cams, num_feat, num_embeds,
            num_scale, num_anchors, num_pts, num_groups,
            is_cam_shared
        );
    }
}
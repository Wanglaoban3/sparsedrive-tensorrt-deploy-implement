#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// 核心计算函数 (Device)
__device__ float bilinear_sampling(
    const float *bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
    // 强制限制范围
    float h_fixed = fmaxf(0.0f, fminf(h_im, (float)height - 1.0001f));
    float w_fixed = fmaxf(0.0f, fminf(w_im, (float)width - 1.0001f));

    int h_low = (int)floorf(h_fixed);
    int w_low = (int)floorf(w_fixed);
    // [修复 Warning] 删除了未使用到的 h_high 和 w_high 变量

    float lh = h_fixed - (float)h_low;
    float lw = w_fixed - (float)w_low;
    
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    int stride = num_embeds;
    int w_stride = width * stride;
    
    const float* ptr_low = bottom_data + h_low * w_stride + w_low * stride + base_ptr;
    const float* ptr_high = ptr_low + w_stride;

    float v1 = ptr_low[0];          
    float v2 = ptr_low[stride];     
    float v3 = ptr_high[0];         
    float v4 = ptr_high[stride];    

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
    
    // === [CRITICAL FIX] 安全性检查开始 ===
    // 在 TensorRT Profile 阶段，spatial_shape 和 scale_start_index 可能是随机垃圾值。
    // 必须进行边界检查以防止 Segfault。

    // 1. 获取当前 scale 的起始索引
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int current_start_index = scale_start_index[cam_scale_index];
    
    // 2. 计算特征图总容量
    const int max_feat_size = batch_size * num_feat * num_embeds;
    const int value_offset = (batch_index * num_feat + current_start_index) * num_embeds + channel_index;

    // 如果起始偏移量已经越界，直接返回
    if (value_offset < 0 || value_offset >= max_feat_size) return;

    // 3. 读取 TRT 传来的 (可能是垃圾值的) h 和 w
    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    // 4. [最优雅的动态拦截] 
    // 任何一层的 h 或 w 绝对不可能超过所有层展平后的总像素数 (num_feat)
    // 这样既没有硬编码，又能防止下方乘法发生 Integer Overflow
    if (h <= 0 || w <= 0 || h > num_feat || w > num_feat) return;

    // 5. [防线增强] 检查当前尺度的最大访问范围是否越界
    // 因为 h 和 w 已经被 num_feat 限制，这里的乘法绝对不会再发生 int32 溢出
    long long max_access_offset = (long long)value_offset + (long long)h * w * num_embeds;
    if (max_access_offset > max_feat_size) return;

    // === 安全性检查结束 ===

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

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
    cudaStream_t stream
) {
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
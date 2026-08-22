#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#define DIVUP(m, n) ((m + n - 1) / n)

// =========================================================================
// 🎯 原子加万能补丁 (支持 float 和 __half)
// =========================================================================
__device__ __forceinline__ void dfa_atomicAdd(float* addr, float val) { atomicAdd(addr, val); }

__device__ __forceinline__ void dfa_atomicAdd(__half* addr, __half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, val);
#else
    unsigned int *addr_as_ui = (unsigned int *)((char *)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short h_raw = (size_t)addr & 2 ? (old >> 16) : (old & 0xFFFF);
        __half h_val = *reinterpret_cast<__half*>(&h_raw);
        __half h_sum = __hadd(h_val, val);
        unsigned short res_raw = *reinterpret_cast<unsigned short*>(&h_sum);
        unsigned int new_val = (size_t)addr & 2 ? (old & 0xFFFF) | (res_raw << 16) : (old & 0xFFFF0000) | res_raw;
        old = atomicCAS(addr_as_ui, assumed, new_val);
    } while (assumed != old);
#endif
}

// =========================================================================
// 1. Map 专用：点并行 + 原子加 (解决 300 点不对齐问题)
// =========================================================================
template<typename T>
__global__ void dfa_atomic_kernel(
    const T* __restrict__ mc_ms_feat, const int* __restrict__ spatial_shape,
    const int* __restrict__ scale_start_index, const T* __restrict__ sample_location,
    const T* __restrict__ weights, T* __restrict__ output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups, bool is_cam_shared) {
    
    // 一个线程处理一个向量 (8个通道)，并行在所有点上
    int num_vec = num_embeds / 8;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * num_anchors * num_pts * num_vec) return;

    // 索引映射
    int v_idx = tid % num_vec;
    int p_idx = (tid / num_vec) % num_pts;
    int a_idx = (tid / (num_vec * num_pts)) % num_anchors;
    int b_idx = tid / (num_vec * num_pts * num_anchors);
    
    int c_start = v_idx * 8;
    int g_idx = c_start / (num_embeds / num_groups);
    int real_a_idx = b_idx * num_anchors + a_idx;

    float fres[8] = {0.0f};

    for (int cam = 0; cam < num_cams; ++cam) {
        int loc_ptr = ((real_a_idx * num_pts + p_idx) * num_cams + cam) << 1;
        float lw = (float)sample_location[loc_ptr], lh = (float)sample_location[loc_ptr + 1];
        if (lw <= -0.5f || lw >= 1.5f || lh <= -0.5f || lh >= 1.5f) continue;

        int cam_off = is_cam_shared ? (cam * (num_feat / num_cams)) : 0;
        int w_base = (((real_a_idx * num_pts + p_idx) * num_cams + cam) * num_scale) * num_groups;

        for (int s = 0; s < num_scale; ++s) {
            int s_idx = is_cam_shared ? s : (cam * num_scale + s);
            int h = spatial_shape[s_idx << 1], w = spatial_shape[(s_idx << 1) + 1];
            float weight = (float)weights[w_base + s * num_groups + g_idx];
            const T* f_ptr = mc_ms_feat + (static_cast<size_t>(b_idx * num_feat + cam_off + scale_start_index[s_idx]) * num_embeds) + c_start;

            float py = lh * h - 0.5f, px = lw * w - 0.5f;
            int yl = floorf(py), xl = floorf(px);
            float dy = py - yl, dx = px - xl;
            float w00 = (1.0f-dy)*(1.0f-dx), w01 = (1.0f-dy)*dx, w10 = dy*(1.0f-dx), w11 = dy*dx;

            #pragma unroll
            for(int i=0; i<8; ++i) {
                float v00=0, v01=0, v10=0, v11=0;
                if (yl>=0 && xl>=0 && yl<h && xl<w) v00 = (float)f_ptr[yl*w*num_embeds + xl*num_embeds + i];
                if (yl>=0 && xl+1<w && yl<h && xl+1>=0) v01 = (float)f_ptr[yl*w*num_embeds + (xl+1)*num_embeds + i];
                if (yl+1<h && xl>=0 && yl+1>=0 && xl<w) v10 = (float)f_ptr[(yl+1)*w*num_embeds + xl*num_embeds + i];
                if (yl+1<h && xl+1<w && yl+1>=0 && xl+1>=0) v11 = (float)f_ptr[(yl+1)*w*num_embeds + (xl+1)*num_embeds + i];
                fres[i] += (w00*v00 + w01*v01 + w10*v10 + w11*v11) * weight;
            }
        }
    }

    T* out_ptr = output + (static_cast<size_t>(b_idx * num_anchors + a_idx) * num_embeds) + c_start;
    #pragma unroll
    for(int i=0; i<8; ++i) { dfa_atomicAdd(out_ptr + i, (T)fres[i]); }
}

// =========================================================================
// 2. Det 专用：高带宽向量化读取 (针对 13 Pts 场景)
// =========================================================================
template<typename T>
__global__ void dfa_vector_kernel(
    const T* __restrict__ mc_ms_feat, const int* __restrict__ spatial_shape,
    const int* __restrict__ scale_start_index, const T* __restrict__ sample_location,
    const T* __restrict__ weights, T* __restrict__ output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups, bool is_cam_shared) {
    
    int num_vec = num_embeds / 8;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * num_anchors * num_vec) return;

    int v_idx = tid % num_vec;
    int a_idx = (tid / num_vec) % num_anchors;
    int b_idx = tid / (num_vec * num_anchors);
    int real_a_idx = b_idx * num_anchors + a_idx;
    int c_start = v_idx * 8;
    int g_idx = c_start / (num_embeds / num_groups);

    float fres[8] = {0.0f};

    for (int p = 0; p < num_pts; ++p) {
        for (int cam = 0; cam < num_cams; ++cam) {
            int loc_ptr = ((real_a_idx * num_pts + p) * num_cams + cam) << 1;
            float lw = (float)sample_location[loc_ptr], lh = (float)sample_location[loc_ptr+1];
            if (lw <= -0.5f || lw >= 1.5f || lh <= -0.5f || lh >= 1.5f) continue;
            
            int cam_off = is_cam_shared ? (cam * (num_feat / num_cams)) : 0;
            int w_base = (((real_a_idx * num_pts + p) * num_cams + cam) * num_scale) * num_groups;

            for (int s = 0; s < num_scale; ++s) {
                int s_idx = is_cam_shared ? s : (cam * num_scale + s);
                int h = spatial_shape[s_idx << 1], w = spatial_shape[(s_idx << 1) + 1];
                float weight = (float)weights[w_base + s * num_groups + g_idx];
                const T* f_ptr = mc_ms_feat + (static_cast<size_t>(b_idx * num_feat + cam_off + scale_start_index[s_idx]) * num_embeds) + c_start;

                float py = lh * h - 0.5f, px = lw * w - 0.5f;
                int yl = floorf(py), xl = floorf(px);
                float dy = py - yl, dx = px - xl;

                #pragma unroll
                for(int i=0; i<8; ++i) {
                    float v00=0, v01=0, v10=0, v11=0;
                    if (yl>=0 && xl>=0 && yl<h && xl<w) v00 = (float)f_ptr[yl*w*num_embeds + xl*num_embeds + i];
                    if (yl>=0 && xl+1<w && yl<h && xl+1>=0) v01 = (float)f_ptr[yl*w*num_embeds + (xl+1)*num_embeds + i];
                    if (yl+1<h && xl>=0 && yl+1>=0 && xl<w) v10 = (float)f_ptr[(yl+1)*w*num_embeds + xl*num_embeds + i];
                    if (yl+1<h && xl+1<w && yl+1>=0 && xl+1>=0) v11 = (float)f_ptr[(yl+1)*w*num_embeds + (xl+1)*num_embeds + i];
                    fres[i] += ((1.f-dy)*(1.f-dx)*v00 + (1.f-dy)*dx*v01 + dy*(1.f-dx)*v10 + dy*dx*v11) * weight;
                }
            }
        }
    }
    T* out_ptr = output + (static_cast<size_t>(tid) * 8);
    #pragma unroll
    for(int i=0; i<8; ++i) { out_ptr[i] = (T)fres[i]; }
}

// =========================================================================
// 3. Launcher: 智能选择 Kernel
// =========================================================================
extern "C" void DeformableAggregationLauncher(
    const void* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const void* sample_location, const void* weights, void* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared, bool is_fp16, cudaStream_t stream) 
{
    size_t out_bytes = batch_size * num_anchors * num_embeds * (is_fp16 ? 2 : 4);
    cudaMemsetAsync(output, 0, out_bytes, stream);

    // 🎯 核心逻辑：分流。如果是 Map 头(pts=300)，原子版更快；Det 头(pts=13)，向量版更快。
    bool use_atomic = (num_pts > 64);

    if (is_fp16) {
        if (use_atomic) {
            int threads = batch_size * num_anchors * num_pts * (num_embeds / 8);
            dfa_atomic_kernel<__half><<<DIVUP(threads, 256), 256, 0, stream>>>(
                (const __half*)mc_ms_feat, spatial_shape, scale_start_index, (const __half*)sample_location, (const __half*)weights,
                (__half*)output, batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        } else {
            int threads = batch_size * num_anchors * (num_embeds / 8);
            dfa_vector_kernel<__half><<<DIVUP(threads, 256), 256, 0, stream>>>(
                (const __half*)mc_ms_feat, spatial_shape, scale_start_index, (const __half*)sample_location, (const __half*)weights,
                (__half*)output, batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        }
    } else {
        // 🎯 即使 TensorRT 把这一层 Fallback 成了 FP32，我们依然有加速逻辑！
        if (use_atomic) {
            int threads = batch_size * num_anchors * num_pts * (num_embeds / 8);
            dfa_atomic_kernel<float><<<DIVUP(threads, 256), 256, 0, stream>>>(
                (const float*)mc_ms_feat, spatial_shape, scale_start_index, (const float*)sample_location, (const float*)weights,
                (float*)output, batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        } else {
            int threads = batch_size * num_anchors * (num_embeds / 8);
            dfa_vector_kernel<float><<<DIVUP(threads, 256), 256, 0, stream>>>(
                (const float*)mc_ms_feat, spatial_shape, scale_start_index, (const float*)sample_location, (const float*)weights,
                (float*)output, batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        }
    }
}
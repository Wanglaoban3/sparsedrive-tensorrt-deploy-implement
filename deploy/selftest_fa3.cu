// wmma 外科级语义验证 (sm_87): 往返 + 单位阵, 解耦 load/store/mma.
// 编译: nvcc -O3 -arch=sm_87 -o selftest_fa3 selftest_fa3.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#include <mma.h>

using namespace nvcuda;

__global__ void probe(const __half* srcA, const __half* srcB,
                      float* rtA, float* rtBcol, float* rtBrow,
                      float* mI_A, float* mI_B, float* mMul) {
  __shared__ __half sA[256], sB[256];
  const int l = threadIdx.x;
  for (int i = l; i < 256; i += 32) {
    sA[i] = srcA[i];
    sB[i] = srcB[i];
  }
  __syncwarp();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fac;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fbr;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fbc;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;

  // 往返(经单位阵): I@B 与 A@I 应还原输入; 同时测 A/B 两种 layout 加载
  __shared__ __half sI[256];
  for (int i = l; i < 256; i += 32)
    sI[i] = ((i >> 4) == (i & 15)) ? __float2half(1.f)
                                   : __float2half(0.f);
  __syncwarp();
  wmma::load_matrix_sync(fa, sI, 16);
  wmma::load_matrix_sync(fbr, sB, 16);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fa, fbr, fc);
  wmma::store_matrix_sync(mI_A, fc, 16, wmma::mem_row_major);  // = B (row 加载)

  wmma::load_matrix_sync(fa, sI, 16);
  wmma::load_matrix_sync(fbc, sB, 16);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fa, fbc, fc);
  wmma::store_matrix_sync(rtBcol, fc, 16, wmma::mem_row_major);  // = B (col 加载)

  wmma::load_matrix_sync(fa, sA, 16);
  wmma::load_matrix_sync(fbr, sI, 16);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fa, fbr, fc);
  wmma::store_matrix_sync(mI_B, fc, 16, wmma::mem_row_major);  // = A (row 加载)

  // 普通 A@B
  wmma::load_matrix_sync(fa, sA, 16);
  wmma::load_matrix_sync(fbr, sB, 16);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fa, fbr, fc);
  wmma::store_matrix_sync(mMul, fc, 16, wmma::mem_row_major);
}

int main() {
  std::vector<__half> A(256), B(256);
  for (int i = 0; i < 256; ++i) {
    A[i] = __float2half(((i >> 4) % 4) * 0.5f + (i & 15) * 0.05f);
    B[i] = __float2half(((i & 15) % 3) * 0.25f - (i >> 4) * 0.1f);
  }
  __half *dA, *dB;
  float *rtA, *rtBc, *rtBr, *mIA, *mIB, *mM;
  cudaMalloc(&dA, 512); cudaMalloc(&dB, 512);
  cudaMalloc(&rtA, 1024); cudaMalloc(&rtBc, 1024); cudaMalloc(&rtBr, 1024);
  cudaMalloc(&mIA, 1024); cudaMalloc(&mIB, 1024); cudaMalloc(&mM, 1024);
  cudaMemcpy(dA, A.data(), 512, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), 512, cudaMemcpyHostToDevice);
  probe<<<1, 32>>>(dA, dB, rtA, rtBc, rtBr, mIA, mIB, mM);
  cudaDeviceSynchronize();
  std::vector<float> a(256), bc(256), br(256), ia(256), ib(256), m(256);
  cudaMemcpy(a.data(), rtA, 1024, cudaMemcpyDeviceToHost);
  cudaMemcpy(bc.data(), rtBc, 1024, cudaMemcpyDeviceToHost);
  cudaMemcpy(br.data(), rtBr, 1024, cudaMemcpyDeviceToHost);
  cudaMemcpy(ia.data(), mIA, 1024, cudaMemcpyDeviceToHost);
  cudaMemcpy(ib.data(), mIB, 1024, cudaMemcpyDeviceToHost);
  cudaMemcpy(m.data(), mM, 1024, cudaMemcpyDeviceToHost);

  auto fA = [&](int i, int j) { return __half2float(A[i * 16 + j]); };
  auto fB = [&](int i, int j) { return __half2float(B[i * 16 + j]); };

  double w;
  printf("== I@B row_major 加载 (期望 D=B) ==\n  ");
  w = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) w = fmax(w, fabs(ia[i * 16 + j] - fB(i, j)));
  printf("maxerr=%.3e %s\n", w, w < 1e-3 ? "OK" : "**FAIL**");

  printf("== I@B col_major 加载 (期望 D=B) ==\n  ");
  w = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) w = fmax(w, fabs(bc[i * 16 + j] - fB(i, j)));
  printf("maxerr=%.3e %s\n", w, w < 1e-3 ? "OK" : "**FAIL**");

  printf("== A row_major 加载 @I (期望 D=A) ==\n  ");

  printf("== mma A=A, B=I (期望 D=A) ==\n  ");
  w = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) w = fmax(w, fabs(ib[i * 16 + j] - fA(i, j)));
  printf("maxerr=%.3e %s\n", w, w < 1e-3 ? "OK" : "**FAIL**");

  printf("== mma A@B (CPU 对拍) ==\n  ");
  w = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      float e = 0;
      for (int k = 0; k < 16; ++k) e += fA(i, k) * fB(k, j);
      w = fmax(w, fabs(m[i * 16 + j] - e));
    }
  printf("maxerr=%.3e %s\n", w, w < 1e-2 ? "OK" : "**FAIL**");
  return 0;
}

// probe_vmm2: Orin iGPU VMM map 完整链路探针 (M8, 带 primary context).
// 逐行 PASS/FAIL: create(fd)/map/setAccess/memset/d2h 验证/export;
// 全部打印真实 driver 返回码.
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const char* cuerr(CUresult r) {
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  return s ? s : "?";
}

int main() {
  CUresult r;
  if (cudaFree(0) != cudaSuccess) { printf("rt ctx fail\n"); return 1; }
  cuInit(0);
  CUdevice dev = 0;
  cuDeviceGet(&dev, 0);
  CUcontext ctx = nullptr;
  r = cuDevicePrimaryCtxRetain(&ctx, dev);
  printf("primary_ctx_retain rc=%d (%s)\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r));
  if (r != CUDA_SUCCESS) return 1;
  r = cuCtxSetCurrent(ctx);
  printf("ctx_set_current rc=%d (%s)\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t gran = 0;
  cuMemGetAllocationGranularity(&gran, &prop,
                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  size_t asz = ((4 << 20) + gran - 1) / gran * gran;

  CUmemGenericAllocationHandle h = {};
  r = cuMemCreate(&h, asz, &prop, 0);
  printf("cuMemCreate rc=%d (%s) asz=%zu\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r), asz);
  if (r != CUDA_SUCCESS) return 1;
  int fd = -1;
  r = cuMemExportToShareableHandle(&fd, h,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                   0);
  printf("export rc=%d (%s) fd=%d\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r), fd);

  CUdeviceptr dptr = 0;
  r = cuMemMap(dptr, asz, 0, h, 0);
  printf("cuMemMap rc=%d (%s) dptr=0x%llx\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r), (unsigned long long)dptr);
  if (r != CUDA_SUCCESS) return 1;
  CUmemAccessDesc acc = {};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = 0;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess(dptr, asz, &acc, 1);
  printf("cuMemSetAccess rc=%d (%s)\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r));

  r = cuMemsetD8(dptr, 0xAB, 256);
  printf("cuMemsetD8 rc=%d (%s)\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r));
  unsigned char host[256] = {};
  r = cuMemcpyDtoH(host, dptr, 256);
  bool all = r == CUDA_SUCCESS;
  for (int i = 0; all && i < 256; ++i) if (host[i] != 0xAB) all = false;
  printf("cuMemcpyDtoH rc=%d (%s) data=%s\n", (int)r,
         r == CUDA_SUCCESS ? "OK" : cuerr(r), all ? "OK" : "BAD");

  // runtime 侧用同一指针 (preproc kernel 将这样用)
  cudaError_t e = cudaMemset((void*)dptr, 0xCD, 256);
  printf("rt cudaMemset on VMM ptr rc=%d (%s)\n", (int)e,
         cudaGetErrorString(e));
  memset(host, 0, sizeof(host));
  e = cudaMemcpy(host, (void*)dptr, 256, cudaMemcpyDeviceToHost);
  all = e == cudaSuccess;
  for (int i = 0; all && i < 256; ++i) if (host[i] != 0xCD) all = false;
  printf("rt cudaMemcpy d2h rc=%d (%s) data=%s\n", (int)e,
         cudaGetErrorString(e), all ? "OK" : "BAD");

  printf("PROBE_DONE\n");
  return 0;
}

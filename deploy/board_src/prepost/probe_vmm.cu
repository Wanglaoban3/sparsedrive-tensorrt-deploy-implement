// probe_vmm: Orin iGPU 跨进程设备内存机制能力探针 (M8).
// 输出逐行 PASS/FAIL: granularity / cuMemCreate(NONE/FD) / fd export /
// cudaIpcGetMemHandle / 跨进程 cudaIpcOpenMemHandle + D2H 数据验证.
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _e = (x); \
  if (_e != cudaSuccess) { printf("  cuda_err %s: %s\n", #x, \
      cudaGetErrorString(_e)); } } while (0)

static size_t align_up(size_t v, size_t g) {
  return (v + g - 1) / g * g;
}

static void vmm_tests() {
  cuInit(0);
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  size_t gran_none = 0, gran_fd = 0;
  CUresult r1 = cuMemGetAllocationGranularity(&gran_none, &prop,
      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  printf("gran_query_none rc=%d gran=%zu\n", (int)r1, gran_none);
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUresult r2 = cuMemGetAllocationGranularity(&gran_fd, &prop,
      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  printf("gran_query_fd rc=%d gran=%zu\n", (int)r2, gran_fd);
  if (gran_none == 0) gran_none = 2 << 20;

  // 1) 无句柄类型 + 对齐尺寸
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  CUmemGenericAllocationHandle h = {};
  size_t asz = align_up(1 << 20, gran_none);
  CUresult r3 = cuMemCreate(&h, asz, &prop, 0);
  printf("cuMemCreate_none aligned(%zu) rc=%d\n", asz, (int)r3);
  if (r3 == CUDA_SUCCESS) cuMemRelease(h);

  // 2) POSIX fd 句柄类型 + 对齐尺寸
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  r3 = cuMemCreate(&h, asz, &prop, 0);
  printf("cuMemCreate_fd aligned(%zu) rc=%d\n", asz, (int)r3);
  if (r3 != CUDA_SUCCESS) return;
  int sh = -1;
  CUresult r4 = cuMemExportToShareableHandle(&sh, h,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  printf("cuMemExportToShareableHandle rc=%d fd=%d\n", (int)r4, sh);
  if (r4 == CUDA_SUCCESS && sh >= 0) close(sh);
  cuMemRelease(h);
}

// 跨进程 IPC: 子进程 cudaMalloc+memset(0xAB)+GetHandle → 管道发 64B;
// 父进程 OpenHandle(flags=0) → D2H 读回验证.
static void ipc_tests() {
  int pp[2];
  if (pipe(pp) != 0) { printf("pipe fail\n"); return; }
  pid_t pid = fork();
  if (pid == 0) {
    close(pp[0]);
    unsigned char* p = nullptr;
    if (cudaMalloc(&p, 256) != cudaSuccess) { _exit(3); }
    CK(cudaMemset(p, 0xAB, 256));
    CK(cudaDeviceSynchronize());
    cudaIpcMemHandle_t h;
    cudaError_t e = cudaIpcGetMemHandle(&h, p);
    if (e != cudaSuccess) {
      printf("ipc_get_child FAIL: %s\n", cudaGetErrorString(e));
      fflush(stdout); _exit(4);
    }
    if (write(pp[1], &h, sizeof(h)) != (ssize_t)sizeof(h)) _exit(5);
    sleep(10);  // 等父进程 open
    _exit(0);
  }
  close(pp[1]);
  cudaIpcMemHandle_t h;
  ssize_t got = 0;
  char* buf = (char*)&h;
  while (got < (ssize_t)sizeof(h)) {
    ssize_t r = read(pp[0], buf + got, sizeof(h) - got);
    if (r <= 0) break;
    got += r;
  }
  if (got != (ssize_t)sizeof(h)) {
    printf("ipc_handle_rx FAIL got=%zd\n", got);
    return;
  }
  void* p = nullptr;
  cudaError_t e = cudaIpcOpenMemHandle(&p, h, 0);
  if (e != cudaSuccess) {
    printf("ipc_open flags=0 FAIL: %s\n", cudaGetErrorString(e));
    e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
    if (e != cudaSuccess) {
      printf("ipc_open lazy FAIL: %s\n", cudaGetErrorString(e));
      return;
    }
    printf("ipc_open lazy OK\n");
  } else {
    printf("ipc_open flags=0 OK\n");
  }
  unsigned char host[256];
  memset(host, 0, sizeof(host));
  e = cudaMemcpy(host, p, 256, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("ipc_d2h FAIL: %s\n", cudaGetErrorString(e));
    return;
  }
  bool all = true;
  for (int i = 0; i < 256; ++i) if (host[i] != 0xAB) all = false;
  printf("ipc_d2h %s (0x%02x)\n", all ? "DATA_OK" : "DATA_BAD", host[0]);
  cudaIpcCloseMemHandle(p);
  waitpid(pid, nullptr, 0);
}

int main() {
  if (cudaFree(0) != cudaSuccess) { printf("ctx fail\n"); return 1; }
  printf("=== vmm ===\n");
  vmm_tests();
  printf("=== ipc ===\n");
  ipc_tests();
  printf("PROBE_DONE\n");
  return 0;
}

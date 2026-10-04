// M1.5 fence 回收测试用 kernel:整槽求和(GPU 侧校验和)。
// 用途:消费进程 acquire 后 CPU 先求和(槽位仍受引用保护),再异步发射
// kernel 求和并 cudaEventRecord;带 fence 时引用挂到事件完成后才 release,
// GPU 和 CPU 两个和必须永远一致;不带 fence 时发布端可在 kernel 执行前
// 覆写槽位 → 两和出现差异 = 复现覆写危害。
// 该 kernel 同时充当"GPU 直接读 registered shm"的带宽探针(M1 风险项)。
#ifndef SP_KERNELS_H_
#define SP_KERNELS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// data 上 n 字节的 64 位折叠和;out 必须是 device 可写 8 字节。
// 返回 kernel 耗时(ms, host 侧 event 计时由调用方做,这里只发 kernel)。
void slot_sum_launch(const uint8_t* data, size_t n, uint64_t* out,
                     void* stream);

// 忙等 ~ns 纳秒(__nanosleep, 真实时间, 与 GPU 时钟门控/DVFS 无关),
// 占住 stream 把后续 kernel 的执行往后推;用于受控实验:race/fence 两模式
// 发同样的延迟+求和,只差释放时机。
void delay_launch(unsigned long long ns, void* stream);

#ifdef __cplusplus
}
#endif

#endif  // SP_KERNELS_H_

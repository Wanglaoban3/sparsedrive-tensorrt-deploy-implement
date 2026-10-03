# Orin X 板端部署环境与启动行为记录

## 硬件

- 设备：智己 IM Orin X 域控制器
- SSH: `root@<BOARD_IP>`（密码经环境变量 `BOARD_PASS` 注入，
  板子地址经 `BOARD_HOST`，见 `deploy/board_*.py` 驱动脚本）
- SoC: NVIDIA Orin-X (sm_87)
- JetPack: 5.1.5
- TensorRT: 8.6.1.2
- CUDA: 11.4

## 交付组合（当前）

- 引擎：`/opt/m0/trt-dev/models/e_T6.engine`（135 MB，INT8 backbone+neck
  QDQ + FP16 head）
- 插件：`/usr/local/lib/libdfaplug_v8.so`（plan+gather 双内核 DFA，
  自分配 workspace，内置 v3 回退）
- 性能：**38.99 ms / 25.6 FPS**，mini mAP **0.4203** / NDS **0.4738**
- **插件配对**：e_T6 按 名字/版本/命名空间 绑定插件 → 换 .so 不用重编
  引擎。v8 与 v3 的 logits 语义逐条一致（ulp 级等价，P3 已逐位验证），
  可互换；更老的 v1 及语义不同的版本**不可**混用（静默算错）。

## 开机行为

开机后系统会**自动重启两次**才稳定进入系统。原因是 Orin X 域控的
启动流程中有个已知的 bootloader/内核初始化时序问题，首次和第二次
启动会在早期阶段自行 reset，第三次才正常引导到用户空间。

**部署注意**：远程 SSH 可用前需等待板子完成两次自动重启并稳定运行。
如果 SSH 连接超时，等 2-3 分钟再试。

## 板端开发环境

### 目录结构

```
/opt/m0/trt-dev/           # 开发根目录 (~2.5GB)
├── models/
│   ├── e_T6.engine        # 交付引擎 (135MB)
│   ├── v5_P1h.onnx        # 交付 ONNX 图 (212MB)
│   └── calib_first.cache  # INT8 校准缓存
├── src/                   # 全部源码 (deploy/ 下有同名正本)
├── include/               # TRT 头文件 (编译插件用)
├── mods/                  # 会话级脚本与 DONE marker
├── repro/
│   └── mini_pipeline*.sh  # mini 81 帧链式推理
└── vec/mini/              # mini 81 帧验证数据 (1.8GB)
    ├── in_00/ ... in_80/  # 每帧输入 bin + manifest.tsv
    └── mini_meta.npz      # 帧序号/token/时间戳
```

### 系统级工具

```
/usr/local/bin/onnx2engine2   # ONNX→TRT 编译工具 (自研 C++, 源码 deploy/onnx2engine.cpp)
/usr/local/bin/run_engine     # 单引擎推理运行器 (deploy/run_engine.cpp)
/usr/local/bin/run_engines    # N 引擎链式 runner (deploy/run_engines.cpp)
/usr/local/lib/libdfaplug_v8.so  # 交付 DFA 插件
/usr/src/tensorrt/bin/trtexec # TRT 自带 profiling 工具
```

### 编译工具链

```bash
# DFA 插件 (nvcc)
nvcc -O3 -shared -Xcompiler -fPIC src/dfaplug_v8.cu \
  -I include -arch=sm_87 -lnvinfer \
  -o /usr/local/lib/libdfaplug_v8.so

# ONNX→TRT 编译工具 (g++, 注意 -ldl)
g++ -O2 -o /usr/local/bin/onnx2engine2 src/onnx2engine.cpp \
  -I include -I /usr/local/cuda/include \
  -L /usr/local/cuda/lib64 -lnvinfer -lnvonnxparser -lcudart -ldl

# 引擎编译 (INT8 需要校准缓存)
onnx2engine2 models/v5_P1h.onnx models/e_T6.engine \
  --int8 --fp16 --ws-mb 2048 \
  --plugins /usr/local/lib/libdfaplug_v8.so
```

### 推理

```bash
# 单引擎
run_engine models/e_T6.engine /usr/local/lib/libdfaplug_v8.so \
  <inputs_dir> [--iters N] [--warmup N] [--dump <dir>]

# 多引擎链式 (按张量名 D2D 零拷贝接线)
run_engines e_bb.engine e_det.engine /usr/local/lib/libdfaplug_v8.so \
  vec/mini/in_00 --iters 100 --warmup 10
```

inputs_dir 结构: manifest.tsv + 每个 tensor 的 .bin 文件
manifest 格式: `name\tdtype\tshape\tfile`

### mini 精度验证流程

```bash
# 1. 板上链式推理 81 帧 (v8 交付流水线)
bash repro/mini_pipeline_v8.sh   # 约 3-4 分钟

# 2. 输出拉回本地，运行 mAP 评估
python deploy/eval_t6_mini_v8.py
```

## 已知限制

1. **NVRTC JIT 损坏**：TRT 8.6.1.2 的 NVRTC 对 fp32 融合 kernel
   编译必失败（`nvrtc_compile.cpp:940 CHECK(false)`）。
   fp16 JIT 正常。绕开方式：确保计算路径走 fp16。

2. **Myelin 退化融合区**：decoder 融合区合计 ~12.3ms 跑在
   ForeignNode 回退上（det 侧是 SSA 注意力物化的真实访存，
   map/quality 侧是大图规划失败）。升级 TRT 可能解决；
   P4/P5 已实测引擎拆分/手写 FlashAttention 插件均无法绕开
   （见 OPTIMIZATION_SUMMARY 第七/八轮）。

3. **磁盘**：/opt/m0 分区 26GB；根分区 / 只有 4.2GB，
   避免在根分区写大文件。

4. **插件版本配对**：e_T6 + v8/v3（语义等价可互换）。
   其他老版本插件 weights 语义不同，混用会静默出错。

## 部署注意事项

1. **场景边界重置**：换场景时（时间差 >2s）必须重置时序状态
   （prev_* 全部清零，time_interval 设 0.5，instance_t_matrix 设单位
   矩阵）。引擎图内的 dt 防护无效，必须在宿主侧处理。

2. **mini 验证数据准备**：用 `deploy/prep_mini_inputs.py` 生成
   81 帧引擎输入。场景边界帧（帧 40）会自动写零状态。

3. **mini 评估**：用 `deploy/eval_t6_mini_v8.py` 在本地解码+评估，
   需要 nuScenes mini 数据集 + deploy/artifacts 下的输出。

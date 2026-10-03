中文版 | [English](README.md)

# SparseDrive TensorRT 部署与加速指南

## 📖 简介

本项目基于官方 [SparseDrive](https://github.com/swc-17/SparseDrive.git)
源码，将端到端 SparseDrive 模型部署到 TensorRT，覆盖**两类硬件目标**：

1. **工作站 GPU（RTX 3090）**——FP16 多引擎流水线（吞吐 ~7.4×，规划
   指标与 PyTorch 对齐，见下文）；
2. **车规域控（NVIDIA Orin X 域控制器）**——量产形态交付：单引擎
   **INT8 backbone/neck + FP16 head** 混合精度 + 自研 **plan+gather
   双内核 DFA 插件** + 八轮优化日志——**82.6 ms → 38.99 ms（2.12×）**，
   感知精度持平。

全部代码面向 Linux（板端 JetPack 5.1.5 / TensorRT 8.6.1.2 / CUDA 11.4 /
sm_87；导出端 CUDA 11.6 构建环境）。所有改动都以 nuScenes 闭环评估做
精度门禁，通过才接受。

## 🚗 Jetson Orin 域控交付（核心结果）

| 项目 | 值 |
|---|---|
| 引擎 | 单引擎 `e_T6.engine`（backbone+neck INT8 QDQ，head FP16，DFA 权重 FP16）|
| 插件 | `libdfaplug_v8.so`——DeformableAggregation plan+gather 双内核（自分配 workspace，内置 v3 回退）|
| 延迟 | **38.99 ms / 25.6 FPS**（100 次 iter 均值；INT8 时代基线 82.6 ms / 12.1 FPS）|
| 精度 | mini mAP **0.4203** / NDS **0.4738**（FP32 引擎基线 0.4178——持平）|

### 八轮优化日志

| 轮次 | 改动 | 延迟变化 |
|---|---|---|
| 1 | kps 6 维广播 MatMul → 3-D GEMM 重写 | **−19.6 ms** |
| 2 | DFA 插件 `__half2` + FP16 IO | **−7.5 ms** |
| 3 | softmax 折入 DFA（logits 直通导出）| **−4.7 ms** |
| 4 | head 全 FP16 图（P1）| 持平——TRT 8.6 退化融合区硬墙 |
| 5 | 7 项图/内核实验（QKV 合并、DFA int8 输入等）| 无收益（已记录）|
| 6 | DFA **plan+gather 双内核**（v8）| **−6.3 ms** |
| 7 | 引擎拆分 bb/head/det/rest（P4）| **负收益**——拆分边界税 > 融合收益 |
| 8 | det 注意力 FlashAttention 插件（P5）| **负收益**——TRT 融合链 ~5 TFLOPS，手写 wmma ~1.6 TFLOPS |

负结果与正结果同等严谨地归档（实测收割上限、probe/replay 数值方法论、
根因定位）——见
[docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md)，它同时是
一份 TRT 8.6 板端部署的踩坑手册。

### 板端部署

前置：JetPack 5.1.5、TensorRT 8.6.1.2、CUDA 11.4（sm_87）。

```bash
# 1. 导出交付 ONNX 图（INT8 QDQ + FP16 head + logits-DFA）
python deploy/export_v5.py --checkpoint ckpt/sparsedrive_stage2.pth \
    --protect-groups det_head_output --attn-fp16 \
    --with-ref work_dirs/sparsedrive_small_stage2/mtq_v5_ref.npz \
    --out work_dirs/sparsedrive_small_stage2/v5_P1h.onnx

# 2. 编译板端引擎编译器并编译（INT8 需要校准缓存）
g++ -O2 -o onnx2engine2 deploy/onnx2engine.cpp -I $TRT_INCLUDE_DIR \
    -L $CUDA_LIB -lnvinfer -lnvonnxparser -lcudart -ldl
./onnx2engine2 v5_P1h.onnx e_T6.engine --int8 --fp16 --ws-mb 2048 \
    --plugins libdfaplug_v8.so

# 3. 编译插件（nvcc, sm_87）
nvcc -O3 -shared -Xcompiler -fPIC -o libdfaplug_v8.so deploy/dfaplug_v8.cu \
    -I $TRT_INCLUDE_DIR -lnvinfer

# 4. 运行（单引擎；多引擎用 run_engines 按 D2D 零拷贝链式跑）
g++ -O2 -o run_engine deploy/run_engine.cpp -I $TRT_INCLUDE_DIR \
    -L $CUDA_LIB -lnvinfer -lcudart -ldl
./run_engine e_T6.engine libdfaplug_v8.so <inputs_dir> [--iters 100 --warmup 10]
```

板端环境（目录结构、开机行为、已知固件限制、mini 81 帧闭环精度门禁，
经 `repro/mini_pipeline_v8.sh` + `deploy/eval_t6_mini_v8.py`）：
[docs/BOARD_ENV.md](docs/BOARD_ENV.md)。

> 引擎在反序列化时按插件**名字/版本/命名空间**动态绑定，因此 `.so`
> 可以不重编引擎直接热替换（第 6-8 轮全程用该机制做模块级迭代，
> 从未整引擎重编）。

## 🖥️ 工作站流水线（RTX 3090，FP16 多引擎）

### 1. 环境准备与插件编译

#### 创建虚拟环境
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

#### 安装依赖包
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

#### 编译 deformable_aggregation CUDA 算子
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

#### 准备数据
下载 [NuScenes 数据集](https://www.nuscenes.org/nuscenes#download) 和 CAN bus 扩展包，将 CAN bus 扩展包放入 `/path/to/nuscenes` 目录下，并创建软链接。
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

打包数据集的元信息和标签，并在 `data/infos` 目录下生成所需的 pkl 文件。`data_converter` 中同时生成 `map_annos`，默认 `roi_size` 为 `(30, 60)`，如需其他范围可修改 `tools/data_converter/nuscenes_converter.py`。
```bash
sh scripts/create_data.sh
```

#### 下载预训练权重
下载官方模型权重，创建 `ckpt` 目录，并将权重文件放入其中。
* 权重下载链接：[sparsedrive_stage2.pth](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth)

**【重要】编译 TensorRT 自定义插件：**
模型包含 Deformable Aggregation 等自定义算子，需提前编译 C++ 插件：
```bash
cd projects/trt_plugin
mkdir build && cd build
cmake ..
make -j8
# 编译成功后，将在当前目录生成 libSparseDrivePlugin.so
```

### 2. 导出 ONNX 模型
真实自动驾驶系统中**感知模块**和**规控模块**往往运行在不同频率，工程上将感知头与规控头剥离、分别导出独立引擎。
时序模型在初始帧（无历史特征）与后续帧（有历史特征）的图结构不同，也分别导出。

**导出感知模块 (Det & Map)：**
```bash
# 将同时导出起始帧 (sparsedrive_multihead_first.onnx) 和后续帧 (sparsedrive_multihead.onnx)
python tools/export_onnx_det_map.py \
    --config projects/configs/sparsedrive_small_stage2.py \
    --checkpoint ckpt/sparsedrive_stage2.pth \
    --out work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx
```

**导出规控模块 (Motion & Planning)：**
```bash
# 将同时导出起始帧和后续帧的 Motion & Plan 模型
python tools/export_onnx_motion.py \
    --config projects/configs/sparsedrive_small_stage2.py \
    --checkpoint ckpt/sparsedrive_stage2.pth \
    --out work_dirs/sparsedrive_small_stage2/motion_plan_engine.onnx
```

### 3. 编译 TensorRT Engine (ONNX -> TRT)
*注：请确保 `onnx2trt.py` 脚本内部已正确加载 `libSparseDrivePlugin.so`。*

```bash
# 编译感知模块
python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx --save work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine

python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.onnx --save work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine

# 编译规控模块
python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/motion_plan_engine.onnx --save work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine

python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.onnx --save work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine
```

### 4. 精度评估 (Evaluation)
执行以下命令进行端到端闭环测试：
```bash
python test_trt.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth \
    --engine_perc_init work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine \
    --engine_perc_temp work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine \
    --engine_mo_init work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine \
    --engine_mo_temp work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine
```

**📊 性能对比报告 (NVIDIA RTX 3090)**

#### 端到端核心指标对比 (End-to-End Performance)
| Method | NDS | AMOTA | minADE (m)* | L2 (m) Avg | Col. (%) Avg | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SparseDrive-S (Official Pytorch)** | **0.525** | **0.386** | **0.620** | **0.610** | 0.100 | 4.8 |
| **SparseDrive-S (TRT FP16)** | 0.520 | 0.370 | 0.648 | 0.612 | **0.092** | **35.7** |

> **注：** `minADE` 采用 Car 类别指标。TRT 引擎在感知与规划核心指标（L2 误差仅差 0.002m）高度对齐的前提下，**平均碰撞率 (Col. Avg)** 更优，推理吞吐（FPS）提升 **~7.4×**。

#### 规划指标详细对比 (Detailed Planning Metrics)
| Method | L2 1s | L2 2s | L2 3s | L2 Avg | Col. 1s | Col. 2s | Col. 3s | Col. Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official (Official Pytorch)** | 0.300 | **0.580** | **0.950** | **0.610** | **0.010%** | **0.050%** | 0.230% | 0.100% |
| **TRT FP16** | **0.299** | 0.581 | 0.957 | 0.612 | **0.010%** | 0.054% | **0.212%** | **0.092%** |

### 5. 推理速度测试 (Latency Profile)
Python 脚本做宏观 FPS 测试：
```bash
python fps.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --mode trt
```

算子级微观耗时用 `trtexec` 生成 Profiling 报告：
```bash
# 测试感知模块 (Det & Map)
trtexec --loadEngine=work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine \
        --plugins=projects/trt_plugin/build/libSparseDrivePlugin.so \
        --dumpProfile --iterations=100 > map_det_inference.log

# 测试规控模块 (Motion & Plan)
trtexec --loadEngine=work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine \
        --plugins=projects/trt_plugin/build/libSparseDrivePlugin.so \
        --dumpProfile --iterations=100 > motion_inference.log
```

## 🗂️ 仓库结构

```
deploy/                       部署与优化工具链 (详见 deploy/README.md)
  export_v5.py                  INT8+FP16 交付图导出 (logits-DFA)
  make_v5_p2h.py / split_*.py   图手术 (P1 head 全 FP16、P4 引擎拆分)
  dfaplug_v8.cu                 交付版 DFA 插件 (plan+gather 双内核)
  dfaplug_v3.cu                 回退内核 (语义逐条一致, ulp 级等价)
  run_engine.cpp / run_engines.cpp / onnx2engine.cpp   板端 C++ 工具
  ssa_surgery2.py / _fa_part.cu / selftest_fa*.cu       P5 FlashAttention (负结果留档)
  board_*.py                    paramiko 板端驱动 (环境变量 BOARD_HOST / BOARD_PASS)
  eval_t6_mini_v8.py            mini 81 帧闭环精度门禁
docs/
  OPTIMIZATION_SUMMARY.md     ★ 八轮完整日志 + 踩坑手册 (先读这个)
  BOARD_ENV.md                  板端环境、工具链、已知限制
  quick_start.md                工作站快速开始 (精简版)
repro/                         板端 mini 流水线脚本 (bash)
projects/ tools/ scripts/      SparseDrive 上游源码 + 导出期改造
```

## 📍 后续方向

- [x] 车规 SoC backbone/neck INT8 PTQ 流水线（本仓库 Orin 交付）
- [ ] 帧间流水线（第 N 帧 backbone 与第 N−1 帧 decoder 重叠；吞吐 ~1.7×，不动引擎）
- [ ] TensorRT 升级验证——~12 ms Myelin 退化融合区与 det 注意力物化链均为版本限制，TRT 9/10 上预期大幅消解
- [ ] 纯 C++ 极速推理部署流水线
- [ ] ROS2 集成

## 📚 技术沉淀与避坑指南

- [TECH_DETAILS.md](TECH_DETAILS_CN.md)——工作站时代深潜：时序对齐的矩阵化重构、ONNX 控制流消除、CUDA 算子极致优化。
- [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md)——Orin 时代日志：每一项接受/拒绝的优化都有实测与方法论（probe/replay 数值对拍、插件热替换绑定），以及踩坑手册（NVRTC JIT 损坏、workspace 烤进 plan、插件版本配对等）。

欢迎交流探讨！

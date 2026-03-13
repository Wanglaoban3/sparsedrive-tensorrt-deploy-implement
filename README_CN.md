中文版 | [English](README.md)

# SparseDrive TensorRT 部署与加速指南

## 📖 简介
本项目参考官方发布的 [SparseDrive](https://github.com/swc-17/SparseDrive.git) 源码，完成了端到端 SparseDrive 模型的 FP16 精度 ONNX 导出，以及基于 TensorRT 的 Engine 编译与极致优化。在保持核心规划指标精度的前提下，大幅提升了推理运行效率，旨在推动端到端自动驾驶社区的落地进程。

## 🚀 快速开始 (How to Use)

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
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)
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

打包数据集的元信息和标签，并在 `data/infos` 目录下生成所需的 pkl 文件。请注意，我们还在 `data_converter` 中生成了 `map_annos`，默认的 `roi_size` 为 `(30, 60)`，如果你想使用不同的范围，可以在 `tools/data_converter/nuscenes_converter.py` 中修改 `roi_size`。
```bash
sh scripts/create_data.sh
```

#### 下载预训练权重
下载官方模型权重，创建 `ckpt` 目录，并将权重文件放入其中。
* 权重下载链接：[sparsedrive_stage2.pth](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth)

**【重要】编译 TensorRT 自定义插件：**
由于模型中包含了 Deformable Aggregation 等自定义算子，需提前编译 C++ 插件：
```bash
cd projects/trt_plugin
mkdir build && cd build
cmake ..
make -j8
# 编译成功后，将在当前目录生成 libSparseDrivePlugin.so
```

### 2. 导出 ONNX 模型
考虑到在真实的自动驾驶系统中，**感知模块**和**规控模块**往往运行在不同的频率上，因此我在工程上将感知头和规控头剥离，分别导出为独立的 Engine。
此外，由于时序模型在初始帧（无历史特征）和后续帧（有历史特征）的图结构不同，我也分别做了针对性导出。

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
*注：请确保你的 `onnx2trt.py` 脚本内部已正确加载了 `libSparseDrivePlugin.so`。*

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
| **SparseDrive-S (Official)** | **0.525** | **0.386** | **0.620** | **0.610** | 0.100 | 4.8 |
| **SparseDrive-S (Ours TRT)** | 0.520 | 0.370 | 0.648 | 0.612 | **0.092** | **35.7** |

> **注：** `minADE` 采用的是 Car 类别的指标。TRT 引擎在保证感知和规划核心指标（L2 误差仅相差 0.002m）高度对齐的前提下，在**平均碰撞率 (Col. Avg)** 上取得了更优的表现，同时推理吞吐量（FPS）实现了 **~7.4x** 的巨大飞跃。

#### 规划指标详细对比 (Detailed Planning Metrics)
| Method | L2 1s | L2 2s | L2 3s | L2 Avg | Col. 1s | Col. 2s | Col. 3s | Col. Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official** | 0.300 | **0.580** | **0.950** | **0.610** | **0.010%** | **0.050%** | 0.230% | 0.100% |
| **Ours TRT** | **0.299** | 0.581 | 0.957 | 0.612 | **0.010%** | 0.054% | **0.212%** | **0.092%** |

> **数据解析：**
> * **L2 误差（L2 Distance）：** TRT 版本与官方原版紧紧咬合。短时（1s）预测甚至达到了 0.299m 的优秀水准。
> * **碰撞率（Collision Rate）：** TRT 部署展现出了极高的安全性。在 1s 碰撞率与官方持平（0.010%）的基础上，3s 维度的长时碰撞率显著下降（从 0.230% 降至 0.212%），这说明我们的 FP16 引擎和算子融合在时序特征的处理上非常鲁棒。

### 5. 推理速度测试 (Latency Profile)
可以使用 Python 脚本进行宏观的 FPS 测试：
```bash
python fps.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --mode trt
```

若需分析各算子的微观耗时，推荐使用 `trtexec` 生成 Profiling 报告：
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

### 6. 未来规划 (To-Do List)
- [ ] 完成 SparseDrive 纯 C++ 极速推理部署流水线
- [ ] 研发 SparseDrive 的 QAT (Quantization-Aware Training) INT8 量化工具链
- [ ] 完成 SparseDrive 与 ROS2 自动驾驶系统的工程集成
- [ ] 针对长时序队列的显存和访存进行更深度的 Kernel 级优化

### 7. 技术沉淀与避坑指南
在算子融合、ONNX 控制流消除（If 节点剪枝）、动态维度支持以及 C++ 插件编写等优化过程中的经验与心得，详见 [TECH_DETAILS.md](TECH_DETAILS.md) 文档。欢迎交流探讨！
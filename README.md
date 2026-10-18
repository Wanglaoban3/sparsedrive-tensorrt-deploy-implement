# SparseDrive TensorRT 车端部署（NVIDIA Orin X）

English: see docs/ — end-to-end SparseDrive on NVIDIA Orin X, 82.6→39.1 ms closed-loop with production hardening.

基于上游 [SparseDrive](https://github.com/swc-17/SparseDrive)（CVPR 2024）
端到端自动驾驶模型（检测 + 在线建图 + 运动预测 + 规划），本项目补齐了
官方没有的**训练之后的全链路**——ONNX 导出 → 量化 → TensorRT 引擎 →
车规域控芯片 **NVIDIA Orin X** 闭环部署 → 量产化运维，同时完整保留上游
训练栈：**一个仓库覆盖训练 → 量化 → 部署全流程**。闭环链延迟
**82.6 → 39.11 ms（2.11×）**，精度与 FP32 基线打平。所有精度结论都是
**闭环口径**——三引擎链在板上按帧递推跑完 nuScenes mini 81 帧，任何改动
过端到端 decode 门禁才算数。

## ✨ 主要贡献

**① 量化全工具链（[deploy/engine/](deploy/engine)）——从 ckpt 到量化
ONNX 的命令级流水线，全部是官方仓库没有的东西**

- **ONNX 导出（官方零部署出口，我们从零补齐）**：端到端时序模型静态图化，
  附带一组导出手术（[docs/onnx_export_surgery.md](docs/onnx_export_surgery.md)）：
  跨帧历史特征对齐的 Slice/Concat 树重写为 mask/swap 矩阵乘（顺带修掉
  trtexec OOM）、kps 头 6-D broadcast MatMul 重写为 3-D GEMM（**-19.6 ms**）、
  softmax 折叠进自定义算子以 raw-logits 导出（**-4.7 ms**）、时序变换
  （t_matrix/time_interval）收进图内实现运行时零 D2H 同步
- **图折叠优化**：Conv+BN 预折叠 → 常量折叠 → QDQ 权重折叠为 int8
  initializer（[qdq_onnx_rewrite.py](deploy/engine/qdq_onnx_rewrite.py)，
  时序图 135 条折叠链，ORT 重放逐位等价验证）
- **量化损失自动分析**（"哪层不能量化"由数据回答，不靠猜）：
  [ptq_sensitivity.py](deploy/engine/ptq_sensitivity.py)（leave-one-out
  逐层敏感度 + skip-topK 曲线自动选 K）→
  [group_sensitivity.py](deploy/engine/group_sensitivity.py)（组级任务
  指标 mAP/NDS 口径）→ task_sentinel / render_ptq_report（门禁哨兵 +
  自动报告）
- **PTQ + QAT 量化脚本与示例**：PTQ = ModelOpt MTQ（16 样本标定，FP32
  ckpt 直出 INT8 QDQ）；[qat.py](deploy/engine/qat.py)（QAT 短程微调 +
  自动 skip-list）；[flashmha_qkv.py](deploy/engine/flashmha_qkv.py)
  （FlashMHA 打包 QKV 参数模块化手术，量化覆盖 112→133 模块）；含
  smoke_test 冒烟示例与
  [ONNX_QUANT_STRATEGY.md](deploy/engine/ONNX_QUANT_STRATEGY.md) 策略文档
- **结论可复现**：量化数学本身只伤 map **-0.4 pt**（0.7473 vs FP32
  0.7512，与引擎编译损伤分离定案）；QAT 零收益的负结果同样留档

**② 模型级部署优化——CUDA 算子与图（[deploy/board_src/](deploy/board_src)）**

- **DFA 插件三代演进**（v3 half2 → v8 双核 → v11 gather 向量化：48B/anchor
  去重 + warp 级 8ch/lane uint4 tap + 自适应 split，模块级 **1.18×**，
  状态显存 5× 优化）；插件按名字/版本运行时绑定，**换 .so 不用重编引擎**
- **前处理 CUDA 融合核**：NV12→RGB → BICUBIC 缩放（Pillow 位级同构定点
  管线）→ crop + 归一化，三核串联直出引擎 img binding
  （[preproc.cu](deploy/board_src/prepost/preproc.cu)）——配合帧数据
  映射内存零拷贝直读，图像进引擎前不落一次主机拷贝
- **精度护栏**：`onnx2engine --f32-names` 把递归敏感头强制回 FP32
  （层名单由 [_gen_f32_names.py](deploy/engine/_gen_f32_names.py) 从输出
  反向遍历生成），救回单引擎被 fp16 tactic 吃掉的 ~10 pt map
- **负结果同权重留档**：wmma FlashAttention 插件实测负收益（先量"可收割
  上限"再动手的方法论见 docs/OPTIMIZATION_SUMMARY.md），21 轮优化每轮
  含归因证据链

**③ 部署调度链路——零拷贝 + CUDA 图 + 多进程流水线（[deploy/board_src/](deploy/board_src)）**

- **引擎间 D2D 零拷贝**：拆分链边界张量（col_feats [1,89760,256]）不落
  主机内存，unified memory 直通
  （[run_engine2](deploy/board_src/runner/run_engine2.cpp)）
- **CUDA graph 捕获**：整引擎逐 kernel CPU 发射收成一张图，吞吐
  23.9→27.5 fps（**+15.5%**）；双缓冲配双 execution context，19 路 D2H
  完全出关键路径
- **跨进程零拷贝总线**：mapped-shm 环形总线（自研 BusLock 解决 glibc
  robust mutex 属主死亡移交失效）+ seqlock 单写多读结果信箱；Orin iGPU
  上 CUDA IPC/VMM 全不可用是探针定案的平台事实，mapped-shm 是唯一路径
- **多进程流水线**：帧发布 → 闭环节点 → 事件触发器三进程解耦，背压/
  租约/跳序（latest-wins）语义齐全；"双流重叠实测负收益"（iGPU 无 SM
  余量）的结论同样留档

**④ 量产化工程外壳——watchdog 到 systemd 的车规硬化（[deploy/systemd/](deploy/systemd) + [deploy/ops/](deploy/ops)）**

分级 watchdog（弃帧续跑 → 二级熔断）、systemd 熔断拉起、开机自检
（金标容差 + 四 md5 指纹）、1Hz 遥测、shm/UDS 权限收紧、14 规则事件
触发器（CPU 10Hz 第三类进程，规则 .so 热接入，零关键链改动）、故障注入
FT 套件、8h 满负荷浸泡 4/4 PASS——这套外壳不含任何 SparseDrive 语义，
换模型直接搬。

## 📊 指标总表

| 指标 | 值 | 口径 |
|---|---|---|
| 闭环链延迟 | **39.11 ms**（起点 82.6 ms，2.11×） | bb2+hd 拆分链 + v11 插件，v8 口径 39.86 |
| det mAP / NDS | **0.4157 / 0.4724** | mini 81 帧闭环，v8 对照 0.4160/0.4729 |
| map mAP | **0.7462** | FP32 PyTorch 0.7512 |
| 量化损伤 | map **-0.4 pt**（B2a 0.7473） | 量化数学 vs 引擎编译损伤已分离定案 |
| EPA(car/ped) / L2 / col | 0.6039/0.5080 / 0.7398 / 0.161% | M-PROD Phase A 闭环（规划侧） |
| 事件触发器 | 14 规则 .so，CPU 10Hz，零关键链改动 | M9a parity 951 tick 集合全等 |
| 优化日志 | 21 轮，负结果同权重留档 | [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) |

单引擎回退组合（`e_T6` + v8 插件，38.99 ms / det 0.4203 / NDS 0.4738 /
map 0.6471）完整保留，随时可切。

## 🔁 全链路技术流程

```
【训练】上游 SparseDrive 训练栈（scripts/ 训练壳，原样保留）
   │  ckpt (FP32)
   ▼
【导出】deploy/engine/export_*.py —— ONNX 导出手术（官方缺失的一步）
   │    · 历史特征对齐：Slice/Concat 树 → mask/swap 矩阵乘（修 trtexec OOM）
   │    · kps 头 6-D broadcast MatMul → 3-D GEMM（-19.6 ms）
   │    · softmax 折叠进 DFA 自定义算子，raw-logits 导出（-4.7 ms）
   │    · 时序变换进图（t_matrix/time_interval），零 D2H 同步
   ▼
【量化】deploy/engine/ —— PTQ 主线 + 损失自动分析
   │    ptq_sensitivity（leave-one-out 敏感度 + skip-topK 自动选 K）
   │      → qat.py（可选 QAT + 自动 skip-list）
   │      → export_quant_onnx（fake-quant → QDQ）
   │      → qdq_onnx_rewrite（权重折叠 int8 initializer，135 条折叠链）
   │    量化数学损伤 map 仅 -0.4 pt（B2a 定案，与编译损伤分离）
   ▼
【引擎】deploy/board_src/runner/onnx2engine（--int8 / --fp16 / --f32-names 护栏）
   │    e_bb2（backbone/neck INT8）· e_hd（head FP16）· e_mp（motion/planning FP16 时序图）
   ▼
【算子】deploy/board_src/ —— CUDA 自定义算子与融合核
   │    · DFA gather 插件（softmax 折叠的落点，三代演进至 v11 向量化）
   │    · 前处理三核串联：NV12→RGB → BICUBIC 缩放（Pillow 位级同构定点）
   │      → crop+归一化，直出引擎 img binding（preproc.cu）
   ▼
【闭环】板上三进程 systemd 常驻（NVIDIA Orin X）—— 单帧数据生命周期全程 GPU：
   │    帧零拷贝映射 ─► GPU 直读（无 H2D）─► CUDA 前处理 ─► 三引擎推理
   │      （bb2 ─D2D 零拷贝─► hd ─► mp，CUDA graph + 双缓冲）
   │      ─► 时序状态驻留 GPU（next_* 回灌 prev_*，跨帧外旋不落主机）
   │      ─► 输出一次 D2H ─► 解码 ─► seqlock 结果信箱发布
   │    进程拓扑：sp-filesrc ── shm 环形总线 ──► sp-modelnode ──► sp-trigger（10Hz 规则）
   ▼
【运维】分级 watchdog · systemd 熔断拉起 · 开机自检（金标+指纹）· 1Hz 遥测
        shm/UDS 权限收紧 · 故障注入 FT 套件 · 8h 满负荷浸泡
```

## 🏗 架构

三引擎闭环数据流（板上每帧递推，状态跨帧外旋）：

```
6×camera ──► e_bb2 (INT8 backbone/neck)
                │  边界张量 /Reshape_9_output_0  col_feats [1,89760,256]
                ▼  D2D 零拷贝（run_engine2）
            e_hd (FP16 head + libdfaplug_v11.so DFA gather 向量化)
                │  感知输出：det (900×10) / map / instance 状态 (递归反馈)
                ▼
            e_mp (FP16 motion/planning，吃感知输出 + 9 张外旋历史状态；
                  场景首帧零状态复位)
                │  motion_cls/reg + plan_cls/reg + final_plan
                ▼
        sp_result_m3 信箱（seqlock 单写多读）──► sp_trigger（10Hz 规则评估）
                                            ──► sp_resultmon / 下游
```

常驻拓扑 = 三个 systemd 单元：`sp-filesrc@m3`（帧发布源）→
`sp-modelnode@m3`（三引擎闭环节点，watchdog/selftest/遥测内建）→
`sp-trigger@m3`（事件触发器，只读信箱）。模型结构见
`resources/overview.png`（上游图）。

节点内单帧数据**全程驻留 GPU**：帧以映射内存零拷贝到达（GPU 直读，无
H2D 拷贝）→ CUDA 前处理核 → 三引擎推理（引擎间 D2D 零拷贝）→ 时序状态
跨帧外旋不出 GPU → 输出一次 D2H 后解码、经信箱发布。

## 🧩 模块化与迁移

按"模型无关的外壳 + 模型相关的内核"切分，迁移到新模型/新板卡时：

**直接搬（模型无关）**：

| 组件 | 位置 | 说明 |
|---|---|---|
| 板端运行时库 | `deploy/board_src/prepost/`（sp_bus 环形总线/BusLock、sp_result seqlock 信箱、sp_safety 异常探测、sp_watch watchdog） | 纯 C++/CUDA，不含任何 SparseDrive 语义 |
| 引擎工具 | `deploy/board_src/runner/`（onnx2engine、run_engines、run_engine2） | 任意 TRT 引擎的构建与 N 引擎链式 runner |
| 量产外壳 | `deploy/systemd/` 单元模板 + `deploy/ops/`（安装/故障注入/浸泡）+ `deploy/spdeploy.py` CLI | 换单元名即用 |
| 事件触发器框架 | `deploy/board_src/prepost/`（sp_rule/sp_ruleload/sp_quota ABI + rules/） | 规则 = .so 插件，阈值外置，开发指南见 docs/M9A_PLUGIN_GUIDE.md |
| 量化分析工具 | `deploy/engine/`（ptq_sensitivity、group_sensitivity、render_ptq_report 等） | 吃任意 nn.Module，输出敏感度排名与报告 |
| QDQ 图重写 | `deploy/engine/qdq_onnx_rewrite.py` | 任意 QDQ ONNX 的权重折叠与标准化 |
| 板驱动 | `deploy/board/`（paramiko 驱动 + 12 个 sp-*.sh 薄壳） | 凭据只运行时注入 |

**需要重做（模型相关，但方法论可复用）**：

- **ONNX 导出手术**——每张图的病灶不同；
  [docs/onnx_export_surgery.md](docs/onnx_export_surgery.md) 给的是
  "怎么找、怎么改、怎么验"的剧本
- **自定义算子插件**——算子语义绑定模型（本项目是 DFA）；nvcc +
  TRT plugin ABI 的框架与三代演进路径直接沿用
- **--f32-names 精度护栏**——敏感头因模型而异；
  [_gen_f32_names.py](deploy/engine/_gen_f32_names.py) 的输出反向遍历
  方法通用
- **前后处理 C++**——decode/发布口径绑定任务头
  （[deploy/board_src/prepost/](deploy/board_src/prepost) 的 postproc.h
  可当模板）

**迁移四步**：导出（手术剧本）→ 量化（敏感度分析自动选层）→ 插件/
护栏（只重做模型绑定件）→ 接入闭环外壳（零改动）。

## 🚀 快速上手

```bash
git clone <this-repo>
cd sparsedrive-tensorrt-deploy-implement
pip install -r requirement.txt          # 上游 SparseDrive 环境（导出/评估用）

# 板端凭据只运行时注入，不落盘不入库：
BOARD_HOST=<board-ip> BOARD_PASS=<pass> scripts/sp-install.sh   # 安装+健康断言
BOARD_HOST=<board-ip> BOARD_PASS=<pass> scripts/sp-status.sh    # 只读全景
BOARD_HOST=<board-ip> BOARD_PASS=<pass> scripts/sp-gate.sh      # 精度门禁
```

统一入口 `python deploy/spdeploy.py <verb>`（13 个 verb，`--help` 看全表）；
`scripts/sp-*.sh` 是带凭据断言的薄壳。引擎构建、量化、数据准备的完整
复刻步骤见 [docs/FULL_CHAIN_REPRO.md](docs/FULL_CHAIN_REPRO.md)。

## 📦 交付物清单（板上 /usr/local/{bin,lib}）

| 交付物 | 指纹（板上 md5） |
|---|---|
| `e_bb2.engine`（INT8 backbone/neck） | `f375467e…` |
| `e_hd.engine`（FP16 head + DFA） | `8c3a6257…` |
| `e_mp.engine`（FP16 motion/planning，26.57 MB） | `5e4a48ab…` |
| `libdfaplug_v11.so`（gate build `f46f5bf7…`） | nvcc 时间戳致每次编译 md5 不同，以功能/自检为准 |
| `libdfaplug_v8.so`（回退） | `55df31bc…` |
| `run_engine2`（D2D 零拷贝链 runner） | `60dff83b…` |
| `run_engines`（N 引擎泛化 runner） | `0b4da654…` |
| `sp_modelnode`（闭环节点）+ `sp_filesrc` + `sp_trigger` + 14 规则 .so | 自检四指纹核对（`--selftest`） |

回退单引擎 `e_T6.engine`（`7fbcdac1…`）完整保留。插件与引擎按
名字/版本/命名空间运行时绑定——**换 .so 不用重编引擎**。

## 🗺 仓库地图

| 目录 | 职责 |
|---|---|
| `scripts/` | 上游训练壳 5 个 + `sp-*.sh` 部署薄壳 12 个 |
| `repro/` | 板端链脚本（`mini_pipeline*.sh` 等逐帧状态旋转） |
| `deploy/` | 全部部署工程（七层地图见 [deploy/README.md](deploy/README.md)） |
| `tools/`, `projects/` | 上游 SparseDrive 原样 |
| `docs/` | 项目文档（见下） |

## 📚 文档地图

| 文档 | 内容 |
|---|---|
| [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) | **先读**：21 轮优化日志，含主要坑与负结果归档 |
| [docs/FULL_CHAIN_REPRO.md](docs/FULL_CHAIN_REPRO.md) | 全链路复刻手册（量化+敏感层分析+阶段命令+坑索引） |
| [docs/onnx_export_surgery.md](docs/onnx_export_surgery.md) | ONNX 导出手术（静态图化与算子级优化） |
| [docs/M9A_PLUGIN_GUIDE.md](docs/M9A_PLUGIN_GUIDE.md) | trigger 规则插件开发指南（ABI v1） |
| [docs/PREPOST_OPS_DESIGN.md](docs/PREPOST_OPS_DESIGN.md) | 前后处理/节点/信箱设计 |
| [deploy/engine/ONNX_QUANT_STRATEGY.md](deploy/engine/ONNX_QUANT_STRATEGY.md) | 量化导出策略（覆盖/折叠/敏感层回退） |
| [docs/BOARD_ENV.md](docs/BOARD_ENV.md) | 板子环境速查 |
| [docs/quick_start.md](docs/quick_start.md) | 上游 SparseDrive 快速开始 |

## 🔒 板子环境与约束

- NVIDIA Orin X：JetPack 5.1.5 / TensorRT 8.6.1.2 / CUDA 11.4 / sm_87
- `/opt/m0` 挂载 **noexec**：脚本用 `bash x.sh` 调起，二进制放 `/usr/local/bin`
- **不要动板子环境**（TRT/CUDA/JetPack/系统/python 包）；允许的动作 =
  编译部署产物到 `/usr/local/{bin,lib}`
- `BOARD_HOST`/`BOARD_PASS` 只运行时注入，注册表/历史/仓库里都没有，
  也不要落任何文件

## 🙏 致谢

- [SparseDrive](https://github.com/swc-17/SparseDrive)（CVPR 2024）——
  上游模型与训练框架
- [nuScenes](https://www.nuscenes.org/)——评估数据集

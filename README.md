# SparseDrive TensorRT 车端部署（智己 Orin X）

English: see docs/ — end-to-end SparseDrive on NVIDIA Orin X, 82.6→39.1 ms closed-loop with production hardening.

基于上游 [SparseDrive](https://github.com/swc-17/SparseDrive)（CVPR 2024），
把端到端自动驾驶模型（检测 + 在线建图 + 运动预测 + 规划）完整部署到
车规域控芯片 **NVIDIA Orin X**，并完成量产化硬化。**所有精度结论都是
闭环口径**——三引擎链在板上按帧递推跑完 nuScenes mini 81 帧，任何改动
过端到端 decode 门禁才算数。

## ✨ 亮点

- **闭环三引擎链 39.11 ms/帧**（INT8 时代起点 82.6 ms，**2.11×**）：
  `e_bb2`（backbone/neck INT8）→ D2D 零拷贝边界张量 → `e_hd`（head FP16
  + v11 DFA gather 向量化 CUDA 插件）→ `e_mp`（motion/planning FP16 时序图）
- **精度与 FP32 基线打平**：det mAP **0.4157** / NDS **0.4724** /
  map mAP **0.7462**（FP32 PyTorch 0.7512）；量化数学本身只伤
  **-0.4 pt**（PTQ MTQ + 敏感层保护，量化损伤与编译损伤已分离定案）
- **量产硬化全套**：节点内 watchdog 分级弃帧、systemd 熔断拉起、
  开机自检（金标容差 + 四 md5 指纹）、1Hz 遥测、shm 权限收紧、
  8h 满负荷浸泡 4/4 PASS
- **14 规则事件触发器**：CPU 10Hz 第三类进程，**零关键链改动**，
  parity 951 tick 事件集合全等；规则以 .so 插件热接入（ABI v1）
- **21 轮优化日志**：正负结果同权重留档，每轮含归因证据链
  （FP32 基线一键复现、门禁数据、探针取证）

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
| [docs/BOARD_ENV.md](docs/BOARD_ENV.md) | 板子环境速查 |
| [docs/M9A_PLUGIN_GUIDE.md](docs/M9A_PLUGIN_GUIDE.md) | trigger 规则插件开发指南（ABI v1） |
| [docs/PREPOST_OPS_DESIGN.md](docs/PREPOST_OPS_DESIGN.md) | 前后处理/节点/信箱设计 |
| [docs/onnx_export_surgery.md](docs/onnx_export_surgery.md) | ONNX 导出手术（静态图化与算子级优化） |
| [docs/quick_start.md](docs/quick_start.md) | 上游 SparseDrive 快速开始 |

## 🔒 板子环境与约束

- 智己 Orin X：JetPack 5.1.5 / TensorRT 8.6.1.2 / CUDA 11.4 / sm_87
- `/opt/m0` 挂载 **noexec**：脚本用 `bash x.sh` 调起，二进制放 `/usr/local/bin`
- **不要动板子环境**（TRT/CUDA/JetPack/系统/python 包）；允许的动作 =
  编译部署产物到 `/usr/local/{bin,lib}`
- `BOARD_HOST`/`BOARD_PASS` 只运行时注入，注册表/历史/仓库里都没有，
  也不要落任何文件

## 🙏 致谢

- [SparseDrive](https://github.com/swc-17/SparseDrive)（CVPR 2024）——
  上游模型与训练框架
- [nuScenes](https://www.nuscenes.org/)——评估数据集

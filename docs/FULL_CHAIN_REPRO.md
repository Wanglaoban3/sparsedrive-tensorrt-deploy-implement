# SparseDrive TensorRT 板端部署 —— 全链路完整复刻指南

> 定稿 2026-10-03。本文是从训练权重到板端闭环节点的**端到端复刻手册**：
> 量化与敏感层分析的完整故事（含 head 量化为何放弃）、每阶段命令、
> 关键指标、坑索引。深度细节见各专篇：
> 量化策略 [`deploy/ONNX_QUANT_STRATEGY.md`](../deploy/ONNX_QUANT_STRATEGY.md)；
> 优化历程与坑 [`OPTIMIZATION_SUMMARY.md`](OPTIMIZATION_SUMMARY.md)；
> 前后处理节点设计 [`PREPOST_OPS_DESIGN.md`](PREPOST_OPS_DESIGN.md)；
> 板子环境 [`BOARD_ENV.md`](BOARD_ENV.md)；日常速查仓库根 `AGENTS.md`（本地文件，不入库）。

## 0. 交付物与链路总览

```
ckpt/sparsedrive_stage2.pth (官方 fp32 权重)
  │  ① 敏感层分析 + PTQ 量化 (本地 P100, ModelOpt MTQ)
  ▼
sparsedrive_int8_v3_fold5.onnx   ← trunk int8 + head float(显式fp16), QDQ 折叠
  │  ② 拆分手术 (本地, onnx-graphsurgery)                 ③ MP 头独立导出
  ├─ sp_backbone2.onnx ──► e_bb2.engine (15.0ms)          v5_mp.onnx ──► e_mp.engine (8.85ms)
  └─ sp_head.onnx ──────► e_hd.engine  (19.9ms)
  │  ④ 板端构建: onnx2engine2 (--f32-names 保层) + DFA 插件 v8 (nvcc sm_87)
  │     单引擎交付备份: e_T6.engine (38.99ms, 25.6 FPS)
  ▼
⑤ 板端闭环: sp_filesrc(图像源) → sp_modelnode(bb2∥hd+mp 三级, CUDA graph, 状态递归)
            → sp_result v2 信箱(det/map/motion/plan) → sp_resultmon(读取解码)
```

两套交付口径并存（按需选型）：
- **单引擎 e_T6 + libdfaplug_v8.so**：38.99ms / 25.6 FPS，mini mAP 0.4203 / NDS 0.4738；
- **拆分链 e_bb2 + e_hd（+ e_mp）**：与单引擎同精度（det 0.4177 / map 0.7478），
  是闭环节点的实际运行形态，也是 M7 重叠优化与 M8 零拷贝采集的载体。
- 引擎按插件 名字/版本/命名空间 运行时绑定 → **换 .so 不用重编引擎**。
  v8 插件指纹 md5 `55df31bcb958c57db05e5ec024d002af`。

## 1. 关键指标（速查）

**延迟（Orin X, e_T6 口径=5 次均值；拆分=单流串行）**

| 链路 | 延迟 | 说明 |
|---|---|---|
| e_T6 + v8 插件 | 38.99ms / 25.6 FPS | 单引擎交付 |
| e_bb2 | 15.0ms | graph×2 parity；无状态 |
| e_hd | 19.9ms | graph×2；8 个递归状态 |
| e_mp | 8.85ms | 单时序引擎 fp16；9 状态 |
| 节点三级端到端 | ~48.1ms | 含前处理/队列开销 |
| M7 双流（已定案） | 23.16 vs 22.48 fps | iGPU SM 争用，无收益，交付维持单流 |

**精度（mini 81 帧 = boston 40 + queenstown 41，场景边界 k=40）**

| 口径 | det mAP | NDS | map | 说明 |
|---|---|---|---|---|
| PyTorch fp32（task_sentinel 配对基准） | 0.4255 | 0.4805 | — | 本地 P100 |
| 节点闭环 legacy（m3fix） | 0.4210 | 0.4740 | — | mat4_inv 修复后 |
| 节点闭环 split（m5fix） | 0.4177 | 0.4735 | 0.7478 | 交付拆分链 |
| 节点闭环 +MP（m6fix） | 同上 | — | 同上 | EPA 0.6072/0.5112, L2 0.7456, col 0.161% |
| PyTorch MP 基准（mini_sp32_mp） | — | — | — | EPA 0.5939/0.5094, L2 0.7434 |

**全量 val（PyTorch，6019 帧，官方 devkit，见 ONNX_QUANT_STRATEGY §6.1）**：
fp32+fp16 注意力 mAP 0.4139 / NDS 0.5248（官方 README 0.5257，差 0.0009）；
INT8 QAT skip28 0.4081 / 0.5200（量化代价 −0.53 / −0.26 点）。

## 2. 阶段 0 —— 环境

- **本地（Windows）**：`H:\miniconda3\envs\sparsedrive_deploy\python.exe`
  （3.10, torch 2.0.1+cu118, mmcv 1.x, nuscenes devkit, ModelOpt 0.11；
  GPU Tesla P100-16GB，只做量化/导出/对拍，不编引擎）。系统默认 `python`
  (3.11) 跑无 mmcv 依赖的 eval/map 脚本。
- **板子（智己 Orin X）**：SSH `root@192.168.2.104` 密码 `nvidia`，paramiko 驱动
  （`deploy/board_m1.py` 的 `launch()`）。TRT 8.6.1.2 / JetPack / sm_87。
  **不动板子环境**；产物只放 `/usr/local/{bin,lib}`；`/opt/m0` noexec（脚本
  `bash x.sh` 调起）；板载只有 python2.7/3.8，对拍解析全在本地做。
- **数据**：mini val 81 帧来自 `data/infos/mini/nuscenes_infos_val.pkl`；
  引擎格式输入向量在 `work_dirs/board_vectors`（推板后= `/opt/m0/trt-dev/vec/mini`，
  每帧 `in_XX/` 目录 + `manifest.tsv`）；量化校准 16 样本在 `work_dirs/board_vectors_mtq`；
  节点模拟图像源 `work_dirs/nv12_r0`（`tools/make_nv12_manifest.py` 生成 manifest）。
- 板端引擎构建器 `/usr/local/bin/onnx2engine2`（源码 `deploy/onnx2engine.cpp`），
  关键开关：`--f32-names list.txt`（按 ONNX 节点名精确保 FP32 层）、
  `--f32-substr a,b,c`（按名字子串）、`--f32-notq`（Q/DQ 邻接区外全 FP32）、
  `--plugins libdfaplug_v8.so`。

## 3. 阶段 1 —— PyTorch 基线复现（一切精度对比的地板）

```bash
# FP32 mini 基线一键复现 (P100 ~3min): dump 引擎同格式张量到 evaldata\mini_sp32
H:\miniconda3\envs\sparsedrive_deploy\python.exe deploy\infer_sp32_mini.py
# map 口径评估 + det 官方评估
python deploy\eval_t6_mini_map.py --eng ...evaldata\mini_sp32 --art ...
H:\miniconda3\envs\sparsedrive_deploy\python.exe deploy\eval_sp32_mini_det.py
# MP 头 PyTorch 基准 (EPA/L2/col 口径): infer_sp32_mp_mini.py + eval_mp_mini.py
```

注意：本 fork 曾把 eval 注意力分支改成纯 FP32，而 ckpt 是官方 flash_attn
**FP16 kernel** 训练的——FP32 注意力反而分布外，切回 fp16 找回 +0.22 NDS
（复现论文数字用 `eval_nuscenes.py --attn-fp16`）。全量 val 数字见 §1。

## 4. 阶段 2 —— 量化与敏感层分析（完整故事，含 head 量化为何放弃）

> 本段是"整条链路包括量化和敏感层分析"的正史。工具全部在 `deploy/`，
> 指标 JSON 在 `deploy/artifacts/`（组级敏感度 eval_grp_* 等）。

**① QKV 模块化手术**（`flashmha_qkv.py`）：FlashMHA 的 QKV 是打包裸参数
`in_proj_weight` + `F.linear`，MTQ 包不到 → 拆成 q/k/v 三个 nn.Linear
（逐位一致，`test_flashmha_qkv.py` 自测），量化模块 112→133、量化器 335→398。

**② 逐模块 leave-one-out 敏感度**（`ptq_sensitivity.py --calib-samples 16
--eval-samples 16`）：全 INT8 漂移、每模块留一敏感度排名、skip-topK 曲线。
**测量陷阱（v1/v2 报告作废原因）**：`mtq.quantize()` 会把模型翻成 train() 模式，
校准循环用 16 个批的统计覆盖 BN running_mean/var，留下 ~0.2 rel_l2 的
非量化漂移——修复：校准循环内强制 `model.eval()` + 校准后恢复 BN 缓冲区
（修复后全 INT8 rel_l2 0.4283→0.2563）。教训：**逐模块 rel_l2 排名是长尾、
头部全在噪声带内，skip-topK 无任务收益**。

**③ 组级任务口径敏感度**（`group_sensitivity.py`，最终选层依据）：功能组
留组法——每组保 FP 其余全量化，81 帧配对测 mAP/NDS（ModelOpt 校准模式前向
恒等只记 amax，组间可比）。结论（噪声底 ±0.005）：
- 全 INT8 任务损伤本身小（−0.009 mAP / −0.009 NDS）；
- **det_head_output（cls/quality 输出 12 模块）保 FP 几乎完全恢复**（−0.0006），
  与 rel_l2 排名（全是 backbone）完全不同；
- det_head_attn 保 FP 反而更差（误差补偿效应，勿保护）；FPN 量化零代价；
- 推荐保护组 = det_head_output（+backbone_deep 次优），其余保持 INT8。

**④ 交付量化口径 = trunk int8 + head float**：int8 覆盖 backbone/FPN/encoder/
FFN/QKV/out_proj/weights_fc；显式 fp16 只有两处（注意力内部 QK^T/PV MatMul、
kps GEMM）；LayerNorm/softmax/DFA 边界(loc/feat)保持 FP。

**⑤ head 量化为什么放弃（编译问题，非精度问题）**：曾计划 head 全量化/
fp16 链（最大化覆盖），板端实测两条编译侧拦路：
- **kps GEMM 走 fp16 链时 TRT 8.6 NVRTC JIT 损坏**（板端 TRT 8.6.1.2 的
  NVRTC 编译管线对这条链直接失败）；
- **Myelin unrolled MatMul 在无 QDQ 锚点时 signal 7 crash**（int8 图必须
  QDQ 结构标准化，`qdq_onnx_rewrite.py` 就是为此存在）。
  两者都属"引擎编译器行为"而非数值问题 → head 保持 float + 显式 fp16，
  int8 只注入 trunk。`--f32-names` 机制（第九轮补）就是给"保层"开的口子。

**⑥ 归因终审（第十一轮，防再翻案）**：单引擎整图 map 一度 −10.4pt——
实验证明是**全图编译 tactic 损伤而非量化**：int8 trunk 注入 PyTorch 干净
（0.7420），拆分链 e_bb2+e_hd 复活到 0.7477（基线 0.7481）。拆分链由此
定为感知模块最终交付形态。

**⑦ QDQ 结构标准化**（`qdq_onnx_rewrite.py`）：权重 Q 折叠 int8 initializer、
Constant scale/zp 移共享 initializer、拓扑排序；135/134 条 Q→DQ 折叠链
ORT 重放逐位相等；DFA 12 个节点输入边全 FP 收尾（报告
`*_rewrite_report.json`）。产物：`sparsedrive_int8_v3_fold5.onnx`（交付图源）。

```bash
# 复刻命令 (sparsedrive_deploy 环境)
python deploy/ptq_sensitivity.py --calib-samples 16 --eval-samples 16
python deploy/render_ptq_report.py
python deploy/group_sensitivity.py            # 组级 mAP/NDS 留组法
python deploy/task_sentinel.py                # 81 帧配对任务哨兵
python deploy/export_quant_onnx.py --checkpoint ckpt/sparsedrive_stage2.pth \
    --protect-groups det_head_output --attn-fp16 \
    --out work_dirs/sparsedrive_small_stage2/sparsedrive_int8_dho.onnx
python deploy/qdq_onnx_rewrite.py work_dirs/.../sparsedrive_int8.onnx
```

## 5. 阶段 3 —— 拆分图手术与 MP 导出（本地）

| 图 | 产生方式 | 引擎 |
|---|---|---|
| `sparsedrive_fp32.onnx` (283MB) | 时序图导出（instance_t_matrix/time_interval 进图，静态 shape, opset13） | — |
| `sparsedrive_int8_v3_fold5.onnx` (173MB) | PTQ+QDQ 折叠（§4） | e_T6 |
| `sp_backbone2.onnx` (26MB) | `deploy/split_engine.py`（拆分）+ `requant_fpn.py`（FPN 再量化口径） | e_bb2 |
| `sp_head.onnx` (93MB) | `deploy/split_engine.py`：特征图+prev/metas → 19 输出 | e_hd |
| `sp_head_det/rest.onnx` | `deploy/split_head.py`（二段拆分，P4 实验留档，无收益） | （留档） |
| `v5_mp.onnx` (47MB) | `deploy/export_mp_onnx.py`：MotionPlanningHead `forward_onnx` 纯函数化（Roll→cat-shift 见 diff 注释） | e_mp |

保层清单：`map_f32_names.txt` / `seg_f32_names.txt`（`_gen_f32_names.py` 生成），
构建 e_T6/e_hd 时 `--f32-names` 传入——时序 bank/TopK 等对半精度敏感层强制 FP32。

## 6. 阶段 4 —— DFA 插件（板端 nvcc）

演进：v3 基线 → v8 交付（half2+FP16 IO → softmax 折入 → plan+gather 双内核，
累计 −19.6−7.5−14.7−6.3ms）+ 内置 v3 回退。**f16 物化调优在 sm_87 上
收益有限（P5 实测 1.6 vs 5 TFLOPS），不要再走手写 wmma 路线。**

```bash
# 板端编译 (缺 -lnvinfer 时 trtexec 加载报 undefined symbol: getPluginRegistry)
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC \
  -o /usr/local/lib/libdfaplug_v8.so deploy/dfaplug_v8.cu -lnvinfer
```

同名/同命名空间多版本 .so 并存安全（按路径显式 dlopen；TRT registry
先注册者生效）。交付 v8 md5 `55df31bc...`；本地备份在 `models/`。

## 7. 阶段 5 —— 引擎构建（板端）

```bash
# 单引擎交付 (整图 + f32 保层)
/usr/local/bin/onnx2engine2 models/sparsedrive_int8_v3_fold5.onnx \
  models/e_T6.engine --f32-names map_f32_names.txt --plugins /usr/local/lib/libdfaplug_v8.so
# 拆分链
/usr/local/bin/onnx2engine2 models/sp_backbone2.onnx models/e_bb2.engine \
  --plugins /usr/local/lib/libdfaplug_v8.so
/usr/local/bin/onnx2engine2 models/sp_head.onnx models/e_hd.engine \
  --f32-names seg_f32_names.txt --plugins /usr/local/lib/libdfaplug_v8.so
# MP 时序引擎 (fp16, 无 DFA 边 → 不需要插件, 但 run_engine 传 v8 兼容)
/usr/local/bin/onnx2engine2 models/v5_mp.onnx models/e_mp.engine --fp16
```

（实际参数以 `deploy/board_bb2.py` / `board_split_build.py` / `board_mp.py`
内命令为准，本节是形状。）引擎与插件绑定：e_T6/e_hd 声明 DFA 插件
名字/版本/命名空间，运行时从 registry 取。**引擎训练/构建很慢，改插件不重编
引擎；反复调试用模块级编译。**

## 8. 阶段 6 —— 板端离线复刻（run_engine 链，对拍权威）

`work_dirs/board_vectors` 推板为 `vec/mini`（81 帧 `in_XX`，每目录
manifest.tsv 列 name/dtype/dims/file）。离线链就是文件版递归推理：

```bash
# 感知链 (e_T6 或 bb2+hd): 前帧 next_* 拷成后帧 prev_* (repro/mini_pipeline_v8.sh)
bash repro/mini_pipeline_v8.sh 0 80        # → vec/mini/outv8_XX (19 张量/帧)
# MP 链: 吃感知 dump + 外部维护 9 状态, 场景首帧(0,40)零复位 (mini_pipeline_mp.sh)
bash repro/mini_pipeline_mp.sh e_mp.engine libdfaplug_v8.so <感知SRC> <OUTM> 0 80 "0 40"
```

评估（本地）：

```bash
python deploy\eval_t6_mini_v8.py                 # det/map 指标 (EVAL_ENG_DIR 可换目录)
python deploy\eval_t6_mini_map.py --eng <dir> --art <outdir>   # map 口径
python deploy\eval_mp_mini.py --eng <dir>        # EPA/L2/col 口径 (MP)
```

铁律：**递归状态链 f1+ 必然发散**（byid rel 0.57、top-300 重叠 30-40% 是正常
混沌底），逐张量 A/B 只用于 f0 与结构性错误；一切跨实现对比只看闭环
mAP/EPA。离线链（mpref）与节点闭环（m6fix）同源对拍 f0 plan 差 ≤0.5%。

## 9. 阶段 7 —— 闭环节点（交付运行形态）

源码 `deploy/prepost/`：`sp_filesrc`（NV12 图像源/模拟器）→ 共享内存总线
（seqlock + 槽位回收锁+fence，M1.5 纪律）→ `sp_modelnode`（前处理 preproc.cu
+ 三级引擎 + 状态递归 + 结果发布）→ `sp_result` v2 信箱（det+map CRC、
motion/plan/最终轨迹 append）→ `sp_resultmon`（读取解码打印）。

```bash
# 板端典型运行 (deploy/_m5_rerun.py legacy|split|mp 是全流程驱动)
/usr/local/bin/sp_filesrc m3 1600 900 5 /opt/m0/trt-dev/nv12_r0/manifest.jsonl \
  /opt/m0/trt-dev/nv12_r0 --fresh --wait-cons 1
/usr/local/bin/sp_modelnode m3 /opt/m0/trt-dev/models/e_bb2.engine \
  /usr/local/lib/libdfaplug_v8.so /opt/m0/trt-dev/nv12_r0/manifest.jsonl <OUT> \
  --hd /opt/m0/trt-dev/models/e_hd.engine --mp /opt/m0/trt-dev/models/e_mp.engine \
  --warmup 2 --frames 81 --img-from /opt/m0/trt-dev/vec/mini \
  --serial --graph --mailbox sp_result_m3 --cmd 2
/usr/local/bin/sp_resultmon sp_result_m3 --interval-ms 10 --wait-ms 60000
```

要点：bb2(k) 无状态产 col_feats [1,89760,256] f16 46MB D2D → e_hd 吃 8 递归
状态（dev[0]=dev[1] 单缓冲，场景首帧 k=0/40 用模板 D2D 复位）→ 9 感知张量
别名喂 e_mp + 9 MP 状态同约定。节点无运行时 cmd 源 → `--cmd` 默认 2 直行，
信箱携带完整 plan_cls/reg 供下游重选。门禁三件套（`_m5_rerun.py` /
`_m6_ref.py` + `_m6_cmp.py` / `_m6b_gate.py`）：闭环 mAP/EPA 不劣化 +
离线参考同源 A/B + 信箱解码 12/12 逐位一致。

M7 双流实验（`--dual`，2026-10-03 定案）：bb2 独立流 ∥ hd+mp，精度无损
（det 0.4177 / map 0.7485 / EPA 0.6048/0.5023）但 iGPU SM 争用吞掉重叠
收益（23.16 vs 22.48fps），**交付维持单流**；开关保留，详见
OPTIMIZATION_SUMMARY 第十三轮。方法学：跨 run 的 bit 比对必然失败
（TRT tactic 非确定），判别新旧路径差异要用"当前二进制原模式重跑"做对照
（`_m7a_gate.py` / `_m7a_bit2.py` / `_m7a_cmp.py`）。

## 10. 坑索引（每条详情见 AGENTS.md 对应小节 / OPTIMIZATION_SUMMARY"遇到的主要坑"）

| # | 坑 | 一句话修法 |
|---|---|---|
| 1 | mat4_inv 双坑（MESA 列主序+抄错项；解析式平移写进行 3） | 刚体解析式按扁平下标 3/7/11 写；numpy 二维验证过 ≠ C 一维写对，必须 1D 镜像复算 |
| 2 | 闭环 −10pt 这类"精度事故" | 元素/张量 A/B 对递归链无效（混沌底），只有闭环 mAP/EPA 是门禁 |
| 3 | 板端 TRT 8.6 NVRTC JIT 损坏（kps fp16 链） | 该层 --f32-names 保回 FP32；head 量化放弃的直接推手 |
| 4 | Myelin unrolled MM signal 7（无 QDQ 锚） | int8 图必须 qdq_onnx_rewrite 结构标准化 |
| 5 | MTQ quantize 翻 train() 污染 BN（0.2 rel_l2 假漂移） | 校准循环强制 eval + 恢复 BN 缓冲区 |
| 6 | nvcc 插件 undefined symbol: getPluginRegistry | -shared 必须配 -lnvinfer |
| 7 | 插件 getWorkspaceSize 烤进 plan（P3 最大坑） | workspace 大小运行时报告，勿在构造期定死 |
| 8 | 逐张量差 0.45 但 mAP 只差 0.0014 | 跨代对比只看 mAP；逐张量只定位结构性错误 |
| 9 | 板上脚本 CRLF 静默死 / marker touch 错目录 / paramiko PipeTimeout | heredoc 生成脚本；绝对路径 touch；括号孤儿 launch() |
| 10 | Windows cmd 多行 python -c 静默失败 | 一律 Write 临时 .py 再跑（本项目最高频坑） |
| 11 | dump 参考数据 provenance 不明被当真值 | 任何历史 dump 先用常量链哨兵验明正身 |
| 12 | 单引擎图 map −10.4pt | 全图编译 tactic 损伤，非量化；拆分链复活——不要再试整图 |

## 11. 工件与数据清单（复刻时去哪找）

| 位置 | 内容 |
|---|---|
| `models/`（本地备份，不入库） | e_T6/e_bb2/e_hd/e_mp.engine + libdfaplug_v8.so（md5 见 §0） |
| `work_dirs/sparsedrive_small_stage2/` | 全部 ONNX 图源、`*_rewrite_report.json`、build 日志 |
| `work_dirs/.../evaldata/` | mini 各口径引擎 dump：mini_sp2_81（PyTorch 81f）、mini_eng_v8（v8 权威）、mini_eng_v8r、mini_eng_sp、mini_mp_eng（离线 MP 权威）、mini_m3fix_81/m5fix_81/m6fix_mp（节点闭环门禁）、mini_sp32_mp（PyTorch MP 基准） |
| `work_dirs/preproc_ref/` | m3fix/m5fix（板端原始 fetch）、m6fix（节点 MP dump）、mpref（离线 MP 参考）、mpzero（状态零模板）、vecmeta（probe 输入对拍） |
| `deploy/artifacts/` | eval_*/summary+metrics（组级敏感度、skip 曲线、各代 mAP JSON） |
| 板 `/opt/m0/trt-dev/` | models/（引擎）、vec/（输入向量+dump）、repro/（链脚本）、m5fix_out（MP 源） |
| 板 `/usr/local/{bin,lib}` | onnx2engine2、run_engine(s)、sp_* 工具、libdfaplug_v8.so |

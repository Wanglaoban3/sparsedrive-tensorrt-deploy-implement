# SparseDrive TensorRT 板端部署优化总结

## 交付版本（2026-10-04 定稿：拆分链 bb2+hd）

**感知模块最终交付 = 两引擎链**（第十一轮终审后用户拍板；原单引擎组合
完整保留作回退）：

| 项目 | 值 |
|---|---|
| 引擎 | `models/e_bb2.engine` (28 MB, md5 `f375467e...`) + `models/e_hd.engine` (104 MB, md5 `8c3a6257...`) |
| 链接 | `run_engine2 e_bb2 e_hd <plugin> in_dir in_dir --dump out`（D2D 零拷贝，边界张量 `/Reshape_9_output_0`；N 引擎泛化版 `run_engines`） |
| 插件 | **`/usr/local/lib/libdfaplug_v11.so`（2026-10-06 起，向量化 gather，见第十八轮）**；回退 `libdfaplug_v8.so`（md5 `55df31bc...`） |
| 图源 | `sp_backbone2.onnx`（bb2 = backbone+FPN+format）+ `sp_head.onnx`（det+map heads） |
| runner md5 | run_engine2 `60dff83b...`；run_engines `0b4da654...` |
| 延迟 | **39.11 ms**（v11；v8 口径 39.86 ms；单引擎 38.99 ms） |
| mini det mAP | **0.4157** / NDS **0.4724**（v8 0.4160/0.4729，单引擎 0.4203/0.4738，det -0.4pt） |
| mini map mAP | **0.7462**（v8 0.7481；单引擎 0.6471 → **+9.9pt**；FP32 PyTorch 0.7512） |
| 量化 | 与单引擎同配置同 scale（backbone/neck INT8 QDQ，head float 权重，DFA 权重 FP16） |
| FP32 参照 (P100) | mini det mAP 0.4256 / NDS 0.4798；mini map mAP **0.7512** |
| 原始基线 | 82.6 ms / 12.1 FPS / mAP 0.4178 → **2.08× 加速**（vs 39.86ms） |

**为什么拆分**（第十一轮终审）：单引擎 map -10.4pt 不是量化代价——引擎
真实 int8 FPN（偏差 13%）注入 MTQ 模型 81 帧链 mAP 0.7420（干净）；
是**全图编译 tactic 损伤**（head 区 fp16 融合内核，图侧/编译侧均不可控，
TRT 8.6.1.2 限制）。同一套图与 scale：全图编 0.646~0.647，拆分编
0.7477~0.7481，PyTorch 0.7494~0.7512。

**回退/留档**：单引擎 e_T6+v8（38.99ms / det 0.4203 / NDS 0.4738 /
map 0.6471，e_T6 md5 `7fbcdac1...`）未触碰；换 TRT 版本后可评估回归
单引擎。交付链 81 帧 dump 留档 `evaldata/mini_sp2_81`，链脚本板上
`mods/mini_pipeline_sp2.sh`。

**注意**: 两引擎均按插件名/版本/命名空间 ("DeformableAggregation" v1)
动态绑定插件，**换插件不用重编引擎**。v11 为当前默认（v11→v8→v3 回落
链；`DFA_V11_DEBUG=1` 可查实际路径与 workspace）；v8 在条件不满足时
自动回落 v3 内核（`DFA_V8_DEBUG=1`）。多个插件 .so 在 /usr/local/lib
并存安全（按路径显式 dlopen）。

**P5 FlashAttention 实验（已完成，负结果不采用）**: det 侧 10 个大注意力
站点换手写 FlashSDPA 插件内核，模块级 +6.1ms（可收割上限仅 ~2.75ms），
维持关闭（见第八轮）。

---

**注意**: e_T6 引擎按插件名/版本/命名空间 ("DeformableAggregation" v1) 动态
绑定插件，**换插件不用重编引擎**。e_T6 ↔ libdfaplug_v8.so（推荐）或
libdfaplug_v3.so（logits 语义相同，v8 语义与 v3 逐条一致且数值 ulp 级等价）。
v8 在条件不满足时自动回落 v3 内核（`DFA_V8_DEBUG=1` 可查实际路径）。

**P4 引擎拆分实验（已完成，不改变交付）**: 拆分所有配置均慢于 e_T6 单引擎
（bb2+hd 两链 39.86ms、bb+det+rest 三链 40.12ms vs 38.99ms），交付维持
e_T6 + v8 不变。拆分产物与工具留档备用（见第七轮）。

**P5 FlashAttention 实验（已完成，负结果不改变交付）**: det 侧 10 个大注意力
站点换手写 FlashSDPA 插件内核，模块级 +6.1ms（可收割上限仅 ~2.75ms，手写
wmma ~1.6 TFLOPS vs TRT 基线链 ~5 TFLOPS），交付维持 e_T6 + v8 不变
（见第八轮）。

---

## 优化全过程

### 前置：INT8 量化图可编译性

MTQ 全 int8 量化图在该板 TRT 8.6.1.2 上不可编译（chooseHigherPrecision 断言）。
经大量实验确认：head 区域的量化是毒药。解决方案 T3：head 免量化（QDQ 剥离 +
权重反量化），backbone/neck 保持 INT8。T3 图可编译，精度与 MTQ 一致。

### 第一轮：kps 6维广播 MatMul 重写（-19.6ms）

**问题**：kps_generator 的投影 MatMul 输出为 6 维广播 [1,6,1,1,4,4]×[1,1,100,300,4,1]，
总共只有 5.8 MFLOP，但 TRT 对高维广播 GEMM 走最差通用内核，单次 2.8ms × 5 处 = 14ms。

**解法**：等价重写为 3D GEMM `x[1,30000,4] @ W^T[6,4,4]` → [6,30000,4] → reshape 回。
float 版逐位验证一致但触发板端 NVRTC JIT 损坏（fp32 kernel 编不了）；
加 fp16 Cast 后正常编译。mini mAP 无回退。

**结果**：14.07ms → 0.57ms。

**坑**：
- `numpy.matmul` 广播规则：in1 必须是 3-D 才能和 [6,4,4] 批广播
- 乘法顺序：原始语义是 `A @ x`（A 在左），写反了输出不报错但数值全错
- 板端 NVRTC JIT 对 fp32 融合 kernel 编译损坏 → 必须加 fp16 Cast 绕开
- 这个 fp16 Cast 导致 kps 坐标引入 fp16 舍入（~1e-3），对 mini mAP 无影响

### 第二轮：DFA 插件 half2 + FP16 IO（-7.5ms）

**问题**：DFA 插件 v1 用 fp32 IO，访存量 92MB × 2（读+写）。

**解法**：
- fp16 路径一线程处理相邻 2 通道（__half2 载入/写出）
- `__ldg` 只读缓存
- shape/ssi 表进共享内存
- 插件 IO 强制 FP16（supportsFormatCombination 限制）
- 内部双线性插值仍 FP32 累加

**结果**：16.1ms → 9.07ms。

**坑**：双线性插值行距漏乘宽度因子（`r0 = h_low * C` 应为 `r0 = h_low * w * C`），
导致全量输出错误。被 mini mAP 回归当场拦下。

### 第三轮：softmax 折入 DFA（-4.7ms）

**问题**：kps_generator 的 softmax + 转置链独立于 DFA 运行，
15 个 softmax 内核共 1.1ms + 权重张量 4.5MB 物化往返。

**解法（用户方案）**：在导出端（export_v5.py）让 weights_fc 的输出以
logits 形式直通到 DFA 节点——不做 softmax、不做转置。DFA 插件 v3 内部：
- 对 logits 在 (cams×S×P) 联合轴做 inline softmax（max-subtracted）
- 组内 16 线程分段扫描 + shuffle 归约
- ε=1e-4 跳过死权重

校准阶段保持原路径（不激活 logits patch），确保量化状态与 v3 逐位一致。

**结果**：GPU 总耗时 50.00 → 45.30ms。

**关键设计**：`DFA_STATE` 开关只在 `torch.onnx.export` trace 时生效，
校准和参考运行走原始路径，量化 amax 完全不受影响。

### 第四轮：head 全 fp16 化（P1, e_T12）—— 图侧已穷尽，退化区确认是板端 TRT 全图编译行为

**背景**：T3 退化区的根因叙述是"NVRTC fp32 JIT 损坏导致 fp32 Myelin 区退化"。
P1 的假设：把 head 区域全部改成 f16（v5_P1h → v5_P2h.onnx），让 Myelin 在
这些区域直接选 fp16 tactic，绕开坏掉的 fp32 JIT 路径。

**图侧工作（全部完成且验证通过）**：
- `deploy/make_v5_p2h.py`：v5_P1h → v5_P2h.onnx（222MB → 125MB）
  - 剥离 3 对残留的 head 权重量化器（layers.0 q/v/kps learnable_fc 的
    WEIGHT QDQ，其 DQ 只喂 head 浮点计算）
  - INT8 区用"Q 浮点输入的全后向闭包"重新围栏（349 节点，
    单层 fence 会漏 BN→Relu→Q 链）
  - 区内全部 f16 化 + 权重 f16 拷贝 + 边界 cast 修复，混合 dtype 节点 0
  - 图 IO 契约不变（19 输出 f32/i32），stale value_info 清除
- 本地 stub-ORT 对拍 `P2H_STUB_PASS`（门控最差 2.7e-4；检测输出 ≤1e-5；
  instance-id 重编号为平局翻转，不入 gate）
- 板端模块 A/B（trtexec --fp16，48 节点 span）：0.232 → **0.123ms (1.9×)**，
  reformat 全消；381 节点 det-decoder 栈：2.93 → **1.83ms (1.6×)**

**整图 e_T12（onnx2engine2 同参重编 v5_P2h）**：构建成功，mini mAP
**0.4182**（gate ≥0.418，e_T6=0.4184），NDS 0.4734（+0.001）。
**但延迟 45.60ms ≈ e_T6 45.30ms——6 个退化 ForeignNode 逐项持平
（2.15/1.78/1.78/1.77/2.78/0.91），reformat 仅 -0.06ms。f16 没有传进 tactic 选择。**

**模块级对照实验（7 项，全部无法复现整图退化）**：

| 变量 | 模块结果 | 结论 |
|---|---|---|
| dtype（P1h fp32 vs P2h f16） | FN 1.19 vs 0.99ms | f16 有效但小 |
| onnx2engine2 工具 + fp16（无 int8） | 1.11ms | 排除工具配置 |
| kINT8 flag（合成 QDQ 模块） | 0.99ms (median) | 排除 kINT8 |
| 跨度（含 int 密集前缀的 513 节点宽 span） | 1.11ms | 排除 span 组成 |
| --f16-notq 逐层精度约束（自编 onnx2engine3） | 1.12ms | 约束不改变 Myelin |
| workspace 64MB / 256MB | 1.12ms | 排除 workspace |
| trtexec vs 工具构建 | 一致 | 排除构建器差异 |

**结论**：退化区只存在于"全图编译"语境。同一段计算在模块里恒定 ~1.1ms、
在整图里恒定 ~2.2ms，与图 dtype/标志/约束均无关。e_T6 文档里
"板端 TRT 8.6.1.2 限制"的说法成立且更进一步：**从 ONNX 侧无法触达**。

**遗留价值**：e_T12 精度中性（mAP +0.0002、NDS +0.001）、速度中性、
head 图已 f16 化 —— 一旦板端 TRT 升级，退化区应直接吃 fp16 融合；
以及 3 对残留权重 QDQ 清理。e_T6 仍是交付组合。

### 第五轮实验（无收益/未采用，已回退）

| 方案 | 想法 | 结果 | 原因 |
|---|---|---|---|
| QKV 合并 | 32 个 MatMul → 13 个宽 GEMM | +0.6ms | Myelin 本来就会重排，Slice 增加额外 reformat |
| f32-substr cls/quality | 130 层强制 FP32 | 无效果 | 退化区根因不是精度而是融合模式 |
| v5 4通道内核 (LDG.64) | 减半 LSU 指令 | 引擎内回退 | 真实数据 ε-skip 占主导，线程减半伤延迟隐藏 |
| v6 map 三内核 | P-split 修复线程饥饿 | 引擎内回退 | 同上，真实数据 ε-skip 已使 map DFA 便宜 |
| v7 live-list | plan+gather 紧凑列表 | misaligned address | 多轮修复未收敛 |
| DFA int8 特征输入 | 减半访存 | +0.7ms | 瓶颈在 L2 反复读而非 DRAM 唯一流量 |
| float 化 (bank int 隔离) | 消除 bool/int32 毒化 | 更慢 (65ms) | 退化区根因不是 bool/int32 |

### 第六轮：DFA plan+gather 双内核（P3，v8 插件，-6.3ms）

**问题**：v3 单内核每线程 (anchor,通道) 全量扫 NCSP=cams×S×P 个采样点。
真实数据上 loc 有效权重只占总质量 18.3%、ε-skip 后活跃条目只占 12.85% ——
967ms 内核时间里 87% 的访存/计算在处理无效点。map 头 (A=100,P=300,NCSP=7200)
尤其吃亏（引擎内单次 1.53ms × 6 次调用）。

**解法（v8 = `deploy/dfaplug_v8.cu`，两内核 + 紧凑列表）**：
- **K1 plan**（每 anchor 一个 block）：pass1 寄存器 online-softmax
  （单遍同时得 max/sum，运行修正）+ 确定性蝶形归约；pass2 只对 loc 合法且
  wt≥1e-4 的 (c,s,p,g) 生成 32B 条目 `{int off[4] 四角绝对偏移, float w[4]
  =wk*wt 折叠权重}`，atomicAdd 挂进 (anchor,group) 专属紧凑列表
- **K2 gather**（每 (anchor,group) 一个 warp，lane=组内通道）：逐条目
  2×int4 广播 + 4×连续 half 合并读 + FMA，无分支（死角点权重 0 + 偏移钳 0）
- workspace = nA·G·(4B 计数 + NCSP·32B 条目)。**引擎由 v3（ws=0）构建，
  TRT 运行时传 nullptr —— 插件自分配**（每实例懒分配只增不减，
  det 72MB / map 184MB，12 实例共 ~1.5GB）
- 语义与 v3 逐条一致：logits 输入（内联 max-subtracted softmax）、loc 有效性
  0<w<1、ε-skip、双线性角点边界条件、索引 i=(c·S+s)·P+p；条件不满足自动
  回落同 .so 内的 v3 内核

**数值**：softmax 归约树与 v3 不同 → 真实 IO 上 v8-vs-v3 l2≈1e-5、
maxabs≤1.95e-3（ulp 级，权重边界条目翻转 + 累加顺序）；端到端一帧
det_cls 差 maxabs 0.16（6 层 decoder 逐级放大），mAP 实测持平略升。

**验证链（三级）**：
1. `dfaplug_dump.so` 运行时 dump 插件抓 e_T6 引擎真实 12 组 DFA IO →
   v3 内核对拍 **bitwise 全等**（l2=0.00，12/12 调用）—— 整条数据路径、
   logits 语义、布局、常量全部实证
2. v8 同 harness 对拍 ulp 级等价 + 分相计时
3. 引擎内 trtexec 100-iter profile + mini 81 帧 mAP 门禁

**结果**：

| 层面 | v3 | v8 | 提升 |
|---|---|---|---|
| harness 单调用 map (A=100,P=300) | 1.84-2.08ms | 0.51-0.61ms | 3.3-3.7× |
| harness 单调用 det (A=900,P=13) | 0.56-0.61ms | 0.48-0.55ms | 1.1-1.2× |
| harness 12 调用序列 | 15.4ms | 6.6ms | 2.35× |
| 引擎内 map 6 调用 | 9.15ms | 2.72ms | **3.4×** |
| 引擎内 det 6 调用 | 2.17ms | 2.31ms | 持平* |
| 引擎内 DFA 桶 | 11.32ms | **5.03ms** | **2.25×** |
| 整帧 Total（100 iters mean） | 45.27ms | **38.99ms** | **-6.3ms** |

\* det 在引擎内 v3 本就 0.36ms（特征 L2 驻留），v8 无优势；收益全部来自 map。
mini mAP 0.4203 / NDS 0.4738（v3: 0.4184/0.4723）——门禁通过。

### 第七轮：引擎拆分（P4，实测无端到端收益，已定论）

**动机**：P1 曾预测把 backbone INT8 与 head f16 分开建引擎（head 侧无
QDQ、kFP16-only 构建）可让退化融合区 ~11.2ms → ~5-6ms，净赚 ~5ms。
P4 实测该预测不成立——历史 profile 对 backbone 的记账本来就是错的
（见下方勘误），拆分从一开始就没有可收割的空间。

**拆分方式**（`deploy/split_engine.py` + `deploy/split_head.py`）：
- `v5_P2h.onnx` → `sp_backbone.onnx` + `sp_head.onnx`：以
  `/Reshape_9_output_0` f16[1,89760,256]（bev 特征展平）为锚，取其
  **完整祖先闭包**（377 节点，含 img 侧计算与权重 Q/DQ 准备）为
  backbone，边界恰好 1 张张量、head 侧 0 个 Q/DQ、无二次跨图。
  按 img 可达性过滤会把权重侧 Q/DQ 留在 head（它们不从 img 出发），
  必须用纯祖先闭包。
- `sp_head.onnx`（4333 节点）二次拆 → `sp_head_det.onnx`（2781 节点，
  6 个 det DFA）+ `sp_head_rest.onnx`（1552 节点，6 个 map DFA），
  边界 7 张量（det_instance_feature/anchor_embed/bbox 的 .pre_o、
  `/layers.38/Add_output_0`、`/LessOrEqual_output_0`、w2th、
  instance_t_matrix）。.pre_o 张量形状推断返回 scalar，须按消费端
  权重形状 OVERRIDE（且必须在 shape-inference 回填之后、graph io
  生成之前写入 vi 表）。
- fpn INT8 再量化手术（`deploy/requant_fpn.py`）：P2h 把 fpn conv
  改成了 f16 权重路径，手术把它接回 DQ 输出 + P1h 的 f32 bias，
  conv 输出后补 f16 Cast（注意 Cast 必须 insert 在 conv 节点之后
  保持拓扑序，append 会报 not-output-of-previous-nodes）。净收益仅
  -0.25ms。
- 板端配套：`deploy/run_engines.cpp` → N 引擎链式 runner，按张量名
  D2D 零拷贝接线（Orin 统一内存，无 host 交接）。

**端到端结果**（run_engines 链式计时，100 iters mean）：

| 配置 | 链数 | 延迟 | vs e_T6 |
|---|---|---|---|
| **e_T6 单引擎（交付）** | 1 | **38.99ms** | — |
| bb2 + hd | 2 | 39.86ms | +0.87ms |
| bb2 + det + rest | 3 | 40.12ms | +1.13ms |
| bb + hd（fpn 不回 INT8） | 2 | 40.14ms | +1.15ms |

精度门禁通过：bb+hd 两链 mini mAP 0.4170 / NDS 0.4729（v3 单引擎
0.4184/0.4723，持平）。逐张量 diff 呈已知可接受签名（P2h fpn-f16 vs
P1h fpn-int8 ~0.45 rel，同 e_T12），ego_feature_map 边界哨兵逐位一致。
注意 bb 侧 dump 只能当链内一致性证据、不能当 frame-0 真值——边界张量
`/Reshape_9_output_0` 有第 13 个消费者 `/Split` 喂 ego 链，单独存的
dump 缺帧间状态，拿它做 isolated 测试连 ego 都会毒化 0.62。

**为什么拆分不赚钱——三个根因全部定位**：

1. **backbone 历史记账错误**：t6_prof.log 中 backbone+neck 各行合计
   ~15ms/iter，与 bb2 引擎单独实测 15.25ms 一致；早年"backbone
   10.1ms"只汇总了部分行。e_T6 的 head 侧 ≈ 23.74ms ≈ 独立 head
   引擎 e_hd 的 23.04ms——**head 从来不是拆分能省的部分**，退化区在
   head 引擎里原样复现。

2. **det 侧"退化" ForeignNode 是真实访存，不是退化 fallback**：
   `deploy/census_det.py` 显示每个 det decoder 层的 attention 区间
   物化 5 个 [1,8,900,900] f16 张量（QK^T MatMul → Cast → Mul
   预 softmax 掩码 → Softmax → Cast，~65MB/层 写+读），6 层 ≈ 9ms。
   掩码在 softmax 之前 Mul 入 QK^T 输出，TRT 无法折成 V 侧融合——
   这是窗口注意力 900 token × 8 head 的模型结构级流量，拆不拆都在。

3. **map/quality ForeignNode 是 Myelin 大图回退，小图确实溶解**
   （e_rest 单引擎 10.23ms、最大 ForeignNode 仅 0.84ms），但这 ~5ms
   收益被**引擎边界税**吃光：45MB 边界特征跨引擎重读 + 双 enqueue
   CPU 开销 ≈ +1.5ms 结构性成本，且小图溶解是以把 det/map DFA 拆出
   L2 驻留环境为代价。o5 大图重规划构建日志（`rank <=
   ChoiceTuple::kMAX_RANK` 断言 + NVRTC 失败，rc=139）证明这些区在
   单引擎内无法换用更优 tactic，是 TRT 8.6 规划器的硬限制。

另证：73 个恒等 f16 Cast 全剥离（`deploy/strip_casts.py`，含 SSA 的
Cast_4/Cast_5）零收益——TRT 早已融合这类 Cast；且剥离会重命名边界
张量（w2th→w2t）破坏跨引擎按名接线。

**结论**：拆分路线实测封死。交付维持 e_T6 + libdfaplug_v8
（38.99ms / 25.6 FPS / mAP 0.4203 / NDS 0.4738）。拆分产物
（sp_backbone2 / sp_head / sp_head_det / sp_head_rest.onnx、
e_bb/e_bb2/e_hd/e_det/e_rest 引擎、run_engines runner）留档，供
未来单测 head 侧（如换 TRT 后验证退化区是否消失）复用。

### 第八轮：det 注意力 FlashAttention 化（P5，实测负收益，已关闭）

**动机**：第七轮定位 det 侧 SSA 注意力物化 ~9ms（每 decoder 层 5 个
[1,8,900,900] 级张量落显存）。P5 用单插件内核（flash-attention 式在线
softmax，S/P 全程驻留 smem 不落显存）替换 det head 的 10 个大站点
（5×[1,8,900,600] + 5×[1,8,900,900]，hd=64，scale=0.125）的 6 节点链
`QK^T MatMul → Cast(f32) → Mul(scale) → Softmax → Cast(f16) → PV MatMul_1`。
11 个 hd=32 小站点不可替换（内核 D=64 硬编码，流量也小，不值得）。

**工作**：
- 图手术 `deploy/ssa_surgery2.py`：10 站点整链 → FlashSDPA 插件节点
  （domain SparseDrive），插件输 [Q, K^T, V, scale常量]，输出沿用 PV
  MatMul_1 名字（下游 Transpose 零改动）
- ScaleSoftmax-only 变体（只融合 Cast/Mul/Softmax 三节点）先测：
  **+1.6ms，死路**——融合掉物化但引入插件边界与精度转换，得不偿失
- FlashSDPA 内核（`deploy/_fa_part.cu`，v10 = v8 + SS + FA）：wmma
  16×16×16 f16/f32acc，QT=16 行/CTA，KB=64 key 块，4 warp 各管一个
  16 列 tile，K^T/V smem 双缓冲软件流水，softmax f32 蝶形归约；
  **内核级修复全程热交换**（引擎按 名字/版本/命名空间 绑定插件，.so
  重编 + re-profile 即可，e_det_fa 引擎从头到尾没重编）

**关键实测经济学（为什么注定赢不了）**：

| 项 | 数值 |
|---|---|
| 基线 10 条链总耗时（TRT 已融合物化链） | **~2.75ms**（可收割上限，全引擎 ~7%） |
| 基线有效吞吐 | ~5 TFLOPS（13.8 GFLOP / 2.75ms，流式融合 tactic） |
| 手写 wmma 内核 | 0.71–1.03ms/站点 ≈ **~1.6 TFLOPS**（v1 为 2.72ms/站点，7×） |
| 模块级净效果（bb+det 两链 100-iter） | 35.17 vs 29.05ms = **+6.1ms** |

3 倍吞吐差距是架构性的：TRT 的链在这组形状（M=900,N=900/600,K=64 批 8 头）
上把 GEMM+softmax 融合流式执行；单 CTA wmma 手写内核没有 ldmatrix、
没有跨块深流水，上限就在 ~2 TFLOPS 附近。**修数值 bug 填不平 3 倍架构差距**
——这是 P5 停止调试的依据，不是放弃排查。

**内核开发过程实锤的坑（全部有探针证据，详见 AGENTS.md）**：
1. **wmma matrix_b col_major 在 sm_87 上 load 语义不符**（SSA7 单位阵
   探针：I@B_col maxerr 1.9，而 row_major/mma/store 全对）→ B 一律
   row_major 加载，靠暂存布局保证内存即行主
2. **v1–v3c 一直带的最大数值 bug**：pass1 的 S 散写漏 `bi*KB` 块偏移，
   15 个 key 块全部覆写同一前 64 列（replay 误差图 809832/810000 坏格、
   列 0–63 恒等于最后一块的值）→ 修复后坏格降至 151070
3. K^T 输入是 **d-major [d][key] 内存布局**（[1,H,64,Kn]），曾按
   [key][d] 误读；d-major 暂存恰好使 K^T 行主直读（与坑 1 的修法自洽）
4. k 切片推进：pass1 的 A tile（sQ 列）与 pass2 的 P tile（sS 列）必须
   随 k 步/块步推进，v1–v2 均曾固定读第 0 片
5. **数值验证方法论**：模块级 A/B 永远不干净——TRT 重融合改变上游投影
   舍入，Q/K^T 跨引擎差异即可把 S 推到 l2 0.17 级。把链内张量暴露为图
   输出 dump（`deploy/probe_fa_io.py`），相同输入 numpy 重放对拍
   （probe/replay）是唯一有效检查
6. 内核确定性已证（SSA8：双跑 device 对拍 0 diff）——replay hash 漂移
   是陈旧产物时间线混乱，不是竞态

**收尾状态**：数值还差最后一档（replay S l2rel 0.242，残余模式指向多块
pass1 尚有一处缺陷），但性能判定不依赖它，按用户决定停止调试。

**结论**：P5 关闭，负结果。交付维持 e_T6 + libdfaplug_v8
（38.99ms / 25.6 FPS / mAP 0.4203 / NDS 0.4738），未消耗全量构建名额。
det 注意力 ~9ms 物化流量的剩余出路只有：改模型结构（window attention /
token reduction，超出部署侧范围）或升级 TRT（新版有望直接融合该链）。
产物留档：`ssa_surgery2.py`、`_fa_part.cu`/`dfaplug_v10.cu`、
`selftest_fa2/3/4.cu`（合成自检 / wmma 语义探针 / 确定性探针）、
`probe_fa_io.py`、`board_ssa1-8.py`。

---

### 第九轮：B2a 量化归因（PyTorch 复刻引擎量化态）—— map 掉分不在量化，在板端编译侧

**背景**：mini map mAP 引擎 0.6471 vs FP32 0.7512（-10.4pt），det 仅 -0.5pt；
"map 口径"一节曾假设 *map 头对 fpn INT8 敏感*（依据 P2h 的 fpn 张量级 rel~0.45）。
本轮用 PyTorch 直接复刻 e_T6 的量化数值态做对拍，检验该假设。

**方法**（`deploy/infer_b2_int8fp16_mini.py`，产出 `evaldata/mini_b2`）：
- 量化路径逐行镜像 `export_v5.py`：同 ckpt(fp32) + 锁种子 + 训练管线 16 样本
  max 标定 + `INT8_DEFAULT_CFG` + `det_head_output` 保护组；head 侧量化器再
  全部禁用（T3 剥离模拟），head 计算 fp32 + attention QK^T/PV fp16（图级保真）
- **QDQ scale 哨兵**：重建 amax/127 与 `v5_P1h.onnx`（引擎源图）内 118 个
  QDQ scale 逐一比对，中位相对差 **7.8e-08**（逐位级），唯一失配是一个
  scale=1.0 的恒等占位 → B2a 与引擎量化状态同源，铁证

**结果**（mini 81 帧，配对比较）：

| 配置 | det mAP | det NDS | map mAP |
|---|---|---|---|
| A: FP32 PyTorch | 0.4256 | 0.4798 | 0.7512 |
| B2a: 引擎量化态 (PyTorch) | 0.4246 (-0.1pt) | 0.4796 | **0.7473 (-0.4pt)** |
| E: e_T6 引擎 | 0.4203 | 0.4738 | **0.6471 (-10.4pt)** |

**场景切分细化**（S1=帧0-39 / S2=帧40-80，map mAP）：

| 配置 | S1（无场景边界） | S2（边界后） |
|---|---|---|
| FP32 | 0.8164 | 0.3073 |
| B2a | 0.8130 (-0.3pt) | 0.3057 (-0.2pt) |
| e_T6 | **0.7420 (-7.4pt)** | **0.2083 (-9.9pt)** |

**结论**：
1. **量化数学无罪**——引擎同款 scale 在 PyTorch 里 map 只掉 -0.4pt
   （逐场景 -0.2~-0.3pt），引擎多掉的 ~10pt 全在板端编译/数值侧。
2. **主损伤载体 = map 头递归环路的数值退化随帧累积**：帧 0 引擎输出与
   B2a 几乎一致（top-5 分数到 3 位小数），序不变孪生率（top-10 预测按
   chamfer<1m 配对）场景内 9/10→7/10、分数差 0.02→0.06 逐帧变差，
   S1（无任何边界）即落后 -7.4pt。特征侧全程干净（ego_fpn cos 0.997）。
3. **次损伤 = 板端链缺场景重置**：`repro/mini_pipeline_v8.sh` 逐帧盲传
   `prev_map_*`，跨场景（帧 40）把上一场景的 33 个地图锚点带进新场景，
   S2 额外多掉 ~2.5pt。修法：链脚本在 `dt>2.0 or dt<0` 的帧清零
   prev_map_*/prev_det_*（对齐 tools/test_trt.py 契约）。
4. **平铺张量 A/B 被递归槽位排列主导**（P5 教训再现并升级）：递归库
   内容集合相同、槽位顺序不同（fp16 噪声翻转近值 conf 的排序），flat
   rel_l2 可以到 1.0+ 而 mAP 无损——递归输出的定位必须序不变或在链
   内、帧 0 上做。
5. **修正旧假设**："map 头对 fpn INT8 敏感"证伪——fpn 反量化（P2h 方向）
   不是 map 精度的解药；QAT 也早已证伪（drift 与 PTQ 持平，qat_summary）。
6. **交付口径修正**："head FP16"实为：图内显式 fp16 仅 attention 内层
   MatMul + kps 3D GEMM（97 个 Cast / 458 个 MatMul 中 54 个），head 其余
   fp32 存储，运行时由 onnx2engine 的 kFP16 flag 让 TRT 自选 tactic；全图
   **零** fp16 权重 initializer（68 个 int8 = backbone/neck 折叠权重链）。
   PyTorch 侧仿真若整 head autocast 会非有限溢出（TRT tactic 内部累加仍
   fp32），必须用 B2a 这种图级保真形态。

**下一步**（编译侧定位）：板端 dump-graph 引擎变体（一次整编）把 map 链
中间张量（逐层 anchor refine / attn 输出 / bank 写入）暴露为图输出，
与 B2a 帧内/帧 0 做序不变对拍，找第一个把噪声写进递归状态的站点；
先落零编译的重置修复并重测 S2。

### 第九轮补：机制确认与修复包（--f32-names / 链重置）

**机制确认（帧级几何画像，推翻"纯累积"旧叙事）**：
- 漂移曲线（`deploy/_cmp_drift_curve.py`）：eng|B2 配对 chamfer **中位数
  从 frame 0 就是 0.47m**（B2|FP32 仅 0.13m），S1 内 0.2~0.7 波动**不随帧
  增长** → 引擎 map 头每帧都有 ~0.5m 结构性几何偏差（非刚性、逐点方向
  不同），不是累积噪声；随帧变多的是尾部 7~18m 的跳轨实例（递归放大）。
- frame 0 细查（`_cmp_f0_detail.py`）：top-5 分数与 B2a 吻合到 3 位小数，
  但 `map_instance_feature` rel_l2=0.24、逐点变形 0.25~1.2m；det 头同样
  有偏差（det_cls rel 0.098、frame5 det_bbox rel 1.2）而 det mAP 无恙
  → **全头性 fp16 tactic 偏差，map 头只是付得起账的受害者**（20 点线 +
  0.5~1.5m 门限 + 33 槽递归库，比 det 的 0.5~4m 中心距阈值敏感一个量级）。
- 量级排除法：TRT fp32 tactic vs PyTorch fp32 应差 ~0.01m（实测 0.47m）；
  TRT 隐式 INT8 不发生在显式量化图的无 QDQ 层；显式 fp16 岛（54/458
  MatMul）量级 mm 级 → 只剩 **kFP16 flag 下 head 走 fp16 tactic、6 层
  decoder 逐 op fp16 舍入混沌放大**（--f32-notq 注释当时的预言）。T12 轮
  "图侧 f16 化精度零变化"旁证：决定精度的是运行时 tactic 不是图侧 Cast。
- **B2 盲传实验**（`B2_NO_RESET=1`，产出 `evaldata/mini_b2_board`）：
  PyTorch 下边界帧不重置 mAP 几乎无损（0.747277 vs 0.747296，差 2e-5）
  —— 干净数值能压制跨场景残影实例；引擎里 fp16 噪声让残影压制失灵。
  **重置修复与 fp32 修复是相乘关系**，都要做。

**修复包（`deploy/board_mapfix.py` 一键会话，凭据运行时注入）**：
1. F1 零编译：`repro/mini_pipeline_reset.sh`（边界帧跳过 prev_* 盲传，
   in_40 零历史先恢复——盲传链当年把它盖了）+ e_T6 → `vec/mini_r`；
2. F2 一次重建：onnx2engine 新增 `--f32-names <list>`（ONNX 节点名精确
   匹配强制 FP32，纯浮点护栏 + QDQ 邻接豁免 + 匹配计数；列表由
   `deploy/_gen_f32_names.py` 从 7 个 map 输出反向遍历生成，1007 层，
   已排除 backbone/neck 与显式 fp16 岛/kps GEMM——数值形态 = B2a 已验证
   的 fp32 head + fp16 attn，且避开 kps GEMM 的 NVRTC 雷）→
   `e_T6m.engine` + 重置链 → `vec/mini_m`；
3. trtexec 双引擎 profile 量速度代价；拉回两套 dump 本地评估。
预期：S1 0.742→~0.81、S2 0.208→~0.30，全量 0.647→~0.73+（B2a 上限
0.7473），速度代价看 profile（map 分支 fp32 退化 tactic，预估 +1~3ms）。

**附**：量化工具链曾随 263a1eb cleanup 误删，已从 git 恢复
（`git checkout 263a1eb~ -- deploy/{qat,ptq_sensitivity,flashmha_qkv,
group_sensitivity,task_sentinel,render_ptq_report,smoke_test,...}.py
deploy/ONNX_QUANT_STRATEGY.md deploy/artifacts/eval_grp_*` 等）；分组
det 敏感度产物同时找回，其中 `eval_grp_img_neck`（fpn 反量化）det
0.4271/0.4823 优于 fp32 对照——det 侧给结构性反量化留了充分余量。

---

## 遇到的主要坑

### 板端 TRT 8.6.1.2 NVRTC JIT 损坏

这是整轮优化最大的环境限制。板端的 NVRTC JIT 对 fp32 融合 kernel
编译必失败（`nvrtc_compile.cpp:940: CHECK(false)`），导致：
- fp32 Myelin 区域只剩退化 fallback tactic（`Tactic: 0x0`）
- 5 个 decoder 融合区（~10.4ms）被迫跑在远低于应有的性能
- 任何引入 fp32 计算路径的优化都会触发更多 NVRTC 失败

**绕开方式**：所有需要新编译路径的改动都加 fp16 Cast，让 TRT 走 fp16
JIT（fp16 JIT 是正常的）。

### Windows → Linux 换行符

pipeline 脚本从 Windows sftp 推到 Linux，`\r\n` 换行导致 bash 把
`\r` 当成路径的一部分——`cd /opt/m0/trt-dev\r` 找不到目录，
脚本第一行就退出，日志为空。看起来像"编译没启动"或"编译挂了"，
实际是脚本根本没执行。

**修复**：`sed -i 's/\r//' script.sh`

### TRT 显式 QDQ + 插件 int8 输入不支持

TRT 8.6 的 QDQ 分析器要求每个 Q 后面必须有 DQ。直接让插件吃 int8
（Q 输出无 DQ）会报错。绕开方式：用 Mul+Clip+Cast 纯算术量化替代 QDQ。
但实测无收益（见上表），所以最终未使用。

### 引擎与插件版本配对

引擎在反序列化时按插件 **名字/版本/命名空间**（"DeformableAggregation" v1
""）向 registry 要实现 —— 所以**换 .so 不用重编引擎**（P3 正是用这个机制
直接热替换 v3→v8）。反过来，语义不同却同名的插件（早期 v1 weights=
post-softmax vs v3 起为 raw logits；布局 [A,P,cams,S,G] vs [A,cams,S,P,G]）
混用会静默算错。现在交付语义只有一种：**logits 进、v8/v3 内核任意**，
两者数值 ulp 级等价（P3 已逐位验证），可互换。

### 真实数据 vs 合成数据的 ε-skip 行为差异

合成随机 logits → softmax 后权重均匀 → ε-skip 不触发 → 计时和真实
数据完全不同。所有内核优化必须在真实引擎 dump 数据上验证。

**dump 方式两代**：
- 老办法：修改 ONNX 加 Identity 节点暴露 DFA IO → 编译 dump 引擎 → run_engine --dump
- 新办法（P3，推荐）：**在插件 .so 的 enqueue 里加 env 门控 dump hook**
  （`deploy/dfaplug_dump.cu` → `DFA_DUMP_DIR=... DFA_DUMP_MAX=12 run_engine ...`）。
  不改 ONNX、不重编引擎，直接抓到引擎真实 IO（feat/shape/ssi/loc/logits/输出
  + manifest），shape/ssi 就是引擎实际传的常量。

### 引擎把 getWorkspaceSize 烤进了 plan（P3 最大坑）

e_T6 由 v3 插件构建，v3 没实现 getWorkspaceSize（基类默认 0）→ build 时
该层 workspace 需求记 0 烤进引擎 → **运行时 TRT 只会给 nullptr**。v8 插件
"TRT 不给 workspace 就回落 v3" 的静默回退在引擎内**每次都触发**，且无任何
告警 —— profile 行与 v3 逐位相同是唯一线索。

**教训**：
- 靠 workspace 的内核不能假设 TRT 会给 —— 插件自分配（懒分配只增不减），
  不重编引擎也能吃到新路径
- 引擎内 A/B 必须验证"新内核真的在跑"：debug 环境变量打印实际路径
  + profile 行对拍，缺一不可
- 静默回退是双刃剑：同名插件热替换很方便，但回退零表象

### nvcc -shared 插件必须 -lnvinfer

`nvcc -shared -Xcompiler -fPIC x.cu` 不链 `-lnvinfer` 时，trtexec
（dlopen RTLD_LOCAL）报 `undefined symbol: getPluginRegistry` 加载失败；
而 run_engine（RTLD_GLOBAL）能掩盖这个 bug —— 同一个 .so 一个工具能跑
另一个不能时先查链接。

### dump 常量必须取自源图（ssi 伪造坑）

旧 dump 转换器把 scale_start_index 写成"所有相机相同"的伪造常量 →
v3 内核对拍 l2 高达 1.0-2.5，误查两轮内核。从 v5_P1h.onnx initializer
提取真实 shape/ssi 后 bitwise 全等。重序列化出的常量一律不可信，
必须从源图 initializer 拉取。

### mat4_inv 双坑与闭环 mAP 盲区（M5 定案）

sp_modelnode 的 4x4 求逆先抄 MESA gluInvertMatrix（列主序约定 + 三项
转录错）→ 行主序 l2g 的 tmat 每帧 Z 平移恒差 -1.55m，闭环 det mAP
0.416→0.312，**M2 起潜伏三代**：f0 哨兵（恒等阵）全绿、张量 A/B 在
混沌地板（f1 起 top-300 重叠仅 30-40%）下全盲。改刚体解析逆后又一脚
踩进"numpy 二维下标验证过、C 一维扁平下标写错"（-R^T t 写到行 3
而非列 3，底行爆炸 1.5e6）。修法与完整方法论（--dump-img 字节探针、
种子实验、闭环 mAP 唯一门禁）见第十二轮；公式对拍脚本
`deploy/_inv_verify.py`。

### glibc robust mutex 属主死亡移交在长跑 Orin 上不可靠（Phase A 最大坑）

pthread PTHREAD_MUTEX_ROBUST 的 EOWNERDEAD 移交是教科书机制，实锤在本板
失效：kill -9 持锁者后 filesrc+node 双进程**永久 futex 等待**（CPU 0%、
日志冻结、gdb 双方栈都在 `__pthread_mutex_lock_full`），EOWNERDEAD 始终
不来。短压测（M1 crash 阶段）测不出——是长跑 + 特定抢占窗口才现形。
环锁整体换自研 BusLock（属主 pid+starttime 落 shm，加锁者读
`/proc/<pid>/stat` 判死/判 pid 复用后 CAS 强制接管，等待一律解锁后 10ms
轮询，全环无 condvar）；锁在 RingMeta 里 → kVersion 必须 bump（v4），
filesrc create 侧 pread 头校验版本不匹配自动重建。教训：**进程间锁的
故障移交路径必须做真实 kill -9 长跑注入，不能只信 glibc 语义**。

### 信箱冻结消息假活性 + resultmon lage（Phase A）

死节点留下的信箱消息 status 恒 NOMINAL、frame_age_ms 冻在发布时的处理
时延（~207ms）不变——单读永远"健康"。下游单读判活必须用读时活年龄
**lage = now − ts_capture_ns**（死信箱上线性增长，resultmon v3 行已加）；
验收脚本判"恢复"要 **连续两次探测 seq 递进 + 末次 lage 新鲜**——双条件
（seq>前值 AND age<1.5s）都会被"kill 前 1.4s 探测窗里旧进程多发几帧冻在
信箱"骗过（1.8s 假恢复，真实热恢复 6.1-6.4s）。

### systemd 245 StartLimit 键放 [Service] 被静默忽略

StartLimitIntervalSec/StartLimitBurst 属 [Unit] 段；放 [Service] 只有一条
"Unknown key name" journal 警告，重启策略照常、熔断**永不触发**（连杀 8
次重启计数涨到 8 不熔断）。症状反查看 journal 该警告。另外熔断触发后的
单元 restart 会被 "Start request repeated too quickly" 拒绝，恢复前必须
reset-failed——且要对**所有**可能连环熔断的单元都 reset（filesrc 失效窗
里 node exit20 循环重启会把自己也熔断）。

### msg_crc 只盖 det/map 区，全零消息 crc≠0 必须显式算（Phase B）

信箱消息 crc 只覆盖 det 容量区 + map 有效条数（不是整 struct）。SELFTEST
心跳 / LATCH 无有效帧分支 memset 全零后直接发布，crc 留 0——但全零 det 区
的 crc32 **不是 0**，读端校验必然 mismatch 整帧拒收 → 心跳在信箱上完全
不可见（FTB1 probe 空的根因）。任何绕过正常 complete_frame 填充路径的
发布点，发布前都要显式 `msg.crc = msg_crc(msg, crc32)`。同族坑：v3 健康
计数（nan_hits/div_hits/resets_60s）发布时忘了填进 msg——信箱上恒 0 而
节点日志里明明在涨，"信箱可见"的断言必须查 msg 字段不是节点内部变量。

### 注入的 NaN 被引擎 fp16 链在状态边界吸收，探测走 state 通道（Phase B）

SP_INJECT_NAN 往 mp 状态缓冲注 NaN（8 或 1024 个元素、单发或每帧），
引擎输出**始终保持有限**——NaN 在 fp16 状态边界被吸收，反馈回去的状态
变成极端有限值（>1e6）→ 探测走 state-absmax 发散通道（div），nan_hits
恒 0。这是机制不是 bug：故障注入验收要认 "DEGRADED_RESET + RESET 日志 +
探测帧 div/nan 任一 ≥1" 的通道组合证据，不能死磕 nan_hits。另注意计数
是**进程累计**：SP_FPS=1 下 81 帧清单 ~81s 枯竭 → node exit20 重启清零，
验收读计数要读探测帧不是恢复帧。

### 门禁独立 run 用 pkill 清场会撞 systemd 复活的双发布端（Phase B）

门禁/隔离 run 清场若只 `pkill -9 sp_filesrc/sp_modelnode`，systemd 的
Restart=on-failure 会在 ~1s 后把发布端**拉回同一 ring**，与独立 run 的
filesrc 形成双发布端：症状是独立 node 中途 `FATAL code=12 seq misalign`
（撞外来 seq，恰好在场景边界附近暴露）或 filesrc `claim timeout`。
清场必须 `systemctl stop` 双单元 + reset-failed，再 pkill 兜底。

### 金标比较 NaN 差值盲区（Phase B）

`fabs(a-b)` 两侧任一 NaN 时差值是 NaN，`d > tol` 恒 false → 往金标里
篡改 NaN 字节自检照样 PASS（FTB1 第一轮 FAIL 根因）。容差比较必须显式
`isnan(d) → 计超差`，"比较表达式对 NaN 恒 false" 是所有数值门禁的通杀
陷阱。

---

## 最终引擎 profile 分布 (T6 + v8 插件, 38.99ms mean)

| 模块 | 耗时 | 占比 | 状态 |
|---|---|---|---|
| backbone+neck INT8 | 15.25ms | 39% | 合理（P4 勘误，原记 10.1ms 系漏统计） |
| DFA 插件 (12 调用) | **5.0ms** | 13% | **P3 v8 双内核**（v3 为 11.3ms） |
| 退化 fp32 融合区 (5个) | 10.4ms | 27% | 不可修复（板端 TRT 限制；P4 证实拆分也无效） |
| map_head cls 区 | 4.0ms | 10% | 同上 |
| Reformat + 层间空隙 | 3.7ms | 9% | 随上述减少 |
| kps 3D GEMM | 0.6ms | 2% | 已优化 |

（v3 插件同引擎为 45.27ms；-6.3ms 全部来自 DFA。各行为 v8 profile 计占比。）

**P4 勘误**：本表早期版本记 backbone+neck = 10.1ms / 26%，系只汇总了
部分 profile 行的错误记账。P4 用 bb2 引擎单独构建实测 backbone+neck =
15.25ms（t6_prof.log 各行合计一致），head 侧 ≈ 23.74ms ≈ 独立 head
引擎 e_hd 的 23.04ms。"Reformat+空隙 7.4ms" 同样吸收了漏掉的 ~5ms
backbone 时间，修正为 ~3.7ms。正因如此，"backbone/head 分开建引擎"
从一开始就没有可收割的空间（详见第七轮）。

---

## mini map mAP 评估（口径）

T6 + v8 的 81 帧 dump 上新增 vectorized-map 指标（与 det mini mAP 同一批帧、
同一引擎输出，`deploy/artifacts/eval_t6_mini_map.json`）：

| 类别 | num_gts | AP@0.5 | AP@1.0 | AP@1.5 | AP (引擎) | AP (FP32) |
|---|---|---|---|---|---|---|
| ped_crossing | 76 | 0.3141 | 0.8934 | 0.9724 | 0.7266 | 0.8554 |
| divider | 465 | 0.5872 | 0.8469 | 0.9315 | 0.7885 | 0.8581 |
| boundary | 287 | 0.0756 | 0.4526 | 0.7504 | 0.4262 | 0.5402 |
| **mAP = mAP_normal** | | | | | **0.6471** | **0.7512** |

**口径**（全部钉死到参考实现）：
- GT：本地从 nuScenes map expansion v1.3 json 重建（`deploy/map_gt.py`，
  无 nuscenes-devkit 依赖，逐字复刻 devkit `_get_layer_line/_get_layer_polygon`
  的旋转 patch 裁剪 + patch 局部系 = lidar 系；divider = lane+road divider，
  ped = 合并同向重叠后取外轮廓，boundary = drivable(roads∪lanes) 轮廓），
  再按 eval 管线 `VectorizeMap(simplify=True)` 等价采样（simplify 0.2，
  无重采样）。
- Pred：引擎 dump `map_cls [100,3]` 过 sigmoid 为分数，`map_pts [100,20,2]`
  为绝对 lidar 系坐标直接 reshape；100 anchor × 3 类全量参与（无分数门限，
  AP 按分数排序积分完整 PR 曲线）。
- 匹配：chamfer 距离 + AP@{0.5,1.0,1.5}，逐字复用参考
  `datasets/evaluation/map/AP.py`（instance_match 返回按阈值位置的 list，
  聚合前需按阈值 re-key 并 hstack score 列——踩过一次）。
- 类序：ped_crossing/divider/boundary = 0/1/2（与 geom2anno、MAP_CLASSES 一致）。

**注意**：mini 81 帧子集数字，不能与论文 full-val 直接对比。FP32 参照在本机
P100 跑 PyTorch checkpoint（`deploy/infer_sp32_mini.py`，喂入契约与板端链
一致：场景边界重置 + t_matrix + 逐帧真实 dt；mini 有 2 个场景，边界在第
41 帧），FP32 det mini = 0.4256 / NDS 0.4798（`deploy/artifacts/
eval_sp32_mini_det.json`），FP32 map mini = 0.7512（`eval_sp32_mini_map.json`）。
**关键发现：量化代价不对称**——det 仅 -0.5pt mAP / -0.6pt NDS（与第四轮
"逐张量差大但 det mAP 持平"结论一致），map 分支却掉 **-10.4pt**，三类普降
（ped -12.9 / divider -7.0 / boundary -11.4）。map 头位于退化融合区、对
fpn INT8 敏感，与 P2h 实验的 fpn 相对误差 ~0.45 相互印证。若 map 精度重要，
优化方向是 map 分支量化策略（map 侧 fpn 反量化或单独再标定），而不是动 det。

几何抽检已通过：两城市 top 置信度（>0.9）锚点与同类 GT 的最近 chamfer 距离
0.12–1.6m，低分锚点远离 GT，无 GT 类别处最大分数 ≤0.02，坐标范围符合
lidar 系约定（col0=±15 前向、col1=±30 横向）。

---

## 下一步优化方向

### 1. 帧间流水线（吞吐 ~1.7×，不改引擎）
第 N 帧 backbone 和第 N-1 帧 decoder 在两个 CUDA stream 上重叠执行。
部署侧改动，不需要重新编译引擎。39ms 延迟不变但吞吐可到 ~44 FPS。

### 2. 全量 trainval val 验证（精度闭环）
在 6019 帧完整验证集上跑 e_T6 的 mAP/NDS。FP32 全量基线已有
(mAP 0.4135 / NDS 0.5226)。板端推理约 4-5 小时，管线已就绪。
map 侧：mini 已闭环（引擎 0.6471 / FP32 0.7512，量化代价 -10.4pt，
见 map 口径节）；full-val map 基线仍待服务端/长跑补齐——复用
`deploy/eval_t6_mini_map.py`（`--eng/--art`），换 dump 集即可。
量化敏感的新方向：map 分支量化策略（fpn 反量化 / map 侧单独再标定）
值得单独立项，mini 口径一晚可验证。

### 3. 退化融合区的出路（P4 已定论：拆分无效，此路封死）
图侧改造已证不可达（见第四轮），引擎拆分也已实测封死（见第七轮）：
bb2+hd 两链 39.86ms、bb+det+rest 三链 40.12ms，均慢于 e_T6 的 38.99ms。
det 侧"退化区"实为 SSA [1,8,900,900] 注意力物化的真实访存（模型结构
级，掩码在 softmax 前注入、不可折入 V 侧）；map/quality 侧虽在小图里
溶解（e_rest 10.23ms）但收益被边界税吃光。剩余真实方向：
- **升级板端 TRT**（见第 5 条）：大图规划器有望直接吃掉 map/quality
  回退区；届时可用留档的 sp_head*.onnx 单测 head 侧验证，无需整引擎重编。
- det 注意力物化流量（~9ms）：FlashAttention 插件路线已实测封死——可收割
  上限仅 ~2.75ms，手写 wmma 吞吐差 TRT 基线链 3 倍，模块级 +6.1ms
  （见第八轮）。剩余出路只有改模型结构（window attention / token
  reduction），超出部署侧范围。

### 4. DLA offload（需要 INT8 conv 支持）
Orin X 有 2 个 DLA 引擎，支持 INT8 卷积。把 backbone 卷积 offload 到 DLA
可释放 GPU 给 decoder。但 DLA 不支持 deformable attention 和 plugin，
需要拆分引擎（P4 的拆分工具链可直接复用）。工程量大。

### 5. 升级 TRT 版本
当前 TRT 8.6.1.2 的 NVRTC 损坏和 Myelin 退化是硬限制。升级到
TRT 9.x/10.x 可能直接解决退化区问题（5 个区 10.4ms → 2-3ms）。
需要确认新版本对该板子的支持情况。

### 6. DFA 剩余空间（P3 已完成主体）
plan+gather 双内核已把 DFA 桶 11.3 → 5.0ms。剩余空间有限：
- det（A=900,P=13）引擎内 v3/v8 都 ~0.36ms —— 特征 L2 驻留，单内核已近极限
- map 调用的 K2 gather ~0.4ms/次，再压要靠把 12 次调用合并成一次 batch
  调用（需改图/改引擎；引擎拆分已实测无收益，见第七轮）
- plan 阶段 host 化/帧间复用（活跃集帧间变化不大）的收益场景已被 v8 大幅压缩

---

## 文件清单

| 文件 | 用途 |
|---|---|
| `deploy/export_v5.py` | v5 导出脚本（logits-DF + 量化状态复刻）|
| `deploy/dfaplug_v3.cu` | v3 插件（half2+FP16 IO+inline softmax+εskip；v8 的回退内核）|
| `deploy/dfaplug_v8.cu` | **交付版 v8 插件**（plan+gather 双内核 + v3 回退 + 自分配 workspace + DFA_V8_DEBUG）|
| `deploy/dfaplug_dump.cu` | 运行时 IO dump 插件（env 门控，抓引擎真实 DFA 输入输出）|
| `deploy/test_dfa_ab.cu` | 真实 IO A/B harness（v3/v8 对拍 + 分相计时）|
| `deploy/sim_dfa_v8.py` | DFA 语义 numpy 模拟器（变体验证/权重质量分析）|
| `deploy/extract_dfa_consts.py` | 从 v5_P1h.onnx initializer 提取真实 shape/ssi |
| `deploy/trace_dfa_inputs.py` | 验证 DFA weights 输入链为 raw logits（无 Softmax）|
| `work_dirs/.../v5_P1h.onnx` | 交付版 ONNX 图 |
| `work_dirs/.../v5_P2h.onnx` | head 全 f16 图（e_T12 源，精度中性备选）|
| `deploy/make_v5_p2h.py` | P1: v5_P1h → v5_P2h（QDQ 清理 + INT8 区闭包 + f16 化）|
| `deploy/extract_region.py` | 模块抽取器（DAG 闭包 span，no-ORT 形状回退）|
| `deploy/board_mod_ab.py` 等 board_*.py | 板端 paramiko 驱动（推模块/编译/抓日志）|
| `deploy/onnx2engine3.cpp` | 工具补丁版（--f16-notq 逐层 fp16 约束，实验用）|
| `deploy/eval_t6_mini.py` / `eval_t6_mini_v8.py` | mini mAP 评估（v3 / v8 插件输出）|
| `deploy/eval_t6_mini_map.py` | mini map mAP 评估（chamfer AP，复用参考 AP.py；`--eng/--art` 可指向任意 dump 集）|
| `deploy/map_gt.py` | 无 devkit 的 map expansion GT 重建（patch 裁剪/ped 合并/boundary 轮廓）|
| `deploy/infer_sp32_mini.py` | FP32 PyTorch mini 推理（P100，dump 引擎同格式张量）|
| `deploy/eval_sp32_mini_det.py` | FP32 det mini mAP/NDS（官方 nuScenes 评估）|
| `repro/mini_pipeline.sh` / `mini_pipeline_v8.sh` | 板端 81 帧链式脚本（v3 → out_XX / v8 → outv8_XX）|
| `deploy/prep_mini_inputs.py` | mini 验证数据准备（含场景边界重置）|
| `deploy/test_dfa_real2.cu` | DFA 真实数据计时器 |
| `deploy/test_dfa_kernels.cu` | DFA 离线双内核对拍计时器 |
| `deploy/split_engine.py` | P4 一级拆分（v5_P2h → sp_backbone + sp_head，anchor 祖先闭包切法）|
| `deploy/split_head.py` | P4 二级拆分（sp_head → sp_head_det + sp_head_rest，.pre_o 形状 OVERRIDE）|
| `deploy/requant_fpn.py` | P4 fpn INT8 再量化手术（sp_backbone → sp_backbone2，净 -0.25ms）|
| `deploy/strip_casts.py` | 恒等 Cast 剥离（实测零收益已弃用，留档防重复尝试）|
| `deploy/run_engines.cpp` | P4 N 引擎链式 runner（按张量名 D2D 零拷贝 + --dump）|
| `deploy/analyze_spans.py` | trtexec profile ForeignNode 行解析 + 图侧区间 span 归因 |
| `deploy/census_det.py` | sp_head_det det 层间区段普查（SSA [1,8,900,900] 物化证据）|
| `deploy/ssa_surgery2.py` | P5 图手术：SSA 6 节点注意力链 → FlashSDPA 插件节点（10 大站点）|
| `deploy/probe_fa_io.py` | P5 探针图生成：注意力链内张量暴露为图输出（probe/replay 法）|
| `deploy/_fa_part.cu` / `deploy/dfaplug_v10.cu` | P5 FlashSDPA wmma 内核（负结果留档；v10 = v8 + SS + FA）|
| `deploy/selftest_fa2/3/4.cu` | P5 合成自检 / wmma 语义单位阵探针 / 确定性探针 |
| `work_dirs/.../sp_backbone2.onnx` 等 | P4 拆分图（sp_backbone/sp_backbone2/sp_head/sp_head_det/sp_head_rest）|

### 板端踩坑补充（P1 期间）
- `/opt/m0` 挂载 **noexec**：二进制/脚本放这里不能直接执行——脚本要
  `bash xxx.sh` 调起，新编译的二进制放 `/usr/local/bin`
- paramiko 后台启动：`setsid nohup bash x.sh > o 2>&1 < /dev/null &`，
  通道可能挂读超时（PipeTimeout）但任务已在板上跑起来，轮询标记文件即可
- **DONE marker 用绝对路径 touch**：启动脚本 `cd` 之后再 `touch XXX_DONE`
  会落到 cd 后的目录，本地 poll 盯着原目录干等（P3 实际踩过：profile
  早就 PASSED，poll 白等一小时）
- trtexec `--dumpProfile` 的计时在 `=== Profile ===` 表（Avg 列为单次），
  没有 "GPU Compute Time" 汇总行
- `python -c` 多行内联在 Windows cmd 会因引号/管道损坏 —— 一律写临时脚本文件
  （已固化进 `~/.zcode/AGENTS.md`，含完整 Windows 坑清单）


## 第十轮：motion/planning 第二引擎（MP，已交付：单时序引擎 fp16）

模型还有 motion/planning 头（时序队列 queue_length=4）。最终方案：**单时序
引擎** e_mp 吃 e_T6 输出（det/map logits+features、ego_feature_map、
t_matrix）+ 9 张外部维护的历史张量，输出 motion_cls/reg、plan_cls/reg/
status + 9 个 next_* 状态；链脚本每帧旋转状态、场景首帧(k=0,40)把零状态
喂进时序图（与感知 reset 链同约定）。forward_onnx 已 TRT 友好化（无
ScatterND、BMM 匹配、roll→cat）。~~原设计为双图双引擎（v5_mp_first 首帧
队列初始化 + v5_mp 时序）~~，用户拍板砍掉 first 图走单引擎（torch.onnx
export 只能 trace 一条执行路径才有 first 图；81 帧指标证明时序图吃零状态
的首帧数值完全可接受，见下）。

### 已完成（本地）
- **评估协议全通，无需下载数据**：motion = UniAD NuScenesEval
  （EPA/minADE/minFDE/miss-rate，car+ped，81 帧全部属于 mini_val split）；
  planning = UniAD L2(0.5~3s)+collision（PlanningMetric）。
  `deploy/eval_mp_mini.py --eng <dumpdir>`（本地 sparsedrive_deploy env）。
- **FP32 PyTorch 基线**（部署形：wrapper det/map + forward_onnx + 外部队列 +
  场景重置，P100）：car/ped EPA **0.614/0.502**，minADE 0.340/0.739，
  minFDE 0.468/1.082，miss 5.2%/14.8%；planning **L2 0.764m**、
  obj_box_col **0.242%**。dump 到 evaldata/mini_sp32_mp（25 张量/帧）。
- **v5_mp.onnx / v5_mp_first.onnx**（deploy/export_mp_onnx.py，opset13，
  attention fp16 导出补丁，双图同 v5 惯例）：本地 onnxruntime 对拍
  6 案例（含两场景首帧）**全绿**：浮点输出 rel_l2≈0（maxabs≤3.4e-4），
  int 状态精确。
- 板端包：`deploy/board_mp.py`（push/tool/engine/chainr/chainmp/prof/fetch）
  + `repro/mini_pipeline_mp.sh`（状态旋转 + 场景重置 + 状态快照落盘 +
  静态 manifest.tsv）。

### 板端实测（2026-10-04，单时序引擎定案）
- **e_mp.engine**：v5_mp.onnx `--fp16 --ws-mb 2048`（onnx2engine_f32），
  26,573,844 B，md5 `5e4a48ab05d82227ae769c2506fd1a76`；19 入 + 14 出。
  trtexec 100 iters：GPU compute **mean 8.74ms** / p90 8.79 / p99 8.83。
- **81 帧闭环链全过**：reset 链（e_T6 场景重置）→ mp 链（状态外旋 +
  首帧零状态 + 18 张量/帧 dump，留档 evaldata/mini_mp_eng）。
- **指标 vs FP32 PyTorch 双图基线**（mini 81 帧，越低越好除 EPA）：

  | 指标 | FP32 基线 | 板端 fp16 单时序 | Δ |
  |---|---|---|---|
  | car EPA | 0.614 | **0.5939** | -0.020 |
  | ped EPA | 0.502 | **0.5094** | +0.007 |
  | car minADE/minFDE | 0.340/0.468 | 0.3461/0.4756 | ≈ |
  | ped minADE/minFDE | 0.739/1.082 | 0.7533/1.0915 | ≈ |
  | miss car/ped | 5.2%/14.8% | 5.43%/14.36% | ≈ |
  | planning L2 | 0.764m | **0.7434m** | -0.02 |
  | obj_box_col | 0.242% | **0.161%** | 略优 |

- **结论**：fp16 单时序引擎与 FP32 双图基线 mini 口径几乎持平（EPA 差
  ≤0.02，planning 反而略优）——**时序图吃零状态的首帧约定成立**，first
  图无存在必要；量化（int8）照旧不做（map 头教训，8.74ms 已够小）。
- 板端踩坑：run_engine 输入目录必须含 manifest.tsv（裸 .bin 报
  `bad inputs dir`），见 AGENTS.md 工具链坑。

### 踩坑
见 AGENTS.md「ONNX 导出/对拍坑（MP 第二引擎实测）」（10000**tensor CPU
常量炸 folding、onnxsim 剪未引用输入、quality=[900,2]、dump 带 batch 维、
roll→cat）。

## 第十一轮：map 掉点终审归因（FPN 注入实验 + 拆分链 mAP 复活）

背景：交付引擎 mini map mAP **0.6471** vs FP32 PyTorch 0.7512（-10.4pt），
此前账记在"量化代价"上。本轮把它彻底审清。

### 归因过程（含作废项）
- **V-A** fp16 注意力 rig 无罪（帧0 rel 与 fp32 注意力完全一致 1.16%）。
- **V-B** "仅 backbone+FPN 量化"重导出（v5_P1h_bf.onnx，55 Q 站点，
  trunk scale 与交付逐位一致）在板上 Myelin 崩
  （matrix_multiply_unroll_1 signal 700），--noTF32 不救；QDQ 是融合锚点，
  去掉后选中带 bug 的 unrolled MM 内核。
- **V-D** 图级剥 3 对输入 QDQ（q3）帧0与交付逐位一致（rel 0.0016）、
  map 0.6465 —— TRT 8.6 不认图结构剥除，此路封死。
- **归因 rig 作废**：MAP3 站点路径写错（7 步进扁平结构 + 命名空间差异），
  fix3/wonly3/engine_sim 三轮"归因"实为同一配置（trunk int8+头部浮点，
  全部 ~1.16% 好）——从未复现引擎损伤。
- **feat_flat_f16.bin 判废**：P3 `Reshape_9_output_0.__dump.bin` 的 fp16
  派生，而该 dump 已证非帧 0（旧坑再踩）；真帧 0/81 帧边界用 bb2 链
  `run_engines e_bb2 --dump` 重取（帧0哨兵：链式输出 vs outv8_00 ~7%
  build 间差异内）。
- **用户判决实验（帧0+81帧）**：把引擎真实 int8 trunk 的 col_feats 边界
  （vs rig FPN 偏差 **13%**，帧0/40/80 稳定）逐帧注入 MTQ PyTorch 模型跑
  81 帧递归链 → map mAP **0.7420**（干净 rig 0.7494）——13% 的 FPN 偏差
  被头部完全吸收，**backbone/FPN 量化无罪**。
- **拆分链 map 复活**：P4 留档的 bb2+hd 81 帧 dump（vec/mini/outs_*，
  帧0与今日 run_engine2 链 rel 0.0011 验明正身）评 map mAP =
  **0.7477**；out5（全图变体）0.6459 ≈ 交付 0.6471。

### 结论
- **"量化损失"实为全图编译 tactic 损伤**（第九轮"退化区只在全图编译语境"
  的 mAP 级确认）：同一套图与量化配置，全图编 ~0.646，bb2+hd 拆分编
  **0.7477**，PyTorch 0.7494~0.7512。det 侧几乎无感（0.4170 vs 0.4184，
  P4 门禁已过）——map 头（20 点线 + 0.5~1.5m 门限 + 33 槽递归）是
  全图 fp16 tactic 舍入混沌的唯一重灾户。
- 排除清单：fp16 注意力（V-A）、trunk int8（注入实验）、头部 3 隐藏
  int8 站点（权重本就是 float，V-D 位级不变）、头部权重"破坏"（_deq
  审计=原始浮点）、DFA 插件（det 无感）。
- **修复 = 拆分链交付**：bb2+hd 39.86ms（vs 单引擎 38.99ms，+0.87ms）
  换 map mAP +10.1pt。V-B 的 bf 单图路线作废（Myelin 崩且无必要）。

### 定案（2026-10-04，用户拍板：拆分链作为感知模块最终交付）
最终交付组合 = e_bb2 + e_hd + libdfaplug_v8 + run_engine2（D2D 链）。
板上 81 帧链实测（mods/mini_pipeline_sp2.sh，dump 留档
evaldata/mini_sp2_81）：**39.86ms / det mAP 0.4160 / NDS 0.4729 /
map mAP 0.7481**（map 分类 AP：ped 0.8463 / divider 0.8598 /
boundary 0.5382）。单引擎 e_T6+v8 转为回退组合。遗留：M 流水线
（sp_modelnode/shm 链路）按单引擎集成，切双引擎链是部署集成后续工作。

## 第十二轮：M5 闭环集成与 -10pt 退化根因定案（2026-10-04）

### 现象
M5 把拆分链（e_bb2+e_hd）接入真实闭环节点 sp_modelnode 后复测 mAP：
**det 0.3124**（离线基线 0.4160，-10.4pt）。三组对照（legacy graph
0.3124 / no-graph 0.3144 / fps=5 0.3136 / M5 拆分 0.3124）全部同病
→ 节点侧通病，与图/流水线/拆分无关；map 类不受影响。

### 排查（假线索全列，防再走）
- 评估装配/口径：mini_sp2_81 sanity eval 精确复现 0.4160 → 评估无罪
- CUDA graph / 流水线 parity 复用：no-graph、fps=5 对照无改善 → 排除
- tmat/dt/proj 的 **numpy 复刻对拍"全对"** → 假阴性（复刻验证的是
  自己的理解，不是节点的字节输出，见坑）
- 节点 f0 输出逐张量干净、in_00 状态逐位零 → f0 哨兵全绿（盲区实证）
- **种子实验**：节点 f0 状态塞进离线链 in_01 → f1 conf 头健康
  （0.9326）→ 状态干净，毒在节点 f1 执行侧（meta 计算路径）
- **--dump-img 直接抓字节（实锤）**：proj |diff|=0，tmat[2][3] 差
  **-1.550631**（每帧恒定）→ 唯一嫌疑 mat4_inv

### 根因（两度踩坑，都在 16 个扁平下标上）
1. **潜伏坑（M2 引入）**：mat4_inv 抄 MESA gluInvertMatrix——列主序
   约定喂行主序 l2g，且抄错 inv[9]/inv[13]/inv[14] 三项 → 非复位帧
   tmat 的 Z 平移每帧恒差 -1.55m，实例锚点时序投影全错 → f1 起 det
   递归受损（top conf 0.9336→0.83、mAVE 0.398→0.89、mAP -10pt）。
   M2-M5 全部闭环 mAP 从未实测，f0 哨兵（恒等阵）+ 张量混沌地板
   （f1 起 byid rel 0.57、top-300 重叠 30-40% 属常态）完全掩盖。
2. **修复引入坑**：改刚体解析逆时 numpy 用二维下标 `o[0,3]` 验证
   （对），C 转录把 -R^T t 写到行 3（下标 12/13/14，错），tmat 底行
   爆炸 1.5e6 —— probe 立即抓住。
3. **定版**：刚体逆平移在**列 3**（下标 3/7/11），行 3 恒 [0 0 0 1]；
   C 一维下标口径逐句对拍 numpy 4.8e-13（deploy/_inv_verify.py）。

### 修复后闭环实测（81 帧，_m5_rerun.py 口径）
| 组合 | det mAP | NDS | mAVE | map mAP |
|---|---|---|---|---|
| legacy e_T6 单引擎（pipe） | **0.4210** | 0.4740 | 0.4164 | - |
| M5 拆分 e_bb2+e_hd（graph，bb2 14.95+hd 20.0ms） | **0.4177** | 0.4735 | 0.4002 | **0.7478** |
| 离线基线 | 0.4160~0.4203 | 0.4729 | 0.3982 | 0.7481 |
| 修复前节点闭环 | 0.3124~0.3144 | - | 0.89 | 不受影响 |

**M5 门禁通过**：拆分链闭环精度与离线基线对齐（det +0.0017 /
map -0.0003）。工具沉淀：`deploy/_m5_rerun.py`（复测驱动）、
`_probe_dumpmeta.py`/`_probe_cmp.py`（meta 字节探针）、
`_inv_verify.py`（公式对拍）、`_seed_f0.py`（种子实验）、
`_mk_evaldir.py`（eval 组装）。板端复测 dump 留档
preproc_ref/m3fix、m5fix → evaldata/mini_m3fix_81、mini_m5fix_81。

### 教训（进"遇到的主要坑"）
- 递归链数值路径（投影/tmat/dt/状态旋转）的任何改动，**唯一可信
  门禁 = 端到端闭环 mAP**；元素/张量 A/B 在混沌地板下是盲的
- 节点喂引擎的数据用 --dump-img 抓字节对拍，numpy 复刻不算数
- 4x4 逆不要抄 MESA：刚体阵用解析逆，且必须按目标语言扁平下标复算

### M6a：MP 二级引擎接入节点（2026-10-04 同日完成）
`sp_modelnode --mp e_mp.engine` 三级链（bb2 15.0 + hd 19.9 + **mp
8.85ms**，e2e p50 48.1ms）81 帧闭环跑通。补齐了节点 mp 状态输入的
缓冲分配（history_*/prev_* 单缓冲双 parity 同址，复位模板 D2D +
反馈外旋写 dev[0] 同流定序；零填口径与离线权威 mini_mpstate0 逐字节
一致：8 张全零 + prev_instance_id 全 -1）。
- **闭环端指标**（eval_mp_mini，dump 留档 evaldata/mini_m6fix_mp）：
  EPA car/ped **0.6072/0.5112**、L2 **0.7456**、obj_box_col 0.161%
  （离线链参考 0.5939/0.5094/0.7434/0.161%，量级过门）
- **同源参考对拍**（链脚本以节点 M5 dump 为 SRC，SRC 符号农场
  mpSRC/outv8_XX → m5fix_out/out_XX）：f0 规划链 rel ≤0.5%、状态
  period/ego_anchor 位级一致；**全 81 帧 plan 侧 ≤0.5%（含 k=40
  边界，ego_period 100% 匹配）**；motion 侧 f1+ rel 0.25→1.0、
  id 匹配降至 ~40% —— 混沌地板常态，端指标为准
- 排障中记档的坑：mp 状态输入漏分配缓冲（"无映射"退出）、驱动脚本
  格式串漏 %s 致 --mp 静默丢失（先验日志指纹再收数据）、板端
  bash -c 嵌套单引号拆断致 BOUNDS='0 40' 吃成 "0"（k=40 复位丢失，
  对拍 f40 ego_period 0% 匹配直接暴露）

### M6b：结果信箱 v2（motion/plan 下行）—— 2026-10-04 完成
sp_result.h kVer=2：v1 前缀不动，尾部 append motion_cls[900,6] /
motion_reg[900,6,12,2] / plan_cls[18] / plan_reg[18,6,2] /
plan_status[10] / final_plan[6,2] / t_mp / cmd。节点发布块直拷 e_mp
输出 + 便捷解码 final_plan（按 --cmd 取 6 模式 argmax → plan_reg 行，
口径 = PlanningDecoder.select；节点无运行时 cmd 源——ego_status 是
accel+rot_rate+vel+steer 不含 cmd，真 cmd 部署上来自车辆接口，下游可
按全量 plan_cls/reg 自行重选）。sp_resultmon 升级打印 v2 plan 摘要。
**门禁 12/12 PASS**：节点 --mp --serial 12 帧，sp_resultmon 逐帧捕获
信箱 v2 final_plan，本地 numpy 按同帧 plan_cls/plan_reg dump 同 cmd
复算 —— mode 全部一致（15/17），maxdiff ≤ 5e-4（%.3f 文本舍入内），
t_mp 8.78ms 在报。排障记档：mon 名字自带 sp_res_ 前缀（传错静默空等）、
重定向文件块缓冲被 pkill -9 丢弃（stdbuf -oL + SIGTERM 双保险）。


## 第十三轮：M7 双流重叠实测负收益（2026-10-03 定案，已优雅退出）

**方案**：sp_modelnode 加 `--dual`——bb2(k) 移到独立流 eng_streamB，
hd(k)+mp(k) 留 eng_stream（H 流），跨流事件定序：B 流守卫 = preproc 完成
(ev_preB) + 边界槽 p 上一读方(k-2 帧)输出 D2H 完成(ev_post，防覆写仍在读
的 col_feats)；H 流守卫 = 输出槽复用(ev_post) + 跨流 ev_b2(col_feats 就绪)。
逐帧交错后 GPU 侧自然重叠 bb2(k+1) ∥ hd(k)+mp(k)，帧周期 ≈ max(bb2, hd+mp)。
graph 按流各捕（gB 捕在 B 流，gH 捕在 H 流），warmup 双流各预热。

**实测（Orin X iGPU, 81 帧）**：

| 模式 | fps | bb2 | hd+fb | mp | pre | e2e@5fps |
|---|---|---|---|---|---|---|
| 单流 pipe+graph | 22.48 | 15.42 | 19.90 | 9.19 | 6.11 | 48.35ms |
| **双流 --dual** | **23.16** | 16.33 | 32.71 | 17.67 | 19.02 | 48.33ms |

**结论：+3%，远低于 25fps 目标，交付维持单流**。原因：iGPU SM 无余量，
两 lane 抢同一批 SM 互相膨胀（hd 19.9→32.7、mp 8.8→17.7、前处理 4.3→19ms），
每帧 ~44ms 的 SM 总工作量才是吞吐天花板（1/44ms≈22.7fps），并发不创造
SM-秒。计划中的"上限 max(15.8,32.7)≈30fps"隐含了"lane 不争用"假设，
对 iGPU 不成立。**--dual 保留为实验开关**（精度已验证无损），换硬件/换 TRT
若争用 profile 变化可复测。

**精度门禁（dual5 全过）**：det mAP 0.4177（=M5/M6）、map 0.7485（基线带
0.7478-0.7481）、EPA 0.6048/0.5023、L2 0.7377、obj_box_col 0.161%。

**判别实验（重要方法学）**：dual dump 与 M6a 基线逐字节比对失败
（1532/1620），一度疑双流引入数值差异；**用当前二进制原模式（串行）重跑
对照，与基线同样差 1532** —— 引擎链路 run-to-run 本就非 bit 确定（f0
det_cls logit 差 ~1e-2，TRT tactic 归约/原子序 + 递归链放大），跨 run
bit 比对必然失败，只有容差与 mAP/EPA 口径有效。已记入 AGENTS 精度坑。

**M7b（e_mp graph）按判据跳过**：节点 mp 段 8.85ms vs 引擎单跑 8.74ms，
发射开销 ~0.11ms < 1ms 阈值，graph 化无肉。

**遗留方向**：吞吐要破 22.7fps 天花板只有减小每帧 SM 工作量——DLA offload
int8 bb2（需验证 DLA INT8 conv 支持与 plugin 边界）或升 TRT 重 tactical。

## 第十四轮：M8 设备池零拷贝采集——iGPU 平台三路全灭（2026-10-03 定案，已优雅退出）

**方案**：dmabuf/NvBufSurface 风格零拷贝采集的 CUDA 模拟器——发布端
sp_filesrc `--dma` 建每槽一块的设备内存池、fd 经 UDS SCM_RIGHTS 一次性分发；
消费端 sp_modelnode `--dma` cudaExternalMemoryImportFd 映射设备指针直喂
现有 preproc kernel（kernel 零改动）；shm payload 不填，校验和对 host 源自
算；槽位回收复用 bus claim/release+fence 纪律（RingMeta 升 v2 带 dma 几何
尾字段，双向 guard：--dma 节点遇未注册池拒绝、非 dma 节点遇 dma 池拒绝）。

**平台实测（Orin X iGPU, CUDA 11.4/JetPack 5, 全部有探针证据）**：

| 路径 | 结果 | 证据 |
|---|---|---|
| cuMemCreate（VMM 分配） | 尺寸按 granularity（本板 2MB）对齐后成功；未对齐=invalid argument（M8 首败根因） | probe_vmm.cu |
| cuMemExportToShareableHandle（POSIX fd） | 成功（rc=0） | probe_vmm.cu |
| **cuMemMap** | **invalid argument——primary ctx 就绪、2MB 对齐仍失败；VMM 映射 dGPU-only，iGPU 上分配了也无法映射** | probe_vmm2.cu |
| **cudaIpcOpenMemHandle** | **invalid argument（两个真实进程，flags=0 与 LazyEnablePeerAccess 均试；GetMemHandle 成功、句柄送达正常）** | sp_filesrc/sp_modelnode 两进程实测 |
| NvBufSurface（真 dmabuf 分配器） | 本镜像无运行库无头文件（裸工业版无多媒体包），无法模拟 | _m8_nvbuf.py |

**结论：交付维持 mapped-shm + cudaHostRegisterMapped（M1.5 机制）**。它
本来就是 Orin iGPU 唯一的跨进程零拷贝路径：unified DRAM 下 host 映射内存的
设备读就是设备侧读，12.96 GB/s 基线即平台带宽。"设备池绕开 C2C 读"的收益
在 iGPU 上不存在对应机制。dma 模式等价门禁（逐字节 preproc 输出比对）按
计划前置条件失效，不适用。

**留档（真实相机迁移点）**：`sp_dmapool.h/.cpp`（fd 池 + 对齐修复 +
OpaqueFd 导入）、`sp_bus` v2 dma 尾字段 + `claim_of`/`commit_dma`、
`sp_filesrc`/`sp_modelnode` 的 `--dma` 与双向 guard、探针
`probe_vmm.cu`/`probe_vmm2.cu`（`deploy/_m8_probe.py` 板端驱动）、门禁
`_m8_gate.py`。base 模式门禁通过（81 帧、pre p50=1.45ms、信箱 81 写），
RingMeta v2 双向 guard 生效被 dma 失败路径反向验证。换 JetPack 多媒体镜像
+ 真 ISP 相机时：池由 NvBufSurface 分配产真 dmabuf fd，消费端仅换
handle 类型 OpaqueFd→DmaBufFd，池/fence/节点机制全部复用。

**附带坑（已记 AGENTS）**：(1) cuMemCreate 尺寸必须按 granularity 对齐，
未对齐报 invalid argument 且错误信息不含"对齐"线索；(2) driver API 与
runtime 的 context 不同步，cuMemMap 前需 cuDevicePrimaryCtxRetain +
cuCtxSetCurrent（本例主因是 iGPU 不支持，但 ctx 缺失同样报 invalid
argument，判障时两者都要排除）；(3) fork 子进程里 CUDA 调用静默失败
（probe_vmm IPC 段 got=0 假象），进程间 CUDA 共享探针必须 fork+exec；
(4) 驱动 API 链接要 -lcuda（nvcc 不会自动带）。

## 第十五轮：M-PROD Phase A 量产化加固——watchdog / systemd 监督 / fail-visible 信箱（2026-10-05 交付）

按用户批准的量产化路线（2026-10-03）交付第一阶段三件事：**①节点内
watchdog**（TRT hang/阶段卡死从"静默挂死"变"FATAL 退出→systemd 拉起"）、
**②systemd 监督**（模板单元+重启策略+熔断，按多模型共置模板设计）、
**③信箱 v3 fail-visible**（下游要么新结果、要么明确知道旧化了多少）。

### ① 节点内 watchdog（sp_watch.h）

20ms 监视线程，每阶段环形记录最近耗时，deadline = **max(3×p50, 2×p99,
2000ms)**（2s 地板：iGPU 与其他进程共卡，post 段实测过 503ms 单次抖动，
500ms 地板误杀）；采集/IO 段与 init 窗口（watch_start 后第一个 touch 前）
不设防（闪存 GC 写停 0.5-2s 是常态）。超时 `FATAL code=13 stage=<名>
stalled_ms= dl_ms= seq=` 后 `_exit(13)`，刻意不做任何清理（CUDA context
可能已坏，清理会二次挂死）。测试钩子 `SP_WD_TEST_STALL=<stage_id>` 在
帧 5 对该段睡 5s，专供故障注入。

### ② systemd 监督（deploy/systemd/）

模板单元 `sp-filesrc@.service` / `sp-modelnode@.service`（%i=环名）：
Restart=on-failure + RestartSec=1s，StartLimit 120s×5 **在 [Unit] 段**
（放 [Service] 被 systemd 245 静默忽略——连杀 8 次不熔断的根因），
日志 append:/var/log/sp/{filesrc,node}-%i.log + logrotate（日切 50M×7
copytruncate），目录由 tmpfiles.d 保证。env 经 `/etc/sp/%i.env`
（EnvironmentFile 每次 start 重读 → 改参数 restart 即可）。node
Requires+After filesrc（不反向拉活：--loop 下源枯竭 node 必须 exit 20
留在重启环里等 filesrc 修复）。常驻 env 三件套：`--loop --frames
1000000 --no-dump`（缺一：钳 81 帧 rc0 退 / dump 7.3MB/帧写满盘）。
SIGTERM 处理后两进程 stop ~0.9s（原 SIGTERM 不响应要吃满 TimeoutStopSec
30s）。安装/健康断言/卸载：`deploy/_prod_install.py`（tmpfiles →
daemon-reload → enable --now → 75s 内 probe 信箱 NOMINAL + 单元态检查）。

### ③ 信箱 v3 fail-visible（sp_result.h v3 / sp_resultmon）

消息尾段追加 status/reason/wd_stage/last_valid_seq/frame_age_ms/
resets_60s/nan_hits/div_hits（Phase A 恒 NOMINAL，DEGRADED/SELFTEST 是
Phase B 钩子）。**frame_age_ms 是发布时冻结的处理时延，不是读时年龄**——
死节点信箱永远显示 NOMINAL+207ms；resultmon v3 行加 **lage=**（读时活
年龄 now−ts_capture_ns，死信箱线性增长）——下游单读判活/告警的唯一
可靠信号。ts_capture_ns 用节点 acquire 墙钟（manifest ts_ns 是 2018
采集纪元，相减会钳 65535）。

### 顺带修复的量产级缺陷

- **环跨进程锁换自研 BusLock**（kVersion 4）：robust mutex 属主死亡移交
  在长跑板上不可靠（kill -9 持锁者 → 双进程永久 futex 等待、EOWNERDEAD
  不来，gdb 双方 `__pthread_mutex_lock_full`）；新锁属主 pid+starttime
  落 shm，判死后 CAS 接管，等待=解锁+10ms 轮询，全环无 condvar
- **持久环生命周期**：kill -9 的消费者 cid 永久占槽 → register 时心跳
  >10s 判死接管（释放槽引用）；中流接入 seq 基线 resync（期望=基线+
  submitted+1）+ manifest 按 (seq−1)%nman 配对（k%nman 中途接入配错帧）
- **退出码契约**：node 0/10 init/11 engines/12 resources/13 watchdog/
  20 源枯竭；filesrc 0/10/20；全部经 fatal_exit 统一打
  `FATAL code= stage= seq= detail=`

### 板端门禁（不回归）

| 指标 | Phase A | m7fix 基线 | 判定 |
|---|---|---|---|
| e2e (ms) | 48.06 | 48.35 | ✓（bb2 14.98/hd 19.89/mp 8.85）|
| map mAP | 0.7463 | 0.7485 | ✓ 跨运行抖动类内 |
| EPA car/ped | 0.6039/0.5080 | 0.6048/0.5023 | ✓ |
| planning L2 | 0.7398 | 0.7377 | ✓ |
| obj_box_col | 0.161% | 0.161% | ✓ |

### 故障注入验收（`deploy/_prod_ft_a.py`，4/4 PASS + 恢复全过）

| 用例 | 注入 | 验收证据 |
|---|---|---|
| FT1 | kill -9 node MainPID | 信箱冻结 6 次探测（seq 恒 8209/9416，lage 419→5304ms 增长）→ **6.1s** 新进程发布（lage 跌回 ~330ms）< 15s 线 |
| FT2 | SP_WD_TEST_STALL=2 | log `FATAL code=13 stage=hd stalled_ms=2001 dl_ms=2000` + journal status=13 → systemd 拉起 4.3s 回 NOMINAL |
| FT3 | env manifest 路径错 | filesrc `FATAL code=10 stage=init`（manifest missing）→ **5.2s 熔断 failed**（5×退出+1s）；node 侧 inactive 不悬空 |
| FT4 | 循环 kill -9 node | 第 5 杀触发 StartLimit → unit 保持 failed、pid 无（不再无限重启）|

恢复判据 = 连续两次探测 seq 递进 + 末次 lage<1.5s（教训见"遇到的主要坑"：
单读 seq>pre 被 kill 前 1.4s 窗里旧进程残帧冻结值骗过 1.8s 假恢复）。

### 产物

`deploy/prepost/`：sp_watch.h（新）、sp_bus.h/.cpp（BusLock v4）、
sp_result.h（v3）、sp_resultmon.cpp（v3+lage）、sp_modelnode.cpp
（watchdog/退出码/resync/快停）、sp_filesrc.cpp（attach-or-create/
exit10/快停）；`deploy/systemd/`：2 单元 + m3.env + sp.conf +
sp-logrotate；`deploy/_prod_install.py`（安装/健康/卸载）、
`_prod_ft_a.py`（FT 套件）、`_bx.py`（板上任意命令）、
`_mprod_mkmap.py`（map 评测组装）+ evaldata/mini_mproda_map。
板上面貌：`systemctl status sp-modelnode@m3` 常驻，48.06ms/帧，
FATAL/熔断/冻结全部可见。

### Phase B 预告（下一阶段）

④ 状态发散/NaN 运行时探测 + 自动模板复位（resets_60s/nan_hits/div_hits
已留计数位，DEGRADED_RESET/DEGRADED_LATCH 状态位已留语义）+
--selftest 开机自检（帧 0 金标 hash 门禁，失败阻止 ACTIVE）。

## 第十六轮：M-PROD Phase B——运行时发散探测/自动模板复位 + 开机自检金标门禁（2026-10-03 交付）

量产化第二阶段两件事：**④运行时发散/NaN 探测 + 自动模板复位**（检测
即处置：复位→标志→再犯熔断，不做无限复位循环）、**⑤开机自检**（帧 0
金标容差 + 引擎/插件指纹，失败**阻止 ACTIVE**）。

### ④ 运行时探测 + 处置状态机（sp_safety.h/.cu）

每帧在完成帧收割处跑全部检查（µs 级宿主循环 + 每态一个设备 absmax
kernel）：

| 检查 | 对象 | 判据 |
|---|---|---|
| 非有限值 | det_cls/det_bbox/motion_reg/plan_reg/plan_status | 任一 NaN/Inf |
| 检出几何 | 解码后 DetOut | \|x\|,\|y\|>100m；w,l,h∉(0,20m] |
| 检出数漂移 | n_det 滚动 100 帧 | z-score>6（std≈0 跳过） |
| plan 运动学 | final_plan 6 点 | \|a\|>15m/s² 或 \|κ\|>1.0，**conf≥0.5 才查** |
| 状态发散 | 全部时序状态缓冲 | 设备 absmax 非有限或 >1e6 |

plan 检查必须按置信门控：板端实测 conf=0 的 argmax plan_reg 是噪声，
姿态超界属正常分布，无条件检查会在 scene-1 真实数据上 60s 攒 6 次
复位误报 → LATCH 误熔断（GATE 前必须先做无注入 3min 稳态观察）。

处置状态机：单帧异常 → 模板复位该帧状态（d_rst/d_mrst 异步 D2D 排
eng_stream，流水线深 2 → 脏状态还会喂 ≤2 帧、各自被探测计数）+ 信箱
status=DEGRADED_RESET；60s 窗复位 >5 → **LATCH**：持续发布 last_valid
+ 真实旧化（age 每 200ms 重算），2s 宽限（SP_DIV_GRACE_MS）后 exit 14
交 systemd；120s 内 5 次由 StartLimit 熔断转人工。env 旋钮 SP_DIV_*。

关键实测：注入的 NaN 被 mp 引擎 fp16 链在状态边界吸收（输出保持有限、
反馈状态变极端值）→ 探测走 state-absmax 通道，nan_hits 不动——见
"遇到的主要坑"。absmax 用 atomicMax(int 位型) 单标量 D2H；per-parity
标量避免 submit(k) 覆盖 complete_frame(k-2) 正在读的结果。

### ⑤ --selftest 开机自检（金标容差 + 指纹）

warmup 后、graph 捕获前，eager 跑帧 0（=模板零态）流水线：读 manifest
目录的 6 路 NV12，输出 9 张量与金标比容差；指纹 md5(bb2/main/mp 引擎 +
插件) 对 meta.txt。金标标定 `_prod_golden.py gen`：独立进程 10 次
`--selftest-dump`（attach 活 ring 不消费），tol = max(3×跨运行最大绝对
偏差, 1e-3 地板)——**不存 hash**（M7：跨运行非位确定，hash 门禁必误报），
指纹+容差双保险。

退出语义（spec 决策 3）：基础设施坏（meta/tol/金标缺失）rc=2 →
fatal_exit(15)；容差不过 rc=1 → **阻止 ACTIVE**：节点存活、1Hz
SELFTEST_FAIL 心跳（seq=0/det 清零/last_valid=0）、不发布有效结果，
SP_SELFTEST_OVERRIDE=1 放行（继续发布但每帧标 SELFTEST_FAIL）。
自检帧跑完必须 zero_states_all（污染时序状态会带进正式循环）。

### 故障注入验收（`deploy/_prod_ft_b.py`，3/3 PASS + 恢复全过 + OVERRIDE 过）

| 用例 | 注入 | 验收证据 |
|---|---|---|
| FTB1 | 金标 det_cls 篡改 4B NaN → restart | probe 抓到 SELFTEST_FAIL/reason=4/**lage 33→344ms 心跳活年龄**、last_valid=0、节点 active；还原金标 5.5s 回 NOMINAL |
| FTB2 | SP_FPS=1 + SP_INJECT_NAN=6 单发 | 信箱 DEGRADED_RESET（探测帧 div=1）+ RESET 日志×2 + **11.9s** 回 NOMINAL |
| FTB3 | SP_INJECT_NAN=0 PERIOD=1 连发 | 60s 窗 6 复位 → LATCH 日志 + exit14 + journal status=14 → StartLimit 熔断 failed 保持 |
| OVERRIDE | FTB1 注入 + SP_SELFTEST_OVERRIDE=1 | 信箱 SELFTEST_FAIL + last_valid=26179（有效结果继续发布、状态标明） |

### 精度门禁（两次独立板端 run，不回归）

| 指标 | mprodb | mprodc | Phase A (mproda) | m7fix 基线 | 判定 |
|---|---|---|---|---|---|
| map mAP | 0.7481 | 0.7483 | 0.7463 | 0.7485 | ✓ |
| EPA car/ped | 0.6098/0.5089 | 0.6035/0.5028 | 0.6039/0.5080 | 0.6048/0.5023 | ✓ 区间重叠 |
| planning L2 | 0.7481 | 0.7343 | 0.7398 | 0.7377 | ✓ 跨运行散布 ±0.007 |
| obj_box_col | 0.242% | 0.161% | 0.161% | 0.161% | ✓ 同上 |

（81 帧 mini 的 L2/col 跨运行散布 ~±0.01/±0.08pp——mprodb 单看 L2/col
像劣化，第二次 run 即回基线类内；门禁结论必须两次独立 run 交叉。）

### 产物

`deploy/prepost/`：sp_safety.h/.cu（新）、sp_modelnode.cpp（B1 检查块/
复位/LATCH/SP_INJECT_NAN/--selftest/--selftest-dump/--golden）、
sp_resultmon.cpp（v3 健康计数真实填充）；`deploy/_prod_golden.py`
（金标标定/篡改/校验）、`_prod_ft_b.py`（FTB 套件）、
`_mprod_mkeval.py`（map+mp 评测组装泛化）、`_mprod_gate.py`
（MPROD_TAG 参数化 + systemctl stop 清场）。板上
/opt/m0/trt-dev/golden/m3/{frame0×9, tol.txt, meta.txt}（0444）。

## 第十八轮：DFA gather 向量化 v11 插件（2026-10-06 交付，门禁全过）
（编号跳过十七，预留给暂停中的 M-PROD Phase C 收尾）

来源：H:\projects\SparseDriveV2 部署调研（71.30→22.92ms 战役）。结论：
其 MHA flash-tile 融合**不可迁移**（它们的 TRT 注意力链 ~0.2 TFLOPS
——K=32 残废 + Gather 拆 QKV；我们 5 TFLOPS 流水线，其内核在我们形状
上反而 ~10ms，P5 关闭结论独立复核成立）；唯一可收割 = **DFA gather
向量化**（其 R3/R4：8ch/lane uint4 tap + block-coop staging + R8
sumfusion anchor 求和入 gather）。我们已经有的：32B 重复 entry、
ε-skip、plan 内联 softmax——欠缺的只是 gather 内存通道宽度。

### v11 设计（deploy/dfaplug_v11.cu）

- **EntryA 48B 每 anchor 去重**：`{int off[4]; uint4 wt8(8×half 组权
  重, <eps→0); uint4 cw4(4×half 角权+pad)}`——v8 是每 anchor×group
  重复 entry(32B)；wt=0 与剔除逐位等价（fmaf(+0,f,acc)=acc）；
- **gather = warp/（anchor,SPLIT-块）**，lane 持 8 通道（uint4 16B
  tap，warp 一次 512B 角点行），8|32 所以 lane 的通道必落在单组内；
  组权用 **8 路编译期 select 链**（动态下标 half 数组会让 ptxas 把
  uint4 降级 local memory：16B 栈帧、map gather 2× 慢——实测修掉后
  0 栈 40 reg）；
- **SPLIT 自适应**：nA≥512→2，否则 ceil(3072/nA) 夹 1..32（板上扫
  split 实测定：map A=100 在 32 平台期 0.30ms，det A=900 在 2 最优
  0.239ms）；split>1 → fp32 partial + 固定顺序 finalize（无原子，
  误差类与 v8 的 atomicAdd 到达序同类）；
- **权重 half 化**：~1e-3/项舍入（SparseDriveV2 v4/v5 同类过门）；
- workspace：counts[复用 v8 语义]+entries+partial，175.8MB→33.7-36MB
  （~5×）；引擎侧 getWorkspaceSize=max(v8,v11)，引擎按 v3（ws=0）编
  的场景下实例懒自分配，capture 前完成故 graph-replay 安全；
- enqueue 优先 v11（G==8 且 C%32==0 且 ws 足）→ v8 → v3 回落链不动。

### 模块级 A/B（dfa_eT6 12 调用真实 dump，test_dfa_v11）

- 数值：v11vseng l2 1.8-3.0e-4 / maxabs ≤7.8e-3（half 权重量化类，
  预测内；v8vseng ~1e-5），rc=0 全过；
- 时延：全帧 12 调用 **6.60→5.59ms（1.18×）**；det 1.14-1.45×，
  map 1.03-1.23×。

### 81 帧闭环门禁（bb2+hd+graph+libdfaplug_v11，`deploy/_v11_gate.py`）

| 指标 | v8 交付基线 | v11 实测 | 判定 |
|---|---|---|---|
| det mAP / NDS | 0.4160 / 0.4729 | **0.4157 / 0.4724** | Δ-0.0003 抖动类内 ✓ |
| map mAP | 0.7481 | **0.7462** | 抖动带 0.746-0.749 内 ✓ |
| e2e p50 | 39.86 ms | **39.11 ms** | -0.75 ms |
| hd 段 p50 | ~19.9 ms | **18.50 ms** | -1.4 ms（=DFA 收益落点） |
| 吞吐 | 5fps 源-paced | **5.06fps 81/81 全跑通** | 无停顿 ✓ |

产物：`libdfaplug_v11.so`（门禁数据对应 build md5
`f46f5bf710e9ee1c117c7e9592ca636d`，nvcc 时间戳致每次编译 md5 不同，
以功能为准）安装 /usr/local/lib 与 v8 并存（按路径显式 dlopen 互不
影响）；引擎零重编。**交付建议：两引擎链接插件换 v11 为默认**（门禁
过、严格更快、ws 5× 更省），v8 保留为回退。评估留档
`deploy/artifacts/eval_v11plug_mini.json`（det）/
`eval_v11plug_map.json`（map），闭环 dump `preproc_ref/v11g1` +
evaldata/v11plug。

### 本轮踩坑（全部已录 AGENTS.md 板端操作坑）

1. **pkill -f 自匹配杀 wrapper**：pkill 与后续命令同链时，载命令的
   bash 自身 cmdline 含 pattern → 自杀，后续 mkdir 全不执行 → node.log
   都不出现的"静默启动失败"。修：`pkill -9 -f 'sp_filesr[c]'`。
2. **gate 重跑旧 RC 假阳性**：CLEAN 没删 RC → 轮询第一跳读旧 rc 直接
   返回并 pkill 掉正在启动的新 run。修：整目录 `rm -rf BD && mkdir`。
3. **--hd 旗标漏拼 format 串（重犯 AGENTS.md 已记坑，第 2 次）**：
   gate 脚本定义了 HD 变量没拼进 nd → 静默 bb2 单引擎：infer 15ms、
   dump 只有 Reshape_9(46MB/帧!)、无 det_cls。修：拼进 format 串 +
   **launch 后日志指纹校验**（out_00/det_cls 存在性）。
4. **/opt/m0 盘满 100% 假死**：46MB/帧 dump 把 26G 盘写满 → dump 写
   停 7-99s（svc 尖刺而 gpu_ms 正常 = 文件 IO 特征）→ filesrc lease
   5s 强收槽 → FATAL 12 / exit 20。修：清 `*_out`（释放 4G）+ gate
   CLEAN 带 df 检查 + filesrc `SP_BUS_LEASE_MS=600000`（瞬态停顿不再
   破坏 seq 连续性；真死锁仍有 30×1s claim timeout→exit 20 兜底）。
5. GMSL 相机序列器内核日志风暴（~7.8 条/s，2026-10-05 起）——观测不
   致卡顿，判障时勿误判为根因。


## 第十九轮：M-PROD Phase D——权限收紧（2026-10-06 交付，mprodd 门禁过）

spec §7 ⑧ 的收敛版（用户批准口径：**只做文件权限收紧，维持 root 运行**，
不做用户隔离/最小权限改造——root 隔离需要重排 /var、/run、设备节点权限，
与"不动板子环境"纪律冲突，收益不抵风险）。与 Phase C1 遥测同批交付，
浸泡即跑最终形态（C1+D+v11 插件）。

### 改动清单（全部随 C1+D 一个 commit）

| 面 | 改动 | 交付验证（2026-10-06 板上实测） |
|---|---|---|
| 共享内存环 | sp_bus.cpp create 路径显式 `fchmod(fd, 0640)`（不依赖 umask） | `/dev/shm/sp_m3` = 0640 ✓ |
| 结果信箱 | sp_result.h create 路径同上 | `/dev/shm/sp_res_sp_result_m3` = 0640 ✓ |
| UDS 套接字 | sp_dmapool 路径 /tmp → **/run/sp**（tmpfiles 建目录 0750），socket chmod 0640 | /run/sp 0750 ✓（mapped-shm 模式下无 socket，--dma 迁移点用） |
| 服务默认掩码 | 双 unit 加 `UMask=0027` | `/proc/<nodepid>/status Umask: 0027` ✓；新文件（telemetry.jsonl/frame_log.tsv）0640 ✓ |
| 目录基线 | tmpfiles sp.conf：/run/sp、/var/log/sp、/var/lib/sp 全 0750 root:root | systemd-tmpfiles --create 实测 ✓ |
| 日志轮转 | sp-logrotate 50M×7 → **100M×5**（文件数换单文件预算） | /etc/logrotate.d/sp 在位 ✓ |
| 日志兜底 | filesrc+node 启动时 >100MB 截断（append 句柄 O_APPEND 下 truncate 安全；无 logrotate 的裸镜像防写满闪存） | 代码路径在位 |

**升级路径已知边界**（诚实记录）：fchmod/UMask 只管**新建**文件——
现场升级时已存在的旧环/旧日志保持旧 0644 权限，直到环被重建（--fresh/
删 /dev/shm）或日志被轮转。本日实测两 shm 对象都是新二进制重建的
（0640），门禁数据即最终形态。

### 附带修复：frame_log 表头/数据列错位（C1 引入，D 轮发现）

C1 给 frame_log.tsv 每行追了 3 列（gpu_temp_c/cpu_temp_c/sm_clock_mhz）
但漏改表头（11 列 vs 数据 14 列）——按表头解析的分析脚本会错位。修：
表头同点补齐 3 列名。教训入 AGENTS.md：**自描述文件改列必须同一提交
改表头**。

### 金标重标（v11）+ 升级顺序实锤

换 v11 插件后自检指纹拦的是 **FATAL code=15 直接退出**（金标过期=基础
设施错，不是 stay-alive 心跳）→ systemd Restart=on-failure 1s×5 →
**StartLimit 熔断**。现场顺序固化：install →（旧金标拦，预期）→
`_prod_golden.py gen --runs 10 --plugin libdfaplug_v11.so`（--selftest-
dump 从 manifest 直读帧 0，不碰环不消费，**无需活链/无需 override**，
gen 的 RING_OK 检查只是环境哨兵）→ `systemctl reset-failed` 双单元 →
start → SELFTEST PASS (9 tensors)。新金标容差（3×跨 run 最大偏差）：
det_bbox 1.383 / motion_cls 3.381 / plan_reg 0.053 / map_pts 0.381
（量级与 v8 版同 class）。_prod_golden.py 补上了 docstring 里承诺但
argparse 没定义的 `--plugin` 参数。

### mprodd 精度门禁（独立 81 帧闭环，插件定死 v8 以隔离 C/D 增量）

| 指标 | mprodd（C1+D） | mprodb | mprodc | m7fix 参考 |
|---|---|---|---|---|
| det mAP / NDS | **0.4169 / 0.4732** | — | — | 0.4177 / 0.4735 |
| map mAP | **0.7479** | 0.7481 | 0.7483 | 0.7485 |
| EPA car/ped | **0.6054 / 0.5000** | 0.6098/0.5089 | 0.6035/0.5028 | 0.6048/0.5023 |
| L2 | **0.7473** | 0.7481 | 0.7343 | 0.7377 |
| obj_box_col | **0.161%** | 0.242% | 0.161% | 0.161% |

全部落在既有跨 run 抖动带内（L2/col 单 run 散布 ±0.01/±0.08pp 口径），
C1+D 代码**零精度回归**。驱动脚本：`deploy/_pd_step1_build.py`（停单元
→build→install --no-start）、`_pd_step2_golden.py`/`_pd_step2b_resume.py`
（金标重标+熔断恢复）、`_pd_step3_verify.py`（D/C1 证据采集）、
`_pd_step3b_rebuild.py`（表头修复重编）、`_pd_step4_det.py`（mkeval+
det 目录组装+评估）；mprodd dump 留档 `work_dirs/preproc_ref/mprodd`
（det/map/mp eval 日志同目录）。

### 本轮踩坑（已录 AGENTS.md）

1. **chmod/fchmod 编译错 = 缺 `<sys/stat.h>`**：sp_dmapool.cpp 独缺
   （sp_bus/sp_result.h 都有），症状 `error: 'chmod' was not declared`。
2. **金标过期自检 = exit 15 → StartLimit 熔断**（见上，reset-failed
   双单元再 start）。
3. **mkeval 产 out_XX 而 det eval 吃 outv8_XX**：det 门禁用
   _pd_step4_det.py 现组 `mini_<tag>_det/outv8_XX` 别名目录（键
   det_cls/det_bbox/det_quality/det_instance_id + mini_meta.npz）。

## 第二十轮：M10 数据面硬化——跳序/latest-wins/分级 watchdog/wrap-lease（2026-10-07 交付，FT 4/4 + mprode 门禁过）

30fps 满负荷前置（spec §5/§8 M10 行）：常驻源枯竭/消费者停顿/晚接入在
30fps 下从"偶发"变"常态"，四个数据面缺口各补一刀。**零 shm 布局改动**
（RingMeta/SlotHdr 不动，kVersion=4 不变）；wrap-lease 纯行为、env 门控
默认关=与 v4 完全同行为。

### 改动清单

| 面 | 改动 | 语义 |
|---|---|---|
| sp_modelnode `--skip-lag` | acquire 到 seq>expect → 跳到最新（`s_skip_lag` 记账，frame_log 新列 `skip_lag`/`skip_wd`，16 列）；rewind/环重置仍走 resync 不计跳序；120 连续环重置 → 源枯竭 exit 20 | latest-wins：30fps 下消费者慢一拍不再 FATAL 12 连环重启；场景边界判定比的是已处理帧的 cur_scene，边界帧被跳过则下一处理帧照常触发 identity+dt=0.5 复位 |
| sp_watch 分级 abandon | dl 到点先发 `abandon_req`（tier1），主循环在**设防段返回后的边界检查点**弃帧（f_void[k]=1 哨兵，complete_frame 只推进 cursor：不产输出/不写信箱/不落 frame_log 行，信箱保持 last_valid、lage 增长=fail-visible）；请求挂起超过 dl+max(dl/2,500) 无人认领 → tier2 FATAL 13 `detail=watchdog-tier2`；SP_WD_MS 强制值仍是平退（逃生门不分级） | 真挂死与"一次卡顿"分开：返回型停顿弃帧续跑，不返回才杀进程。弃帧后 skip_wd 连续≥3/8 帧 → LATCH（kReasonStageFail），防持续饿死静默化 |
| sp_bus wrap-lease | claim-blocked 槽=发布者绕圈撞上长持有者；`SP_BUS_WRAP_LEASE_MS`（默认 0=关）下按第二租约强抢心跳陈旧持有者。默认 800ms 依据：盖过活读者 hb 老度峰值（2×svc+post 503ms 抖动 ≈690ms），远小于 5s 死租约 | 30fps 单消费者停 >5s 才会死租约强收——绕圈 4 槽 @30fps 下 800ms 就撞回，必须更短租约；release() 加 slot_idx<0 幂等护栏（TDD 意外收获，防强抢后消费者二次 release 解引用 slot(-1)） |
| sp_filesrc --wait-cons | 只等**第一个**消费者（原 = 等满 N 个才开拍）；`starting anyway (late joiners resync)` 日志行 | 晚接入消费者走 skip-lag/resync 自然对齐；生产不再因第二消费者缺席而无限等待 |

### FT 轮（deploy/_prod_ft_m10.py，板上独立进程 4/4 PASS）

| 用例 | 注入 | 判据 | 实测 |
|---|---|---|---|
| FT1 wrap-lease | fps30 --loop + sp_sub --hold-ms 60000 不心跳 | 20s seq≥300 且 forced≥1；对照活消费者 forced=0 | seq=600 forced=1 / forced=0 ✓ |
| FT2 skip-lag | SIGSTOP 6s 跨场景边界（k=40） | skip-lag rc0 + skipped≥20 + scene 复位行 + 无 FATAL；严格对照 rc12 | skipped=28 + `scene 0 -> 1 at frame 40` / rc12 ✓ |
| FT3 watchdog | STALL（2.5s 返回型）@pre 帧5 / HANG（不返回）@post 帧5 | stall：ABANDON+弃帧 rc0；hang：ABANDON 在前→rc13 tier2 | rc0 abandon=True / rc13 tier2=True ✓ |
| FT4 late join | --wait-cons 2 零消费者起拍 + 8s 后第二消费者 | 零消费者等待、starting-anyway、seq 前进、无 claim timeout | 4/4 断言 ✓ |

### mprode 精度门禁（两独立 run，新二进制 + --skip-lag，插件定死 v8）

| 指标 | mprodd 带 | mprode_a | mprode_b | 判定 |
|---|---|---|---|---|
| det mAP / NDS | 0.4169 / 0.4732 | 0.4173 / 0.4731 | 0.4176 / 0.4729 | ✓ |
| map mAP | 0.7479 | 0.7482 | 0.7476 | ✓ |
| EPA car/ped | 0.6054 / 0.5000 | 0.6069 / 0.5070 | 0.6046 / 0.4967 | ✓ |
| L2 | 0.7473 | 0.7369 | 0.7405 | ✓（±0.02 带内） |
| obj_box_col | 0.161% | 0.161% | 0.161% | ✓ |

5fps 节拍下两 run 零 SKIP/零 ABANDON 行（skip-lag 路径休眠等价，符合
预期）；e2e p50 48.11/48.23ms 带。单测：test_bus_lease 4/4、
test_watch_tier 4/4（板端）。install 收口断言：ExecStart 经
$SP_NODE_ARGS 生效 `--skip-lag`、env 含 SP_BUS_WRAP_LEASE_MS=800、
SELFTEST PASS (9 tensors)（金标 v11 工件指纹不含节点二进制 md5，M10
零工件改动故直接过）、信箱 seq 双探递进。dump 留档
`work_dirs/preproc_ref/mprode_{a,b}`（eval 日志同目录），eval driver
`deploy/_m10_eval_run.py <tag>`（det/map/mp 一键，mprodd 的
_pd_step4_det.py 泛化版）。

### Review Focus 五条的落地证据（plan 终审项）

1. wrap-lease 误抢 fence-in-flight 读者 → 默认 800ms > hb 老度峰值
   ~690ms 定量推导 + FT1 对照组 forced=0 实证；
2. skip-lag 跨场景边界 → FT2 实证边界帧被跳后复位行照出（判定基于
   已处理帧）；
3. 弃帧 parity 竞态 → 弃帧只发生在设防段返回后的边界点（GPU 已完成），
   单测 tier1 用例覆盖"卡住的调用返回后排干"形状；
4. watchdog 不得放过真挂死 → FT3b HANG 不返回必 rc13 + SP_WD_MS 逃生
   门平退不分级（单测 case3）；
5. unit/env 交互 → 零 unit 改动，SP_NODE_ARGS 通道透传（install 断言
   cmdline 实证）。

### 本轮踩坑（已录 AGENTS.md）

1. **driver launch 重定向串线**：`'> log'` 拼在 `; echo rc=$? > rcfile`
   链尾 → 重定向落到 echo 头上，echo 启动截断 node 日志且 rc 写进日志
   （rc=None 假象）——重定向只由 launch 包 `{ ...; }` 组做一次。
2. **pgrep -f 先命中 wrapper bash**：`{ node ...; }` 包装下 bash 命令行
   同含 node 字样，SIGSTOP 停 bash、子进程照跑（skipped=0 假阴性）——
   定点操作用 `pgrep -x`（comm 精确）并验 /proc state==T。
3. **返回型 STALL 注入必须落 tier 窗内**：睡 5s > dl(2s 地板)+grace(1s)
   = 真挂死输入，tier2 先杀，tier1 路径根本测不到——hook 改 2.5s。
4. **mini manifest scene id 从 0 起**（boston=0/queenstown=1），复位行
   `scene 0 -> 1`——断言以板上实际措辞为准，别凭记忆写编号。

### 终审修复轮（fresh-context 全分支 review → 1C+5I 全修，2026-10-07）

评审（1 Critical / 5 Important / 6 Minor）→ 修复与验证：

| # | 发现 | 修法 | 验证 |
|---|---|---|---|
| C1 | release 无代校验：wrap-lease 强抢+refill 换代后，陈旧 view 的 release 偷走新持有者 ref → 槽被提前放圈覆写（GPU 在读=fault 类） | release 加 `s.meta.seq == v->meta.seq` 代校验（claim 置 seq=0/commit 写新 seq，比对可靠），ref 与 held_slot 一并守卫 | test_bus_lease 新增 Case 5 交错用例：RED（claim 0ms 拿到槽）→ GREEN，5/5 |
| I1 | skip 分支不回滚 seq_base → 后续每帧重入 skip（计数按帧倍增、SKIP 行每帧刷；FT2 日志 11 行全是 skipped=28 实锤）；rewind 分支 `seq-1` 只对 submitted==0 正确 | 两分支统一 `seq_base = v.meta.seq - (submitted+1)` | FT2 强化断言 skip_lines==1 过（修复前 11 行） |
| I2 | 场景内跳序原始 dt（秒级）直接喂时序递归，违反离线链 dt>2.0 清状态契约 | reset 条件加 `dt>2.0‖dt<0` → 同边界口径复位（tmat=identity+dt=0.5），健康节拍（5fps=0.2s）不触发 | 代码路径审查 + mprodf 两 run 无误触发 |
| I3 | A 点弃帧时 prev_l2g/prev_ts/cur_scene 已折算 → 边界帧被弃丢复位、时序参考错一帧 | 折算前 save_*，A 点（hd 未提交、状态停 k-1）弃帧回滚折算；B/C 点（hd 已提交、状态到 k）保持折算但补发 mp 模板复位（mp 段被跳过，cur_scene 已折算不能再靠 k+1 清） | 逐检查点状态推进表审查（A/B/C/D × 状态/折算/mp 清零） |
| I4 | 环重置 x120 放弃路径 rc0，违反"常驻源枯竭必须 exit 20"契约（docs 也写错） | give-up 分支置 abort_run=true（--loop 下走 exit20） | 代码审查（:2173 分支实证） |
| I5 | wrap-lease 默认 800ms 对 hb 老度峰值 690ms 余量仅 110ms | 默认改 1000ms（尾量 310ms，仍 5× 低于 5s 死租约）；失败模式在 C1 修复后降级为撕裂帧 | m3.env 更新 + install 断言生效 |
| M1/M2/M4 | B 点注释失实（hd 实已提交）/abandon_ms 死字段/环重置 resync 白丢一帧 | 注释改写；删字段；`last_seq = ls-1` 立即消费现存最新 | 同批 |
| M3/M5/M6 | telemetry 无尺寸帽（既有 C1 行为）/env 每 claim 重读（开销可忽略）/m3.env v11 切换混入提交 | 接受并记录，不修 | — |

修复后门禁（mprodf 两独立 run，全部入 mprodd 带；节点日志零 SKIP）：

| 指标 | mprodd 带 | mprodf_a | mprodf_b |
|---|---|---|---|
| det mAP / NDS | 0.4169 / 0.4732 | 0.4164 / 0.4725 | 0.4174 / 0.4734 |
| map mAP | 0.7479 | 0.7484 | 0.7479 |
| EPA car/ped | 0.6054 / 0.5000 | 0.6028 / 0.5028 | 0.6020 / 0.4991 |
| L2 | 0.7473 | 0.7435 | 0.7453 |
| obj_box_col | 0.161% | 0.161% | 0.161% |

单测 bus 5/5 + tier 4/4；FT 全轮 4/4（FT2 skip_lines=1 新证）；install
收口断言复跑全绿（wrap=1000 生效、SELFTEST PASS、seq 双探递进）。
dump 留档 `work_dirs/preproc_ref/mprodf_{a,b}`。




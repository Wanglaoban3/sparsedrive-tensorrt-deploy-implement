# SparseDrive TensorRT 板端部署优化总结

## 交付版本

| 项目 | 值 |
|---|---|
| 引擎 | `models/e_T6.engine` (135 MB) |
| 插件 | `/usr/local/lib/libdfaplug_v8.so`（P3 DFA 双内核；内置 v3 回退） |
| 图源 | `v5_P1h.onnx` (v6_P1h 为 float 化实验版，不用于部署) |
| 导出脚本 | `deploy/export_v5.py` |
| 延迟 | **38.99 ms / 25.6 FPS** (100 iters mean; v3 插件 45.27 ms / 22.1 FPS) |
| mini mAP | **0.4203** (v3 插件 0.4184, +0.19pt; 持平) |
| mini NDS | **0.4738** (v3 插件 0.4723, +0.15pt; 持平) |
| 量化 | backbone/neck INT8 QDQ, head FP16, DFA 权重 FP16 |
| 原始基线 | 82.6 ms / 12.1 FPS / mAP 0.4178 → **2.12× 加速** |

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

## 下一步优化方向

### 1. 帧间流水线（吞吐 ~1.7×，不改引擎）
第 N 帧 backbone 和第 N-1 帧 decoder 在两个 CUDA stream 上重叠执行。
部署侧改动，不需要重新编译引擎。39ms 延迟不变但吞吐可到 ~44 FPS。

### 2. 全量 trainval val 验证（精度闭环）
在 6019 帧完整验证集上跑 e_T6 的 mAP/NDS。FP32 全量基线已有
(mAP 0.4135 / NDS 0.5226)。板端推理约 4-5 小时，管线已就绪。

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

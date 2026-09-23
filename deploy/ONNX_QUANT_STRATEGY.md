# SparseDrive 量化 ONNX 导出最优策略（sparsedrive-tensorrt-deploy-implement）

> 适用对象：本项目两个部署引擎中的 **感知引擎（det+map，`sparsedrive_multihead.onnx`）**
> 运行环境：Windows + RTX P100（无 nvcc/MSVC），ModelOpt 0.11 + torch 2.0.1+cu118

## 0. 结论速览

| 策略 | 动作 | 收益 |
| --- | --- | --- |
| 拆分双引擎（维持本仓库设计） | 感知(det+map)与规划(motion+plan)分开导出、分别量化验收 | 调度灵活、显存/workdir 独立、量化误差不跨引擎放大（§8） |
| 时序变换进模型图 | `instance_t_matrix`/`time_interval` 作为图输入，anchor 投影=图内 MatMul/仿射 | 零 D2H 同步、不新增 plugin、数值与 QAT 一致（§9） |
| 最大化量化覆盖 | MTQ 量化全部 Conv/Linear + QKV 模块化手术（`deploy/flashmha_qkv.py`） | 112 → 133 个量化模块（335 → 398 量化器），QKV 进入 W8A8（§10） |
| 敏感层回退 | 按 `sparsedrive_ptq_sensitivity.json` 的 ranked 列表保留 top-K 敏感层 FP | 用极小的 FP 代价换取主要漂移 |
| 层折叠 | Conv+BN 预折叠 → 常量折叠 → QDQ 权重折叠（int8 initializer） | 时序图 135 / 首帧图 134 条折叠链，ORT 重放逐位等价 |
| DFA 边界保护 | `SparseDrive::DeformableAggregation` 自定义节点输入保持 FP | TRT plugin 无需处理 Q/DQ 输入；loc 保持 FP 保几何精度（§10.3） |
| 结构标准化 | 导出后跑 `deploy/qdq_onnx_rewrite.py`（SparseDriveV2 同款） | 标准 ORT 风格 QDQ，板端 int8 rewriter 可直接识别 |

## 1. 流水线

```
ckpt/sparsedrive_stage2.pth
  │  (可选) tools/fuse_conv_bn.py 逻辑: Conv+BN 预折叠
  ▼
deploy/ptq_sensitivity.py   ──► deploy/artifacts/sparsedrive_ptq_sensitivity.json
  │  MTQ INT8(max校准) + leave-one-out 敏感度 + skip-topK 曲线
  ▼
deploy/qat.py               ──► ckpt/sparsedrive_stage2_qat.pth
  │  MTQ INT8(带skip-list) + mini 真实任务损失短程微调
  ▼
deploy/export_quant_onnx.py ──► work_dirs/sparsedrive_small_stage2/sparsedrive_int8{,_first}.onnx
  │  opset13, 静态shape, fake-quant → Q/DQ, do_constant_folding=True + onnxsim
  ▼
deploy/qdq_onnx_rewrite.py  ──► *_rewritten.onnx (标准 ORT 风格 QDQ)
  │  权重Q折叠为int8 initializer; Constant scale/zp → 共享 initializer; 拓扑排序
  ▼
(用户侧 Linux + TRT 8.5) onnx2trt / trtexec
```

## 2. 量化范围（最大化覆盖）

- MTQ `INT8_DEFAULT_CFG`（W8A8，max 校准）作用于 wrapper 全部 Conv/Linear：
  ResNet50 主干 + FPN + det/map 双头（decoder FFN、refine、DFA 的 weights_fc/output_proj、
  anchor/instance encoder、depth branch）。
- **QKV 模块化手术**（§10.2）：本仓库 `FlashMHA` 的 QKV 是打包裸参数
  `in_proj_weight [3E,E]` + `F.linear` 函数调用，MTQ 包不到；`pack_to_linears()`
  把它拆成 q/k/v 三个 `nn.Linear`（数值逐位一致），MTQ 即可同规格覆盖。
  感知引擎 21 个注意力 → 量化模块 112 → 133，量化器 335 → 398。
  `deploy/qat.py` 与 `deploy/export_quant_onnx.py` 已内置该手术，顺序见 §10.2。
- 每个 Conv/Linear 模块带 weight + input + output 三个量化器
  （`QuantMaxPool2d` 只有 input/output）——skip 控制必须三件套一起关，
  `ptq_sensitivity.set_module_quant()` 已如此实现。
- 首帧/时序双模型共用同一量化模型（fake-quant 与部署图一致）。
- 校准数据：nuScenes-mini 真实图像（训练管线增强流，首帧历史输入置零），
  与导出契约完全一致。

## 3. 已知量化盲区与对策

| 盲区 | 原因 | 对策 |
| --- | --- | --- |
| ~~QKV 投影（FlashMHA in_proj 裸参数）~~ | 打包 Parameter + `F.linear` 函数路径，MTQ 无法替换 | **已解决**：`deploy/flashmha_qkv.py` 模块化手术（§10.2） |
| DFA 自定义算子输入 | plugin 期望 FP I/O | 保持边界 FP；`weights` 的生产者 weights_fc 已量化，feat 上游 INT8（§10.3） |
| LayerNorm / softmax | 对激活不友好 | 按默认配置保持 FP（TRT 侧自动融合） |
| attention 内部 matmul×softmax | 纯函数计算，无参数可量化 | softmax 保持 FP；QKV/out_proj 两端 Linear 已 W8A8，TRT 可对中间 QK^T/PV matmul 自行选 FP16/INT8 |
| `MotionPlanONNXWrapper`（规划引擎） | 本阶段只验收感知引擎 | 规划引擎复用同一套手术+导出+重写流程即可（§8） |

## 4. 敏感层回退（skip-list）

- `deploy/ptq_sensitivity.py` 输出：
  - 全 INT8 漂移（rel_l2 / cos，见 JSON `full_int8`）；
  - 每模块 leave-one-out 漂移差 `sensitivity`（越大越敏感）；
  - `skip_curve`：保留 top-K 敏感层 FP 后的整体漂移，用于选 K
    （自动选则：取 rel_l2 ≤ 50% 全 INT8 漂移的最小 K；无点达标时取曲线
    argmin 兜底，`qat.apply_skip_from_report` 已实现该策略）。
- `deploy/qat.py --skip-json ...` 自动按该列表对这些层关闭
  weight/input/output 三个量化器（三件套必须一起关，漏掉 output_quantizer
  会让"全 FP"测量变成假的）。

### 4.1 组级任务指标敏感度（mAP/NDS 口径，最终选层依据）

逐模块 rel_l2 排名是长尾、头部在噪声带内，skip-topK 无任务收益
（81 帧配对哨兵：skip14≈skip0）。因此改用**功能组留组法**：
每组保持 FP（其余全量化），在固定 81 帧 mini-val 上配对测 mAP/NDS
（`deploy/group_sensitivity.py`，每组独立重建+校准；ModelOpt 校准在
calib 模式下前向恒等、只记录 amax，故 amax 一致、组间可比）。

| 组（保FP模块数） | mAP | ΔmAP | NDS | ΔNDS |
|---|---|---|---|---|
| fp32 | — | 0.4255 | — | 0.4805 |
| 全 INT8（无保护） | 0 | 0.4169 | −0.0086 | 0.4717 | −0.0088 |
| det_head_output（cls/quality，12） | 0.4249 | **−0.0006** | 0.4786 | −0.0019 |
| det_head_ffn（6） | 0.4222 | −0.0033 | 0.4731 | −0.0074 |
| map_head_attn（44） | 0.4209 | −0.0046 | 0.4747 | −0.0058 |
| backbone_deep（layer3+4，29） | 0.4204 | −0.0050 | 0.4756 | −0.0049 |
| backbone_shallow（12） | 0.4201 | −0.0054 | 0.4772 | −0.0033 |
| depth_branch（3） | 0.4189 | −0.0066 | 0.4716 | −0.0089 |
| det_head_dfa（6） | 0.4163 | −0.0092 | 0.4730 | −0.0075 |
| backbone_mid（13） | 0.4156 | −0.0098 | 0.4718 | −0.0087 |
| det_head_enc（2） | 0.4130 | −0.0125 | 0.4714 | −0.0091 |
| det_head_attn（QKV+proj，40） | 0.4063 | −0.0191 | 0.4647 | −0.0158 |
| img_neck（FPN，8） | 0.4271 | +0.0016 | 0.4823 | +0.0018 |
| 全 FP 控制组 | all | 0.4255 | +0.0000 | 0.4805 | +0.0000 |

要点（噪声底约 ±0.005，来自 map_head_attn/img_neck 等与 det 路径无关
组的波动和校准集噪声）：
- 全 INT8 的任务损伤本身就小（−0.009 mAP / −0.009 NDS）；
- **det_head_output（cls/quality 输出层，仅 12 个模块）保 FP 几乎完全
  恢复损伤**——任务口径下性价比最高的保护组，与 rel_l2 排名
  （全是 backbone）完全不同；
- backbone_deep 次之；FPN 量化无代价；det_head_attn 保 FP 反而更差
  （误差补偿效应，勿保护）；
- 任务口径的推荐配置：保 FP = det_head_output + backbone_deep
  （41 模块，其余 134 个量化模块含全部 QKV/DFA 生产者保持 INT8），
  预期 mAP 与 fp32 差异在噪声带内。

### 4.2 测量陷阱（重要教训，已修复）

- **mtq.quantize() 会把整个模型翻成 train() 模式**：默认的校准循环在
  train 模式下跑，BatchNorm 会用 16 个 mini-batch 的批统计覆盖
  running_mean/running_var，留下 ~0.2 rel_l2 的、与量化无关的永久漂移，
  且 leave-one-out 全部失真（v1/v2 报告因此作废）。
- 修复：校准循环内 `model.eval()` + 校准后按键恢复 BN 缓冲区
  （`ptq_sensitivity.py` / `qat.py` 均已内置）。
- 修复前后对照：全 INT8 drift rel_l2 0.4283(污染) → **0.2563(干净)**，
  cos 0.8343 → 0.8822。

## 5. 层折叠与图清理

1. **Conv+BN 预折叠**（训练后一次性）：`tools/fuse_conv_bn.py`（注意把 `mmdet3d.apis.init_model`
   换成 mmdet 的 build_detector+load_checkpoint，见 deploy/ 脚本内实现）。P100 上折叠与
   量化联合做更准（BN 的 running_var 直接参与 scale 计算）。
2. **常量折叠**：导出 `do_constant_folding=True` + onnxsim。
3. **QDQ 权重折叠**：`deploy/qdq_onnx_rewrite.py`——每个 site 的权重 Q 节点折叠成
   int8 initializer → DQ；所有 Constant 携带的 scale/zp 移为共享 initializer；
   SparseDriveV2 上验证过 94 条链逐位等价。
4. **DFA 边界检查**（导出后）：遍历 `SparseDrive::DeformableAggregation` 节点的 5 个输入，
   断言没有 Q/DQ 插在其数据通路上（spatial_shape/scale_start_index 为 int 常量，
   mc_ms_feat/sampling_location/weights 为 FP）。

## 6. 数值验收（本机实测）

### 6.1 正式指标：完整 val 集（nuScenes trainval val，6019 帧，官方 devkit）

| 模型 | mAP | NDS | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ |
|---|---|---|---|---|---|---|---|
| fp32（stage2 权重） | 0.4135 | 0.5226 | 0.5638 | 0.2768 | 0.5404 | 0.2697 | 0.1905 |
| INT8 QAT（skip-28 保护组） | 0.4081 | 0.5200 | 0.5692 | 0.2767 | 0.5313 | 0.2737 | 0.1901 |
| **Δ（量化代价）** | **−0.0053** | **−0.0026** | +0.0054 | −0.0001 | −0.0091 | +0.0040 | −0.0004 |

- INT8 量化代价：**−0.53 mAP 点 / −0.26 NDS 点**（146 个可量化模块中
  28 个敏感模块保 FP + QAT 微调）；mAOE 反而略优（噪声带内）。
- fp32 基线与 SparseDrive 论文数字（mAP≈41.9/NDS≈52.5）吻合，验证
  本机评测链路（无 CAN bus 的 ego status 重建、无矢量地图扩展、
  纯 torch DFA）正确。
- 复现：`deploy/eval_nuscenes.py --mode fp32/quant --version v1.0-trainval
  --ann-file data/infos/nuscenes_infos_val.pkl --data-root <trainval 根>`；
  结果 JSON 在 `deploy/artifacts/eval_fp32_val.json` /
  `eval_int8qat_val.json`。

### 6.2 torch 侧输出漂移（mini，16 校准 / 8 评估样本）
  - 全 INT8（无 skip）：rel_l2=0.2563，cos=0.8822；
  - QAT 起点（skip-k=28，即 146 个可量化模块中 28 个敏感层保 FP）：
    rel_l2=0.2561，cos=0.8840；
  - QAT（漂移门控，best@iter20，lr=1e-6）：rel_l2=0.2555，cos=0.8859
    ——门控保证不劣于 PTQ；100 iter 的增益有限（误差主要来自激活量化的
    长尾传播，不是权重噪声），如需进一步收敛可加大 skip-K 或延长微调。
  - skip-top112（全 FP）rel_l2=0.00000 / cos=1.00000，测量链路自校验通过。
- ONNX 侧（`*_rewrite_report.json`）：
  - 时序图：6687→6552 节点，Q 268→133、DQ 268 保持，135 个权重 Q 折叠为
    int8 initializer，135 条 Q→DQ 折叠链 ORT 重放**逐位相等**；
  - 首帧图：6310→6176 节点，134 条链逐位相等；
  - DFA 边界：两图各 12 个 `DeformableAggregation` 节点，所有输入边均为
    FP 收尾（Q→DQ 对），无需修补（`dfa boundary: int8_edges_fixed=0`）。
- 最终精度以用户 TRT 侧对齐验证为准（本机不做引擎编译）。
- **任务指标哨兵**（`deploy/task_sentinel.py`，mini-val 81 帧固定子集、
  同帧配对比较；绝对值受子集噪声影响，配对差值是有效信号）：

  | 变体 | mAP | ΔmAP | NDS | ΔNDS |
  |---|---|---|---|---|
  | fp32 | 0.4255 | — | 0.4805 | — |
  | PTQ 全 INT8 (skip0) | 0.4151 | −0.0103 | 0.4733 | −0.0072 |
  | PTQ skip14 | 0.4151 | −0.0104 | 0.4728 | −0.0077 |
  | PTQ skip28 | 0.4178 | −0.0077 | 0.4773 | −0.0032 |
  | PTQ skip56 | 0.4303 | +0.0048 | 0.4836 | +0.0031 |
  | QAT skip28（交付） | 0.4197 | −0.0058 | 0.4777 | −0.0028 |

  结论：无灾难性崩塌；skip 越多损伤越小；QAT 优于同 K 的 PTQ；
  skip56 略超 fp32（在 81 帧噪声带内）。正式数字以完整 val 集
  （6019 帧，`deploy/eval_nuscenes.py --version v1.0-trainval`）为准。

## 7. 复现命令（本项目根目录，sparsedrive_deploy 环境）

```bash
# 1. 敏感层分析（~1h @P100）
python deploy/ptq_sensitivity.py --calib-samples 16 --eval-samples 16
python deploy/render_ptq_report.py          # 生成 Markdown 报告

# 2. QAT（漂移门控：每 iters/10 评估一次漂移，只保存最优状态，
#    微调无收益时自动回滚到 PTQ 状态，保证不劣化）
python deploy/qat.py --iters 100 --lr 1e-6 \
    --skip-json deploy/artifacts/sparsedrive_ptq_sensitivity.json --skip-k 28

# 3. 量化 ONNX 导出（进程内重建量化结构 + 复原 QAT state_dict，
#    含 amax；ModelOpt Quant* 动态类不可整体 pickle）
python deploy/export_quant_onnx.py --checkpoint ckpt/sparsedrive_stage2_qat.pth \
    --skip-json deploy/artifacts/sparsedrive_ptq_sensitivity.json --skip-k 28 \
    --out work_dirs/sparsedrive_small_stage2/sparsedrive_int8.onnx

# 4. QDQ 结构标准化 + DFA 边界归一（报告 *_rewrite_report.json）
python deploy/qdq_onnx_rewrite.py work_dirs/sparsedrive_small_stage2/sparsedrive_int8.onnx
```

## 8. 整体导出 vs 感知/规划拆分 —— 结论：维持拆分

本仓库已经是"感知引擎（det+map）+ 规划引擎（motion+plan）"的拆分设计，
建议保持，理由：

1. **算力结构**：感知引擎占绝大部分算力（6 视角 ResNet50+FPN），规划引擎只有
   轻量 decoder（几百 token 的注意力/FFN）。拆开后规划可低频运行或复用上一帧
   检测结果，两段独立 build/反序列化/复用 workspace，显存峰值不叠加——
   对 16GB 级 GPU 和车端都友好。
2. **量化精度**：两段分别校准/QAT/验收，误差不跨引擎级联。整体图量化时
   backbone 的漂移会污染 planning 分支的校准统计（amax 偏移），skip-list
   也无法按引擎分开调。
3. **桥接口径已经稳定**：det→motion 的桥（instance_feature / anchor_embed /
   cls 分数 / ego queue / `T_temp2cur`）在 `MotionPlanONNXWrapper.forward`
   中已是显式张量契约，规划头只需 `forward_onnx` 纯函数化。整体化的唯一收益
   是省一次 engine 间的显存指针传递（同一 GPU 上接近零拷贝），不构成理由。
4. **动态 shape 风险**：整体化后 det 的 top-k/排序/ID 匹配结果直接进规划，
   有效实例数是动态的；拆分时规划引擎按固定 900×Q 队列消费（置零填充），
   图保持静态 shape，TRT 才能吃到固定 profile。

如果调度上确实想要"一次 enqueue"，可以在 TRT 层把两个 engine 串进同一个
CUDA graph，而不必合并 ONNX。

## 9. 预测框的时序变换：进模型图 vs CUDA 算子 —— 结论：进模型图

时序变换指：把上一帧输出的 anchor 状态（位置/速度/朝向）按
`T_temp2cur = T_global_inv(cur) @ T_global(prev)` 投影到当前帧（含速度×Δt 外推）。

| 方案 | 评估 |
| --- | --- |
| **图内计算（本方案，已实现）** | `instance_t_matrix [B,4,4]`、`time_interval [B]` 作为 ONNX 输入；投影 = Slice + MatMul + Add 仿射。TRT 把这些小算子与相邻层融合，开销可忽略；无任何自定义代码；与 QAT fake-quant 图完全一致；batch 显式 |
| 宿主侧 CPU | 900×11 的仿射本身 ~µs 级，但需要 anchors D2H + 同步 + 回传 H2D，引入毫秒级延迟并打断异步流水线——不可取 |
| TRT plugin / 自定义 CUDA 算子 | 性能无增益（图内 MatMul 已融合），却多一个要编译维护的 plugin；本项目 DFA plugin 是唯一的自定义算子，应保持唯一 |

注意事项：位姿矩阵路径**不做量化**（4×4 pose 的 INT8 无意义且会传染给 DFA
采样坐标）；`time_interval` 必须是张量输入而非 Python 常量，否则每换 Δt 都要
重新导出/编译；`|Δt| > max_time_interval` 的失效分支在部署图里由首帧/时序
双 ONNX 承担（宿主端选图），避免图内动态控制流。

## 10. 量化覆盖面扩展（QKV / proj / DFA 输入）

### 10.1 覆盖现状

| 计算类别 | 数量/位置 | 状态 |
| --- | --- | --- |
| Backbone/Neck Conv | ResNet50 + FPN 全部 | W8A8（weight+input+output） |
| FFN / refine / encoder Linear | det/map decoder 各层 | W8A8 |
| attention `out_proj` | 21 处 FlashMHA | W8A8 |
| **QKV 投影** | 21 处 FlashMHA `in_proj` | **手术前是盲区 → 手术后 W8A8**（§10.2） |
| DFA `weights_fc`（采样权重生产者） | 每层 DFA 前 | W8A8（其输出= DFA weights，amax 已校准） |
| DFA `kps_generator`（采样坐标生产者） | 每层 DFA 前 | 线性尾接仿射几何变换，坐标刻意保持 FP（§10.3） |
| LayerNorm / softmax / GridMask / 位置编码 | — | FP（标准做法） |

### 10.2 QKV 模块化手术

`FlashMHA`（`projects/mmdet3d_plugin/models/attention.py`）的 QKV 是
`in_proj_weight [3E,E]` 裸 Parameter + `_in_projection_packed()` 函数调用，
MTQ 按模块替换量化器，因此包不到它。`deploy/flashmha_qkv.py`：

```python
pack_to_linears(model)   # in_proj_weight/bias 按 E 切三份 → q/k/v nn.Linear
                         # 并把 FlashMHA.forward 换成走三个 Linear 的等价实现
```

- 数值逐位一致（同一权重切片、同一 matmul 语义），自测见
  `deploy/test_flashmha_qkv.py`；
- **调用顺序**：QAT 路径 = build → load fp32 ckpt → surgery → MTQ；
  导出路径 = build → surgery → load QAT ckpt（键名才能对上）→ 导出；
- 感知引擎 21 个注意力 × 3 = 63 个新量化器，总模块 112 → 133，
  总量化器 335 → 398；

### 10.3 DFA 三输入（loc / feat / weights）的量化取舍

DFA 自定义节点 5 个输入：`mc_ms_feat`（特征）、`spatial_shape`/`scale_start_index`
（int 常量）、`sampling_location`（采样坐标）、`weights`（采样权重）。

- **weights**：由 `weights_fc` Linear（W8A8）产生，输出侧 amax 校准完成——
  "权重量化"已通过生产者覆盖；
- **feat**：FPN 输出以 FP32 进 plugin，但其上游 backbone/neck 全部 INT8——
  重计算已经是 int8 完成，边界 FP 只是反量化后的表达，不损失已获得的收益；
- **loc**：刻意保持 FP。采样坐标经 lidar2img 投影后是亚像素量，CUDA kernel
  依赖双线性插值精度（loc∈[0,1) 的严格边界排除 + pixel-center 对齐），
  INT8 坐标会直接转化为采样错位；且坐标链路是几何仿射而非可学习线性层，
  本就不在 MTQ 覆盖内。
- 若未来要在 plugin 内做 int8 gather/bilinear（真正的 DFA int8 kernel），
  需要重写 CUDA 算子（本项目范围外，P100 无法编译验证）；届时把边界上的
  Q→DQ 对改为只保留 Q（plugin 直收 int8）即可，本次导出的图已经把每个
  DFA 输入边都归一到"浮点收尾"（Q→DQ 对或原始 FP），检查报告见
  `*_rewrite_report.json` 的 `dfa boundary` 字段。

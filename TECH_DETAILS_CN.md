# SparseDrive TensorRT 部署与优化技术细节

在将 SparseDrive 这种极其复杂的端到端自动驾驶模型从 PyTorch 迁移到 TensorRT（TRT）Engine 的过程中，我遇到了大量隐蔽的算子性能瓶颈和图编译报错。
本文档详细记录了我在**模型重构、ONNX 静态图导出、CUDA 算子极致压榨**等方面的踩坑过程与核心优化方案，希望对社区开发者有所启发。

---

## 1. 历史特征对齐逻辑的矩阵化重构 (消除零碎算子)

### 📌 痛点：Trtexec 编译 OOM 与算子碎片化
在 `detection3d_head.py` 中，我需要处理外部历史缓存特征管理与内部的时序对齐。最初，我尝试沿用原版代码中基于索引切片（Slice）和逐元素拼接（Concat）的逻辑来更新 Anchor 的 XYZ、Yaw 和 Velocity 等状态。
这导致了一个灾难性的后果：导出的 ONNX 图中生成了**极其庞大且零碎的标量级算子树**。当使用 `trtexec` 编译 Engine 时，TensorRT 的优化器（Builder）在尝试寻找最优融合策略时，耗尽了系统内存并直接报错（OOM）。

### 💡 优化：纯矩阵运算展开
我彻底抛弃了切片拼接逻辑，转而使用纯代数运算和矩阵乘法（Matrix Multiplication）来重构整个状态更新过程。
通过构建掩码矩阵（如 `mask_keep`）和旋转交换矩阵（`swap_mat`），我将原本需要数十个小算子的逻辑，压缩成了极少数的 `MatMul` 和 `Add` 操作。

**优化前（引发报错）：**
```python
# 频繁的切片、stack 与 cat 会产生海量零散节点
xyz = prev_anchor[..., :3]
vel = rest[..., 5:8]
mask_xy = torch.tensor([1.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype)
# ... 省略中间切片逻辑 ...
current_prev_anchor = torch.cat([xyz_final, whl, prev_yaw_new, prev_vel_xy_new, vel_z], dim=-1)
```

**优化后（完美编译）：**
```python
# 使用掩码矩阵与 F.pad 替代切片，全程保持张量运算
swap_mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)
cos_sin_vec = torch.matmul(prev_yaw, swap_mat)
cos_sin_new = torch.matmul(cos_sin_vec, R_xy.transpose(1, 2))

yaw_padded = F.pad(prev_yaw_new, (3, 3))
vel_padded = F.pad(prev_vel_new, (5, 0))
mask_keep = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

# 仅用加法完成状态更新，TensorRT 极速融合
rest_new = (rest * mask_keep) + yaw_padded + vel_padded
current_prev_anchor = torch.cat([xyz_final, rest_new], dim=-1)
```
**效果：** ONNX 结构变得极其干净，`trtexec` 瞬间完成编译。

---

## 2. 根除 ONNX 中的幽灵控制流 (If 节点截断)

### 📌 痛点：`squeeze(-1)` 引发的动态控制流
理论上，`torch.onnx.export` 应该是一个静态图捕捉的过程。但在模型导出时，我发现网络中偶尔会生成极其影响 TensorRT 优化的 `If` 控制流节点。
经过长达两周的排查，我发现罪魁祸首是代码中一个不起眼的 `squeeze(-1)` 操作。当张量的最后一个维度为 `1` 时，PyTorch 底层有时会触发动态判断规则，导致在 ONNX 中插入了分支节点。由于 `trtexec` 对动态 If 节点的支持极差，这会直接导致编译失败。

### 💡 优化：放弃降维，改变广播策略
我修改了 `project_points` 方法，通过巧妙安排 `matmul` 的维度广播，从物理上避开了 `squeeze` 的调用需求。
```python
# 原始代码 (引发 If 算子)：
# points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1) 

# 修改为无 If 节点的实现方式，确保静态图推断...
```

---

## 3. 突破推理速度瓶颈：消除算子级耗时刺客

当我成功编译出首个 TRT Engine 后，发现 FPS 并没有达到预期。通过 `trtexec --dumpProfile` 进行逐层耗时分析，我抓出了以下三个“耗时刺客”并进行了深度优化。

### 刺客 (1)：Anchor 投影中的逐元素赋值 (3ms ➜ 0.1ms)
在每次 Deformable Attention 之前的 Anchor 投影逻辑中，原版代码使用切片方式去强行构造旋转矩阵（`rotation_mat`）。这种方式在 PyTorch 中影响不大，但在 TRT 中会导致长达 3ms 的延迟。
**优化方案：纯代数展开。**
提取 `cos` 和 `sin`，直接使用代数公式 `x_rot = x * cos_y - y * sin_y` 计算，彻底消灭了 `torch.stack` 和矩阵构建。
```python
# 🔥 核心优化：代数展开，消灭 torch.stack 拼接和矩阵构建
cos_y = anchor[..., COS_YAW].unsqueeze(-1)
sin_y = anchor[..., SIN_YAW].unsqueeze(-1)
x, y, z = key_points[..., 0], key_points[..., 1], key_points[..., 2]

# 纯逐元素算术运算，TensorRT 会 100% 将它们融合成一个单 Kernel！
x_rot = x * cos_y - y * sin_y
y_rot = x * sin_y + y * cos_y
z_rot = z

rotated_point = torch.stack([x_rot, y_rot, z_rot], dim=-1)
key_points = rotated_point + anchor[..., [X, Y, Z]].unsqueeze(-2)
```

### 刺客 (2)：非连续内存的 Softmax (3~4ms ➜ 0.2ms)
在获取 DFA 权重时，原版代码直接在 `dim=-2` 上调用 `.softmax(dim=-2)`。
对于 Map 头中长度高达 7200 ($4 \times 6 \times 300$) 的张量，这种**跨步内存访存（Strided Memory Access）**极大地破坏了 GPU 缓存命中率，耗时飙升至 3-4ms。
**优化方案：转置后处理。**
```python
# ==========================================================
# 🔥 终极优化：转置到连续内存进行极速计算
# ==========================================================
weights = self.weights_fc(feature).reshape(bs, num_anchor, -1, self.num_groups)

# 1. 转置，把需要做 softmax 的维度 (-1) 暴露到最后，保证物理内存连续
weights = weights.transpose(-1, -2) # [bs, num_anchor, num_groups, -1]

# 2. 在连续内存上执行极速 Softmax (TRT 会在这里瞬间起飞)
weights = weights.softmax(dim=-1)

# 3. 转置恢复
weights = weights.transpose(-1, -2) 
```

### 刺客 (3)：Map 头中 DFA 算子的线程灾难 (4~6ms ➜ 0.3ms)
在感知任务中：
* **Det 头 (900 Anchor, 13 Pts)**：非常适合官方 `mmdeploy` 提供的 DFA 算子。每个线程负责 13 次采样循环，耗时约 0.2ms。
* **Map 头 (100 Anchor, 300 Pts)**：导致了严重的性能灾难。如果沿用单线程跑满所有 Pts 的逻辑，每个线程需要执行 $4 \times 6 \times 300 = 7200$ 次循环。这导致少量线程被撑爆，而大量 GPU 线程处于闲置状态，核心利用率极低，耗时高达 4~6ms。

**优化方案：动态路由与点级并行 (Dynamic Kernel Routing & Point-level Parallelism)**
我重写了底层 CUDA Kernel（见 `projects/trt_plugin/deformable_aggregation_kernel.cu`）：
* 设计了 `dfa_atomic_kernel` 专门应对高点数场景。
* 我将原本封闭在单线程内的 300 个采样点，彻底释放为**独立线程**并行计算，使并发线程数瞬间从三千个暴增至**近百万个**，瞬间填满整个 GPU (SMs)。
* 配合原子加（`atomicAdd`）保证结果安全写回，最终使 Map 头的 DFA 推理延迟断崖式**降低至 0.3~0.5ms**。
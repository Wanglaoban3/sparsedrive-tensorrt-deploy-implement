# SparseDrive TensorRT Deployment and Optimization Technical Details

During the process of migrating an extremely complex end-to-end autonomous driving model like SparseDrive from PyTorch to a TensorRT (TRT) Engine, I encountered numerous hidden operator performance bottlenecks and graph compilation errors.
This document details my troubleshooting journey and core optimization solutions in areas such as **model refactoring, ONNX static graph export, and extreme CUDA operator squeezing**, hoping to inspire community developers.

---

## 1. Matrix-based Refactoring of Historical Feature Alignment Logic (Eliminating Fragmented Operators)

### 📌 Pain Point: Trtexec Compilation OOM and Operator Fragmentation
In `detection3d_head.py`, I needed to handle external historical cache feature management and internal temporal alignment. Initially, I tried to follow the original code's logic based on index slicing (`Slice`) and element-wise concatenation (`Concat`) to update the XYZ, Yaw, and Velocity states of the Anchors.
This led to a disastrous consequence: the exported ONNX graph generated an **extremely massive and fragmented scalar-level operator tree**. When compiling the Engine using `trtexec`, the TensorRT Builder exhausted system memory and threw an Out-Of-Memory (OOM) error while trying to find the optimal fusion strategy.

### 💡 Optimization: Pure Matrix Operation Unrolling
I completely abandoned the slice-and-concat logic, turning to pure algebraic operations and Matrix Multiplication to reconstruct the entire state update process.
By constructing mask matrices (like `mask_keep`) and rotation swap matrices (`swap_mat`), I compressed the logic that originally required dozens of small operators into a very small number of `MatMul` and `Add` operations.

**Before Optimization (Causes Error):**
```python
# Frequent slicing, stack, and cat operations generate massive fragmented nodes
xyz = prev_anchor[..., :3]
vel = rest[..., 5:8]
mask_xy = torch.tensor([1.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype)
# ... intermediate slice logic omitted ...
current_prev_anchor = torch.cat([xyz_final, whl, prev_yaw_new, prev_vel_xy_new, vel_z], dim=-1)
```

**After Optimization (Perfect Compilation):**
```python
# Use mask matrices and F.pad instead of slicing, keeping tensor operations throughout
swap_mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)
cos_sin_vec = torch.matmul(prev_yaw, swap_mat)
cos_sin_new = torch.matmul(cos_sin_vec, R_xy.transpose(1, 2))

yaw_padded = F.pad(prev_yaw_new, (3, 3))
vel_padded = F.pad(prev_vel_new, (5, 0))
mask_keep = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

# Complete state updates using only addition, allowing blazing-fast TensorRT fusion
rest_new = (rest * mask_keep) + yaw_padded + vel_padded
current_prev_anchor = torch.cat([xyz_final, rest_new], dim=-1)
```
**Result:** The ONNX structure becomes extremely clean, and `trtexec` finishes compilation instantly.

---

## 2. Eradicating Ghost Control Flows in ONNX (If-Node Pruning)

### 📌 Pain Point: Dynamic Control Flow Triggered by `squeeze(-1)`
Theoretically, `torch.onnx.export` should be a static graph tracing process. However, during model export, I found that the network occasionally generated `If` control flow nodes, which severely impacted TensorRT optimization.
After two weeks of debugging, I found the culprit was an inconspicuous `squeeze(-1)` operation in the code. When the last dimension of the tensor is `1`, PyTorch's backend sometimes triggers dynamic evaluation rules, inserting branch nodes into ONNX. Since `trtexec` has extremely poor support for dynamic `If` nodes, this directly led to compilation failures or plummeted runtime efficiency.

### 💡 Optimization: Abandoning Dimensionality Reduction and Changing Broadcast Strategy
I modified the `project_points` method to physically bypass the need for `squeeze` by cleverly arranging the dimensional broadcasting of `matmul`.
```python
# Original code (Triggers If operator):
# points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1) 

# Modified to an implementation without If nodes, ensuring static graph inference...
```

---

## 3. Breaking Through Inference Speed Bottlenecks: Eliminating Operator-Level Latency Assassins

When I successfully compiled the first TRT Engine, I found the FPS did not meet expectations. By conducting layer-by-layer latency analysis using `trtexec --dumpProfile`, I caught the following three "latency assassins" and deeply optimized them.

### Assassin (1): Element-wise Assignment in Anchor Projection (3ms ➜ 0.1ms)
In the Anchor projection logic before each Deformable Attention, the original code used slicing to forcibly construct a rotation matrix (`rotation_mat`). This approach has little impact in PyTorch but caused a delay of up to 3ms in TRT.
**Optimization Solution: Pure Algebraic Unrolling.**
Extract `cos` and `sin`, and directly calculate using the algebraic formula `x_rot = x * cos_y - y * sin_y`, completely eliminating `torch.stack` and matrix construction.
```python
# 🔥 Core Optimization: Algebraic unrolling, eliminating torch.stack concatenation and matrix construction
cos_y = anchor[..., COS_YAW].unsqueeze(-1)
sin_y = anchor[..., SIN_YAW].unsqueeze(-1)
x, y, z = key_points[..., 0], key_points[..., 1], key_points[..., 2]

# Pure element-wise arithmetic operations, TensorRT will 100% fuse them into a single Kernel!
x_rot = x * cos_y - y * sin_y
y_rot = x * sin_y + y * cos_y
z_rot = z

rotated_point = torch.stack([x_rot, y_rot, z_rot], dim=-1)
key_points = rotated_point + anchor[..., [X, Y, Z]].unsqueeze(-2)
```

### Assassin (2): Softmax on Non-Contiguous Memory (3~4ms ➜ 0.2ms)
When obtaining DFA weights, the original code directly called `.softmax(dim=-2)`.
For tensors in the Map head with a length up to 7200 ($4 \times 6 \times 300$), this **Strided Memory Access** greatly destroyed the GPU cache hit rate, causing the latency to soar to 3-4ms.
**Optimization Solution: Post-Processing after Transpose.**
```python
# ==========================================================
# 🔥 Ultimate Optimization: Transpose to contiguous memory for ultra-fast calculation
# ==========================================================
weights = self.weights_fc(feature).reshape(bs, num_anchor, -1, self.num_groups)

# 1. Transpose to expose the softmax dimension (-1) to the end, ensuring physical memory is contiguous
weights = weights.transpose(-1, -2) # [bs, num_anchor, num_groups, -1]

# 2. Execute blazing-fast Softmax on contiguous memory (TRT will take off instantly here)
weights = weights.softmax(dim=-1)

# 3. Transpose back to recover the shape
weights = weights.transpose(-1, -2) 
```

### Assassin (3): Thread Disaster of DFA Operator in Map Head (4~6ms ➜ 0.3ms)
In perception tasks:
* **Det Head (900 Anchor, 13 Pts)**: Very suitable for the DFA operator provided by official `mmdeploy`. Each thread is responsible for 13 sampling loops, taking about 0.2ms.
* **Map Head (100 Anchor, 300 Pts)**: Caused a severe performance disaster. If the logic of running all Pts in a single thread is kept, each thread needs to execute $4 \times 6 \times 300 = 7200$ loops. This causes a small number of threads to be overwhelmed, while a large number of GPU threads remain idle. The core utilization rate is extremely low, taking up to 4~6ms.

**Optimization Solution: Dynamic Kernel Routing & Point-level Parallelism**
I rewrote the underlying CUDA Kernel (see `projects/trt_plugin/deformable_aggregation_kernel.cu`):
* Designed `dfa_atomic_kernel` specifically for high-point-count scenarios.
* I completely released the 300 sampling points originally enclosed in a single thread into **independent threads** for parallel computation. This instantly boosted the number of concurrent threads from three thousand to **nearly a million**, instantly filling the entire GPU (SMs).
* Combined with atomic addition (`atomicAdd`) to ensure safe write-back of results, ultimately causing the DFA inference latency of the Map head to drop off a cliff **down to 0.3~0.5ms**.
[中文版](README_CN.md) | English

# SparseDrive TensorRT Deployment and Acceleration Guide

## 📖 Introduction

Based on the official [SparseDrive](https://github.com/swc-17/SparseDrive.git)
source code, this project deploys the end-to-end SparseDrive model to
TensorRT on **two targets**:

1. **Workstation GPU (RTX 3090)** — FP16 multi-engine pipeline
   (~7.4× throughput, planning metrics aligned with PyTorch, see below).
2. **Automotive SoC (NVIDIA Orin X domain controller)** — production-style
   delivery: a single engine with **INT8 backbone/neck + FP16 head**, a
   custom **plan+gather dual-kernel DFA plugin**, and an eight-round
   optimization journal — **82.6 ms → 38.99 ms (2.12×)** with perception
   accuracy at parity.

Everything here targets Linux (JetPack 5.1.5 / TensorRT 8.6.1.2 / CUDA 11.4 /
sm_87 on the board side; CUDA 11.6 build host for export). Accuracy gates are
closed-loop: every accepted change re-runs nuScenes evaluation.

## 🚗 Jetson Orin Delivery (headline result)

| Item | Value |
|---|---|
| Engine | single `e_T6.engine` (INT8 backbone+neck QDQ, FP16 head, FP16 DFA weights) |
| Plugin | `libdfaplug_v8.so` — DeformableAggregation plan+gather dual kernel (self-allocating workspace, v3 fallback inside) |
| Latency | **38.99 ms / 25.6 FPS** (100-iter mean; INT8-era baseline 82.6 ms / 12.1 FPS) |
| Accuracy | mini mAP **0.4203** / NDS **0.4738** (FP32 engine baseline 0.4178 — parity) |

### Optimization journal (8 rounds)

| Round | Change | Δ latency |
|---|---|---|
| 1 | kps 6-D broadcast MatMul → 3-D GEMM rewrite | **−19.6 ms** |
| 2 | DFA plugin `__half2` + FP16 IO | **−7.5 ms** |
| 3 | softmax folded into DFA (raw-logits export) | **−4.7 ms** |
| 4 | head full-FP16 graph (P1) | neutral — TRT 8.6 degraded-region wall |
| 5 | 7 graph/kernel experiments (QKV merge, int8 DFA input, …) | no gain (documented) |
| 6 | DFA **plan+gather dual kernel** (v8) | **−6.3 ms** |
| 7 | engine splitting bb/head/det/rest (P4) | **negative** — split tax > fusion gain |
| 8 | FlashAttention plugin for det attention (P5) | **negative** — TRT's fused chains run at ~5 TFLOPS, hand-written wmma ~1.6 TFLOPS |

Negative results are documented with the same rigor as the wins (measured
ceilings, probe/replay numerics methodology, root causes) — see
[docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md), which doubles
as a pitfall log for TRT 8.6 deployment on this class of hardware.

### Deploy on the board

Prerequisites: JetPack 5.1.5, TensorRT 8.6.1.2, CUDA 11.4 (sm_87).

```bash
# 1. Export the delivery ONNX graph (INT8 QDQ + FP16 head + logits-DFA)
python deploy/export_v5.py --checkpoint ckpt/sparsedrive_stage2.pth \
    --protect-groups det_head_output --attn-fp16 \
    --with-ref work_dirs/sparsedrive_small_stage2/mtq_v5_ref.npz \
    --out work_dirs/sparsedrive_small_stage2/v5_P1h.onnx

# 2. Build the board-side engine compiler and compile (INT8 needs a calibration cache)
g++ -O2 -o onnx2engine2 deploy/onnx2engine.cpp -I $TRT_INCLUDE_DIR \
    -L $CUDA_LIB -lnvinfer -lnvonnxparser -lcudart -ldl
./onnx2engine2 v5_P1h.onnx e_T6.engine --int8 --fp16 --ws-mb 2048 \
    --plugins libdfaplug_v8.so

# 3. Build the plugin (nvcc, sm_87)
nvcc -O3 -shared -Xcompiler -fPIC -o libdfaplug_v8.so deploy/dfaplug_v8.cu \
    -I $TRT_INCLUDE_DIR -lnvinfer

# 4. Run (single engine; or chain engines D2D with run_engines)
g++ -O2 -o run_engine deploy/run_engine.cpp -I $TRT_INCLUDE_DIR \
    -L $CUDA_LIB -lnvinfer -lcudart -ldl
./run_engine e_T6.engine libdfaplug_v8.so <inputs_dir> [--iters 100 --warmup 10]
```

Board environment (directories, boot behavior, known firmware limitations,
mini 81-frame closed-loop accuracy gate via
`repro/mini_pipeline_v8.sh` + `deploy/eval_t6_mini_v8.py`):
[docs/BOARD_ENV.md](docs/BOARD_ENV.md).

> Engines bind plugins by **name/version/namespace** at deserialize time, so
> the `.so` can be hot-swapped without rebuilding the engine (rounds 6-8 were
> developed entirely this way, module-level, per the no-full-rebuild rule).

## 🖥️ Workstation Pipeline (RTX 3090, FP16 multi-engine)

### 1. Environment Setup & Plugin Compilation

#### Set up a new virtual environment
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

#### Install dependency packages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

#### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

#### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that `map_annos` is also generated in data_converter, with a roi_size of (30, 60) as default; if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

#### Download trained weights
Download the official model weights, create a `ckpt` directory, and place the weights inside.
* Weight download link: [sparsedrive_stage2.pth](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth)

**[Important] Compile TensorRT Custom Plugins:**
Since the model contains custom operators like `Deformable Aggregation`, you must compile the C++ plugins beforehand:
```bash
cd projects/trt_plugin
mkdir build && cd build
cmake ..
make -j8
# After successful compilation, libSparseDrivePlugin.so will be generated in the current directory.
```

### 2. Export ONNX Models
Considering that in real-world autonomous driving systems, the **perception module** and the **planning/control module** often run at different frequencies, the perception head and the motion & planning head are decoupled at the engineering level and exported as independent engines.
Additionally, since the temporal model has different graph structures for the initial frame (without historical features) and subsequent frames (with historical features), they are exported separately.

**Export Perception Module (Det & Map):**
```bash
# This will export both the initial frame (sparsedrive_multihead_first.onnx) and subsequent frames (sparsedrive_multihead.onnx)
python tools/export_onnx_det_map.py \
    --config projects/configs/sparsedrive_small_stage2.py \
    --checkpoint ckpt/sparsedrive_stage2.pth \
    --out work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx
```

**Export Planning & Control Module (Motion & Planning):**
```bash
# This will export both the initial frame and subsequent frames of the Motion & Plan model.
python tools/export_onnx_motion.py \
    --config projects/configs/sparsedrive_small_stage2.py \
    --checkpoint ckpt/sparsedrive_stage2.pth \
    --out work_dirs/sparsedrive_small_stage2/motion_plan_engine.onnx
```

### 3. Compile TensorRT Engine (ONNX -> TRT)
*Note: Please ensure that your `onnx2trt.py` script correctly loads `libSparseDrivePlugin.so` internally.*

```bash
# Compile Perception Module
python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx --save work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine

python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.onnx --save work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine

# Compile Planning & Control Module
python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/motion_plan_engine.onnx --save work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine

python onnx2trt.py --onnx work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.onnx --save work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine
```

### 4. Evaluation
Run the following command for end-to-end closed-loop testing:
```bash
python test_trt.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth \
    --engine_perc_init work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine \
    --engine_perc_temp work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine \
    --engine_mo_init work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine \
    --engine_mo_temp work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine
```

**📊 Performance Comparison Report (NVIDIA RTX 3090)**

#### End-to-End Performance
| Method | NDS | AMOTA | minADE (m)* | L2 (m) Avg | Col. (%) Avg | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SparseDrive-S (Official Pytorch)** | **0.525** | **0.386** | **0.620** | **0.610** | 0.100 | 4.8 |
| **SparseDrive-S (TRT FP16)** | 0.520 | 0.370 | 0.648 | 0.612 | **0.092** | **35.7** |

> **Note:** `minADE` uses the metrics for the Car category. Under the premise of highly aligning core perception and planning metrics (L2 error difference is only 0.002m), the TRT engine achieved even better performance in **Average Collision Rate (Col. Avg)**. Meanwhile, inference throughput (FPS) achieved a massive leap of **~7.4x**.

#### Detailed Planning Metrics
| Method | L2 1s | L2 2s | L2 3s | L2 Avg | Col. 1s | Col. 2s | Col. 3s | Col. Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official (Official Pytorch)** | 0.300 | **0.580** | **0.950** | **0.610** | **0.010%** | **0.050%** | 0.230% | 0.100% |
| **TRT FP16** | **0.299** | 0.581 | 0.957 | 0.612 | **0.010%** | 0.054% | **0.212%** | **0.092%** |

### 5. Inference Speed Test (Latency Profiling)
You can use the Python script for macroscopic FPS testing:
```bash
python fps.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --mode trt
```

To analyze the microscopic latency of each operator, it is recommended to use `trtexec` to generate a Profiling report:
```bash
# Test Perception Module (Det & Map)
trtexec --loadEngine=work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine \
        --plugins=projects/trt_plugin/build/libSparseDrivePlugin.so \
        --dumpProfile --iterations=100 > map_det_inference.log

# Test Planning & Control Module (Motion & Plan)
trtexec --loadEngine=work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine \
        --plugins=projects/trt_plugin/build/libSparseDrivePlugin.so \
        --dumpProfile --iterations=100 > motion_inference.log
```

## 🗂️ Repository Layout

```
deploy/                       deployment & optimization toolchain (see deploy/README.md)
  export_v5.py                  INT8+FP16 delivery graph export (logits-DFA)
  make_v5_p2h.py / split_*.py   graph surgery (P1 head-FP16, P4 engine split)
  dfaplug_v8.cu                 delivery DFA plugin (plan+gather dual kernel)
  dfaplug_v3.cu                 fallback kernel (same semantics, ulp-equal)
  run_engine.cpp / run_engines.cpp / onnx2engine.cpp   board-side C++ tools
  ssa_surgery2.py / _fa_part.cu / selftest_fa*.cu       P5 FlashAttention (negative result, kept for the record)
  board_*.py                    paramiko board drivers (env: BOARD_HOST / BOARD_PASS)
  eval_t6_mini_v8.py            mini 81-frame closed-loop accuracy gate
docs/
  OPTIMIZATION_SUMMARY.md     ★ the full 8-round journal + pitfall log (read this first)
  BOARD_ENV.md                  board environment, toolchain, known limitations
  quick_start.md                condensed workstation quick start
repro/                         board-side mini pipeline scripts (bash)
projects/ tools/ scripts/      SparseDrive upstream + export modifications
```

## 📍 Roadmap

- [x] INT8 PTQ pipeline for backbone/neck on automotive SoC (this repo, Orin delivery)
- [ ] Frame-level pipelining (backbone of frame N overlapped with decoder of frame N−1; ~1.7× throughput, no engine change)
- [ ] TensorRT upgrade validation — the ~12 ms Myelin degraded-fusion regions and the det-attention materialization chain are both version-limited and expected to collapse on TRT 9/10
- [ ] Pure C++ ultra-fast inference deployment pipeline
- [ ] ROS2 integration

## 📚 Technical Details & Pitfall Avoidance Guide

- [TECH_DETAILS.md](TECH_DETAILS.md) — workstation-era deep dive: matrix-refactoring of temporal alignment, ONNX control-flow elimination, CUDA op squeezing.
- [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) — Orin-era journal: every accepted and rejected optimization, with measurements, methodology (probe/replay numerics, hot-swap plugin binding) and the pitfall log (NVRTC JIT breakage, workspace baked into plan, plugin version pairing, …).

Discussions and exchanges are highly welcome!

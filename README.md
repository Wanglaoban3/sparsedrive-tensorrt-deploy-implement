[中文版](README_CN.md) | English

# SparseDrive TensorRT Deployment and Acceleration Guide

## 📖 Introduction
Based on the official [SparseDrive](https://github.com/swc-17/SparseDrive.git) source code, this project achieves the FP16 precision ONNX export of the end-to-end SparseDrive model, alongside TensorRT engine compilation and extreme optimization. While strictly maintaining the accuracy of core planning metrics, the inference efficiency has been significantly improved, aiming to accelerate the deployment of end-to-end autonomous driving in the community.

## 🚀 Quick Start

### 1. Environment Setup & Plugin Compilation
### Set up a new virtual environment
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

### Install dependency packpages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Download trained weights
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
Considering that in real-world autonomous driving systems, the **perception module** and the **planning/control module** often run at different frequencies, we decoupled the perception head from the motion & planning head at the engineering level, exporting them as independent engines.
Additionally, since the temporal model has different graph structures for the initial frame (without historical features) and subsequent frames (with historical features), we exported them separately.

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
| **SparseDrive-S (Official)** | **0.525** | **0.386** | **0.620** | **0.610** | 0.100 | 4.8 |
| **SparseDrive-S (Ours TRT)** | 0.520 | 0.370 | 0.648 | 0.612 | **0.092** | **35.7** |

> **Note:** `minADE` uses the metrics for the Car category. Under the premise of highly aligning core perception and planning metrics (L2 error difference is only 0.002m), the TRT engine achieved even better performance in **Average Collision Rate (Col. Avg)**. Meanwhile, inference throughput (FPS) achieved a massive leap of **~7.4x**.

#### Detailed Planning Metrics
| Method | L2 1s | L2 2s | L2 3s | L2 Avg | Col. 1s | Col. 2s | Col. 3s | Col. Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official** | 0.300 | **0.580** | **0.950** | **0.610** | **0.010%** | **0.050%** | 0.230% | 0.100% |
| **Ours TRT** | **0.299** | 0.581 | 0.957 | 0.612 | **0.010%** | 0.054% | **0.212%** | **0.092%** |

> **Data Analysis:**
> * **L2 Distance:** The TRT version tightly matches the official PyTorch version. The short-term (1s) prediction even reaches an excellent level of 0.299m.
> * **Collision Rate:** The TRT deployment demonstrates extremely high safety. While the 1s collision rate matches the official version (0.010%), the long-term (3s) collision rate drops significantly (from 0.230% to 0.212%). This indicates that our FP16 engine and operator fusion are highly robust in handling temporal features.

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

### 6. Future Plans (To-Do List)
- [ ] Complete a pure C++ ultra-fast inference deployment pipeline for SparseDrive.
- [ ] Develop a QAT (Quantization-Aware Training) INT8 pipeline for SparseDrive.
- [ ] Integrate SparseDrive with the ROS2 autonomous driving system.
- [ ] Further in-depth Kernel-level optimization for VRAM and memory access in long temporal queues.

### 7. Technical Details & Pitfall Avoidance Guide
For experience and insights gained during the optimization process—such as operator fusion, ONNX control flow elimination (pruning `If` nodes), dynamic dimension support, and C++ plugin development—please refer to the [TECH_DETAILS.md](TECH_DETAILS.md) document. Discussions and exchanges are highly welcome!
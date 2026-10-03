#!/bin/bash
# 板端环境说明文件，放在板上 /opt/m0/README.md
cat > /opt/m0/README.md << 'READMEEOF'
# Orin X SparseDrive 部署环境

## 开机行为
开机自动重启两次，第三次才稳定。SSH 可用前等 2-3 分钟。

## 交付物
- 引擎: /opt/m0/trt-dev/models/e_T6.engine (135MB)
- 插件: /usr/local/lib/libdfaplug_v8.so (plan+gather 双内核, 自分配 ws)
- ONNX: /opt/m0/trt-dev/models/v5_P1h.onnx
- 性能: 38.99ms / 25.6 FPS / mini mAP 0.4203 / NDS 0.4738

## 推理
run_engine models/e_T6.engine /usr/local/lib/libdfaplug_v8.so \
  <inputs_dir> [--iters N] [--warmup N] [--dump <dir>]

# 多引擎链式 (D2D 零拷贝)
run_engines e_bb.engine e_det.engine /usr/local/lib/libdfaplug_v8.so \
  vec/mini/in_00 --iters 100 --warmup 10

## 编译插件
nvcc -O3 -shared -Xcompiler -fPIC src/dfaplug_v8.cu \
  -I include -arch=sm_87 -lnvinfer \
  -o /usr/local/lib/libdfaplug_v8.so

## 编译引擎
onnx2engine2 models/v5_P1h.onnx models/e_T6.engine \
  --int8 --fp16 --ws-mb 2048 \
  --plugins /usr/local/lib/libdfaplug_v8.so

## 已知限制
- TRT 8.6.1.2 NVRTC fp32 JIT 损坏 (fp16 正常)
- decoder 融合区 ForeignNode ~12.3ms (TRT 8.6 规划器限制, 升级有望消解)
- 插件用 v8 (v3 语义等价可互换; 其他老版本不可混用)
- 场景边界需宿主侧重置时序状态

## 验证
mini 81 帧在 vec/mini/，链式推理 bash repro/mini_pipeline_v8.sh
本地评估 python deploy/eval_t6_mini_v8.py

## 详细文档
见项目 docs/BOARD_ENV.md 和 docs/OPTIMIZATION_SUMMARY.md
READMEEOF
echo "README written to /opt/m0/README.md"

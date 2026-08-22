python tools/onnx2trt.py `
  --onnx work_dirs/sparsedrive_small_stage1/sparsedrive_multihead.onnx `
  --save work_dirs/sparsedrive_small_stage1/sparsedrive_multihead.engine `
  --plugin projects/trt_plugin/build/libSparseDrivePlugin.so `
  --fp16 `
  --workspace-mb 2048 `
  --builder-optimization-level 0
/usr/src/tensorrt/bin/trtexec \
  --onnx=work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.onnx \
  --saveEngine=work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine \
  --plugins=./projects/trt_plugin/build/libSparseDrivePlugin.so \
  --fp16 \
  --memPoolSize=workspace:8192 \
  --verbose \
  --tacticSources=-JIT_CONVOLUTIONS
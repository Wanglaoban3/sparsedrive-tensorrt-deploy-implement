@echo off
REM Full-val INT8-QAT eval (6019 frames), direct log to _eval_int8_val.log
cd /d H:\projects\sparsedrive-tensorrt-deploy-implement
H:\miniconda3\envs\sparsedrive_deploy\python.exe deploy\eval_nuscenes.py ^
  --mode quant ^
  --checkpoint ckpt\sparsedrive_stage2_qat.pth ^
  --skip-json deploy\artifacts\sparsedrive_ptq_sensitivity.json ^
  --skip-k 28 ^
  --tag int8qat_val ^
  --version v1.0-trainval ^
  --ann-file data\infos\nuscenes_infos_val.pkl ^
  --data-root H:\datasets\nuscenes-trainval ^
  > _eval_int8_val.log 2>&1

@echo off
REM Full-val fp32 eval (6019 frames), direct log to _eval_fp32_val.log
cd /d H:\projects\sparsedrive-tensorrt-deploy-implement
H:\miniconda3\envs\sparsedrive_deploy\python.exe deploy\eval_nuscenes.py ^
  --mode fp32 ^
  --checkpoint ckpt\sparsedrive_stage2.pth ^
  --tag fp32_val ^
  --version v1.0-trainval ^
  --ann-file data\infos\nuscenes_infos_val.pkl ^
  --data-root H:\datasets\nuscenes-trainval ^
  > _eval_fp32_val.log 2>&1

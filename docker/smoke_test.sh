#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sparsedrive
python - <<'PY'
import os
import torch
from nuscenes.nuscenes import NuScenes

assert torch.cuda.is_available(), 'CUDA is not available inside the container'
print('GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__)

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
sample = nusc.sample[0]
camera = nusc.get('sample_data', sample['data']['CAM_FRONT'])
image_path = os.path.join('data/nuscenes', camera['filename'])
assert os.path.isfile(image_path), image_path
print('nuScenes samples:', len(nusc.sample))
print('first sample token:', sample['token'])
print('camera input:', image_path)
print('camera bytes:', os.path.getsize(image_path))
PY

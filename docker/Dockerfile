FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=6.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3.8 \
    python3.8-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /workspace/sparsedrive
COPY requirement.txt /tmp/requirement.txt

RUN python -m pip install --upgrade 'pip<24' setuptools wheel \
    && python -m pip install \
        torch==1.13.0+cu116 \
        torchvision==0.14.0+cu116 \
        torchaudio==0.13.0 \
        --extra-index-url https://download.pytorch.org/whl/cu116 \
    && python -m pip install \
        numpy==1.23.5 \
        mmcv-full==1.7.1 \
        mmdet==2.28.2 \
        urllib3==1.26.16 \
        pyquaternion==0.9.9 \
        nuscenes-devkit==1.1.10 \
        yapf==0.33.0 \
        tensorboard==2.14.0 \
        motmetrics==1.1.3 \
        pandas==1.1.5 \
        opencv-python==4.8.1.78 \
        prettytable==3.7.0 \
        scikit-learn==1.3.0 \
    && python -m pip install tensorrt==8.5.3.1 --extra-index-url https://pypi.nvidia.com

COPY . /workspace/sparsedrive

RUN python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda runtime', torch.version.cuda)
PY

RUN cd projects/mmdet3d_plugin/ops && python setup.py build_ext --inplace

COPY docker/entrypoint.sh /usr/local/bin/sparsedrive-entrypoint
RUN chmod +x /usr/local/bin/sparsedrive-entrypoint

ENTRYPOINT ["/usr/local/bin/sparsedrive-entrypoint"]
CMD ["bash"]

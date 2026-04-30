#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== 安装核心 Python 依赖 ==="

pip install \
    pytorch-lightning \
    pyrender \
    opencv-python \
    yacs \
    scikit-image \
    einops \
    timm \
    dill \
    pandas \
    rich \
    hydra-core \
    hydra-submitit-launcher \
    hydra-colorlog \
    pyrootutils \
    webdataset \
    chump \
    roma \
    joblib \
    seaborn \
    wandb \
    appdirs \
    ffmpeg \
    cython \
    jsonlines \
    pytest \
    xtcocotools \
    loguru \
    optree \
    fvcore \
    pycocotools \
    tensorboard \
    huggingface_hub

echo ""
echo "=== 安装 networkx (固定版本) ==="
pip install "networkx==3.2.1"

echo ""
echo "=== 核心依赖安装完成 ==="

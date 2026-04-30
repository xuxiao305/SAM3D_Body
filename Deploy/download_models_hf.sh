#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PYTHON=$ENV/bin/python
PIP=$ENV/bin/pip

# 设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

echo "=== 下载 SAM3D Body DINOv3 模型 ==="
$PYTHON -c "
from huggingface_hub import snapshot_download
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print('下载 sam-3d-body-dinov3 ...')
path = snapshot_download(
    'jetjodh/sam-3d-body-dinov3',
    local_dir='/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3',
)
print(f'✅ 下载完成: {path}')
" 2>&1

echo ""
echo "=== 下载 SAM3D Body ViT-H 模型 ==="
$PYTHON -c "
from huggingface_hub import snapshot_download
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print('下载 sam-3d-body-vith ...')
path = snapshot_download(
    'jetjodh/sam-3d-body-vith',
    local_dir='/root/SAM3D_Body/checkpoints/sam-3d-body-vith',
)
print(f'✅ 下载完成: {path}')
" 2>&1

echo ""
echo "=== 验证模型文件 ==="
echo "--- DINOv3 ---"
ls -lh /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model.ckpt 2>/dev/null || echo "❌ model.ckpt 缺失"
ls -lh /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model_config.yaml 2>/dev/null || echo "❌ model_config.yaml 缺失"
ls -lh /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt 2>/dev/null || echo "❌ mhr_model.pt 缺失"

echo "--- ViT-H ---"
ls -lh /root/SAM3D_Body/checkpoints/sam-3d-body-vith/model.ckpt 2>/dev/null || echo "❌ model.ckpt 缺失"
ls -lh /root/SAM3D_Body/checkpoints/sam-3d-body-vith/model_config.yaml 2>/dev/null || echo "❌ model_config.yaml 缺失"
ls -lh /root/SAM3D_Body/checkpoints/sam-3d-body-vith/assets/mhr_model.pt 2>/dev/null || echo "❌ mhr_model.pt 缺失"

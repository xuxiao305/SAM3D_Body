#!/bin/bash
# 在丹炉上运行最小推理（无 detector / segmentor / fov）
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

cd /root/SAM3D_Body/Main
export PYOPENGL_PLATFORM=egl

python /tmp/minimal_inference.py 2>&1

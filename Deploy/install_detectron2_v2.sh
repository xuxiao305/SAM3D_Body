#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== 安装编译工具 ==="
apt-get update -qq && apt-get install -y -qq g++ python3-dev 2>&1 | tail -5

echo "=== 安装 Detectron2 ==="
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -10

echo "=== 验证 ==="
python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
echo "✅ Detectron2 安装完成"

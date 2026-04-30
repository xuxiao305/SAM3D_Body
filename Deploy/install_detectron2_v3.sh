#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

# 设置 conda 编译器路径
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=/usr/local/cuda

echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"
ls -la $CC $CXX 2>&1 | head -5
ls $CUDA_HOME/bin/nvcc 2>&1

echo ""
echo "=== 安装 Detectron2 ==="
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -15

echo ""
echo "=== 验证 ==="
python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
echo "✅ Detectron2 安装完成"

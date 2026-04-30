#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== 安装 CUDA Toolkit via conda ==="
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit 2>&1 | tail -15

echo ""
echo "=== 检查 nvcc ==="
which nvcc
nvcc --version

echo ""
echo "=== 设置编译环境变量 ==="
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX

echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"

echo ""
echo "=== 安装 Detectron2 ==="
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -15

echo ""
echo "=== 验证 ==="
python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
echo "✅ Detectron2 安装完成"

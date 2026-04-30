#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== Step 1: 安装 CUDA Toolkit via conda ==="
conda install -y -c nvidia cuda-toolkit=12.4 2>&1 | tail -15

echo ""
echo "=== Step 2: 检查 nvcc ==="
which nvcc && nvcc --version || echo "nvcc still not found, trying conda bin..."

# 如果 nvcc 在 conda 环境的 bin 目录
if [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
    echo "nvcc found at $CONDA_PREFIX/bin/nvcc"
else
    echo "Searching for nvcc..."
    find $CONDA_PREFIX -name nvcc 2>/dev/null
fi

echo ""
echo "=== Step 3: 安装 Detectron2 ==="
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"

pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -15

echo ""
echo "=== Step 4: 验证 ==="
python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
echo "✅ Detectron2 安装完成"

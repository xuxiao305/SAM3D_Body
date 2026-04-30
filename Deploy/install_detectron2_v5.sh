#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# 关键：设置 CUDA include 路径
export C_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "CUDA_HOME=$CUDA_HOME"
echo "CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH"
echo ""
echo "验证 cuda_runtime.h 可找到..."
echo '#include <cuda_runtime.h>' | $CC -E -x c - > /dev/null 2>&1 && echo "✅ gcc 能找到 cuda_runtime.h" || echo "❌ gcc 找不到 cuda_runtime.h"

echo ""
echo "=== 安装 Detectron2 ==="
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -20

echo ""
echo "=== 验证 ==="
python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
echo "✅ Detectron2 安装完成"

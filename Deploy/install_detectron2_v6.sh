#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== Step 1: 安装完整 CUDA 开发包 ==="
conda install -y -c nvidia cuda-cudart cuda-cupti cuda-libraries cuda-nvtx cuda-nvrtc cuda-nvcc cuda-libraries-dev 2>&1 | tail -10

echo ""
echo "=== Step 2: 安装兼容的 GCC 12 ==="
conda install -y gxx_linux-64=12 2>&1 | tail -10

echo ""
echo "=== Step 3: 验证 ==="
echo "GCC version:"
$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ --version | head -1
echo ""
echo "nvcc version:"
$CONDA_PREFIX/bin/nvcc --version | tail -1
echo ""
echo "CUDA headers check:"
ls $CONDA_PREFIX/targets/x86_64-linux/include/cusolverDn.h 2>/dev/null && echo "✅ cusolverDn.h found" || echo "❌ cusolverDn.h NOT found"
ls $CONDA_PREFIX/include/cusolverDn.h 2>/dev/null && echo "✅ cusolverDn.h found in include" || echo "❌ cusolverDn.h NOT found in include"
find $CONDA_PREFIX -name "cusolverDn.h" 2>/dev/null | head -3

echo ""
echo "=== Step 4: 安装 Detectron2 ==="
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -20

echo ""
echo "=== 验证 Detectron2 ==="
python -c "import detectron2; print('detectron2 version:', detectron2.__version__)" 2>&1

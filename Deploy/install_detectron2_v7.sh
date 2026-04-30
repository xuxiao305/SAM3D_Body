#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh

# 不用 conda activate（环境损坏），直接用绝对路径
ENV=/root/miniconda3/envs/sam3d_body
PYTHON=$ENV/bin/python
PIP=$ENV/bin/pip

echo "=== 安装 CUDA dev headers via pip ==="
$PIP install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 2>&1 | tail -5

echo ""
echo "=== 收集所有 CUDA include 路径 ==="
CUDA_INCLUDES=""
for dir in $(find $ENV/lib/python3.11/site-packages/nvidia -type d -name "include" 2>/dev/null); do
    echo "Found: $dir"
    CUDA_INCLUDES="$CUDA_INCLUDES -I$dir"
done
# 也加上 conda targets
CUDA_INCLUDES="$CUDA_INCLUDES -I$ENV/targets/x86_64-linux/include"
echo ""
echo "Extra include flags:$CUDA_INCLUDES"

echo ""
echo "=== 设置编译环境 ==="
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc
export CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH

# 关键：所有 CUDA include 路径
export CFLAGS="$CUDA_INCLUDES"
export CXXFLAGS="$CUDA_INCLUDES"
export C_INCLUDE_PATH=$(echo $CUDA_INCLUDES | sed 's/-I/:/g' | sed 's/^://'):${ENV}/targets/x86_64-linux/include:${ENV}/include
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 关键：NVCC 额外标志 — 允许 GCC 15 + 加入 include 路径
export TORCH_CUDA_ARCH_LIST="8.0"
export NVCC_FLAGS="--allow-unsupported-compiler $CUDA_INCLUDES"
export CUDAFLAGS="--allow-unsupported-compiler $CUDA_INCLUDES"

echo "CC=$CC"
$CC --version | head -1
echo "CUDA_HOME=$CUDA_HOME"
echo "NVCC_FLAGS=$NVCC_FLAGS"

echo ""
echo "=== 安装 Detectron2 ==="
# 先 clone 再手动安装，以便控制编译参数
cd /tmp
rm -rf detectron2_build
git clone https://github.com/facebookresearch/detectron2.git detectron2_build 2>&1 | tail -3
cd detectron2_build
git checkout a1ce2f9 2>&1 | tail -1

# 设置 NVCC_PREPEND_FLAGS 来注入 --allow-unsupported-compiler
export NVCC_PREPEND_FLAGS="--allow-unsupported-compiler"

$PIP install -e . --no-build-isolation --no-deps 2>&1 | tail -25

echo ""
echo "=== 验证 ==="
$PYTHON -c "import detectron2; print('detectron2 version:', detectron2.__version__)" 2>&1

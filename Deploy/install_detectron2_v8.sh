#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== Step 1: 修复缺失依赖 ==="
$PIP install six 2>&1 | tail -3

echo ""
echo "=== Step 2: 验证基础导入 ==="
$PYTHON -c "import six; print('six ok'); import matplotlib; print('matplotlib ok'); import pytorch_lightning; print('pytorch_lightning ok')" 2>&1

echo ""
echo "=== Step 3: 重新安装 detectron2 (从源码编译，不用 editable) ==="
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc
export CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0"

# 设置 CUDA include 路径
NVIDIA_INC=$ENV/lib/python3.11/site-packages/nvidia
export CPLUS_INCLUDE_PATH=$ENV/targets/x86_64-linux/include:$NVIDIA_INC/cublas/include:$NVIDIA_INC/cusparse/include:$NVIDIA_INC/cusolver/include:$NVIDIA_INC/cuda_runtime/include:$NVIDIA_INC/cudnn/include:$NVIDIA_INC/curand/include:$NVIDIA_INC/cufft/include
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export NVCC_PREPEND_FLAGS="--allow-unsupported-compiler -I$NVIDIA_INC/cublas/include -I$NVIDIA_INC/cusparse/include -I$NVIDIA_INC/cusolver/include -I$NVIDIA_INC/cuda_runtime/include -I$NVIDIA_INC/cudnn/include -I$NVIDIA_INC/curand/include -I$NVIDIA_INC/cufft/include -I$ENV/targets/x86_64-linux/include"

# 先清理旧 build
rm -rf /tmp/detectron2_build

# clone + pip install (non-editable)
cd /tmp
git clone https://github.com/facebookresearch/detectron2.git detectron2_build 2>&1 | tail -3
cd detectron2_build
git checkout a1ce2f9 2>&1 | tail -1

$PIP install . --no-build-isolation --no-deps 2>&1 | tail -20

echo ""
echo "=== Step 4: 验证 detectron2 ==="
$PYTHON -c "import detectron2; print('detectron2 version:', detectron2.__version__)" 2>&1

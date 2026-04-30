#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== 查找 cuda_runtime.h ==="
find $CONDA_PREFIX -name "cuda_runtime.h" 2>/dev/null | head -5
echo ""
echo "=== 查找 CUDA include 目录 ==="
find $CONDA_PREFIX -type d -name "include" -path "*cuda*" 2>/dev/null | head -10
echo ""
echo "=== 查找所有 conda cuda 相关包 ==="
conda list | grep cuda | head -20
echo ""
echo "=== 检查 $CONDA_PREFIX/include ==="
ls $CONDA_PREFIX/include/ 2>/dev/null | grep -i cuda | head -10

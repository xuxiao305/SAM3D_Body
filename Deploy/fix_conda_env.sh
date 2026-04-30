#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh

echo "=== 修复 conda 环境 ==="
# conda 环境可能缺少 conda-meta，需要修复
if [ ! -d /root/miniconda3/envs/sam3d_body/conda-meta ]; then
    echo "创建 conda-meta 目录..."
    mkdir -p /root/miniconda3/envs/sam3d_body/conda-meta
    conda create -n sam3d_body_fix --clone sam3d_body -y 2>&1 | tail -5 || true
fi

echo ""
echo "=== 检查环境 ==="
conda env list | grep sam3d

echo ""
echo "=== 尝试直接安装 gcc_linux-64=12 到环境 ==="
# 使用 --prefix 模式绕过环境检查
conda install -p /root/miniconda3/envs/sam3d_body -y gcc_linux-64=12 gxx_linux-64=12 -c conda-forge 2>&1 | tail -10

echo ""
echo "=== GCC 版本 ==="
/root/miniconda3/envs/sam3d_body/bin/x86_64-conda-linux-gnu-g++ --version 2>&1 | head -1

echo ""
echo "=== 安装 CUDA cusparse dev via pip ==="
/root/miniconda3/envs/sam3d_body/bin/pip install nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12 2>&1 | tail -5

echo ""
echo "=== 查找 cusparse.h ==="
find /root/miniconda3/envs/sam3d_body -name "cusparse.h" 2>/dev/null | head -5
find /root/miniconda3/envs/sam3d_body -name "cusolverDn.h" 2>/dev/null | head -5

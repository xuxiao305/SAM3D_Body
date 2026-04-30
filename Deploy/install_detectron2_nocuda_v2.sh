#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

# 创建 g++ symlink 使其可用
ln -sf $ENV/bin/x86_64-conda-linux-gnu-gcc $ENV/bin/gcc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-g++ $ENV/bin/g++
ln -sf $ENV/bin/x86_64-conda-linux-gnu-gcc-ar $ENV/bin/gcc-ar 2>/dev/null || true

export CC=$ENV/bin/gcc
export CXX=$ENV/bin/g++
export PATH=$ENV/bin:$PATH

echo "g++ version:"
g++ --version | head -1

echo ""
echo "=== 从 GitHub 安装 detectron2 (CPU-only, 用 conda g++) ==="
FORCE_CUDA=0 $PIP install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation 2>&1 | tail -15

echo ""
echo "=== 验证 ==="
$PYTHON -c "import detectron2; print('detectron2 version:', detectron2.__version__)" 2>&1

echo ""
echo "=== 检查关键导入 ==="
$PYTHON -c "
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2 import model_zoo
from detectron2.layers import ShapeSpec
print('✅ 关键 detectron2 模块导入成功')
" 2>&1

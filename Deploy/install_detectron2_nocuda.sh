#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== 从 GitHub 安装 detectron2 纯 Python 模式 ==="
# 不设 CUDA 编译环境，强制跳过 CUDA ops 编译
unset CC CXX CUDA_HOME TORCH_CUDA_ARCH_LIST NVCC_PREPEND_FLAGS

# 用 FORCE_CUDA=0 强制不编译 CUDA 扩展
FORCE_CUDA=0 $PIP install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation 2>&1 | tail -10

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
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
print('✅ 所有关键 detectron2 模块导入成功')
" 2>&1

echo ""
echo "=== 验证 SAM3D Body 导入 ==="
cd /root/SAM3D_Body/Main
PYTHONPATH=/root/SAM3D_Body/Main $PYTHON -c "
from sam_3d_body.build_models import build_sam3d_body_estimator
print('✅ SAM3D Body 核心导入成功')
" 2>&1

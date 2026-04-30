#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== 降级 setuptools 确保 pkg_resources 可用 ==="
$PIP install "setuptools<71" 2>&1 | tail -5

echo ""
echo "=== 验证 pkg_resources ==="
$PYTHON -c "import pkg_resources; print('pkg_resources ok, version:', pkg_resources.__version__)" 2>&1

echo ""
echo "=== 验证 detectron2 ==="
$PYTHON -c "
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2 import model_zoo
from detectron2.layers import ShapeSpec
print('✅ detectron2 关键模块导入成功')
" 2>&1

echo ""
echo "=== 验证 SAM3D Body ==="
cd /root/SAM3D_Body/Main
PYTHONPATH=/root/SAM3D_Body/Main $PYTHON -c "
import sam_3d_body
print('sam_3d_body 导入成功')
from sam_3d_body.build_models import load_sam_3d_body, load_sam_3d_body_hf
print('✅ load_sam_3d_body, load_sam_3d_body_hf 导入成功')
" 2>&1

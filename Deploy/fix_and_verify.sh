#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== 安装 setuptools (提供 pkg_resources) ==="
$PIP install setuptools 2>&1 | tail -3

echo ""
echo "=== 验证 detectron2 关键导入 ==="
$PYTHON -c "
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2 import model_zoo
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
print('✅ 所有 detectron2 关键模块导入成功')
" 2>&1

echo ""
echo "=== 验证 SAM3D Body 完整导入 ==="
cd /root/SAM3D_Body/Main
PYTHONPATH=/root/SAM3D_Body/Main $PYTHON -c "
from sam_3d_body.build_models import build_sam3d_body_estimator
from sam_3d_body.metadata.mhr70 import MHR70_JOINTS
print('✅ SAM3D Body 核心导入成功, MHR70 关节数:', len(MHR70_JOINTS))
" 2>&1

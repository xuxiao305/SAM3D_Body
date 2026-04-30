#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PYTHON=$ENV/bin/python

echo "=== 验证 detectron2 ==="
$PYTHON -c "
import detectron2
print('detectron2 version:', detectron2.__version__)

# 检查关键模块
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
print('✅ 核心模块导入成功')

# 检查 CUDA ops 是否可用
try:
    from detectron2.layers import batched_nms
    print('✅ batched_nms 可用 (可能走 CPU fallback)')
except Exception as e:
    print(f'⚠️  batched_nms: {e}')

try:
    from detectron2.modeling import build_model
    print('✅ build_model 可用')
except Exception as e:
    print(f'❌ build_model: {e}')
"

echo ""
echo "=== 验证 SAM3D Body 核心导入 ==="
cd /root/SAM3D_Body/Main
PYTHONPATH=/root/SAM3D_Body/Main:$PYTHONPATH $PYTHON -c "
import sys
sys.path.insert(0, '.')

# 基础模块
from sam_3d_body.build_models import build_sam3d_body_estimator
print('✅ build_sam3d_body_estimator 导入成功')

from sam_3d_body.metadata.mhr70 import MHR70_JOINTS
print('✅ MHR70_JOINTS 导入成功, 关节数:', len(MHR70_JOINTS))

print('核心模块导入验证完成')
"

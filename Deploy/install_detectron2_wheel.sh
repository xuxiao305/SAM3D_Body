#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
ENV=/root/miniconda3/envs/sam3d_body
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

# 先清理之前的残留
$PIP uninstall detectron2 -y 2>&1 | tail -3
rm -rf /tmp/detectron2_build

echo "=== 尝试安装 detectron2 预编译 wheel ==="
# 方式1: 官方预编译 wheel (cu121 + torch2.5)
$PIP install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.5/index.html 2>&1 | tail -10

echo ""
echo "=== 验证 ==="
$PYTHON -c "import detectron2; print('detectron2 version:', detectron2.__version__)" 2>&1

echo ""
echo "=== 检查 CUDA ops ==="
$PYTHON -c "
from detectron2.layers import batched_nms,roi_align
import torch
boxes = torch.tensor([[0,0,100,100],[0,0,50,50]], dtype=torch.float32, device='cuda')
scores = torch.tensor([0.9,0.1], dtype=torch.float32, device='cuda')
idxs = torch.tensor([0,0], dtype=torch.int64, device='cuda')
result = batched_nms(boxes, scores, idxs, 0.5)
print('batched_nms result:', result)
print('✅ CUDA ops 正常工作')
" 2>&1

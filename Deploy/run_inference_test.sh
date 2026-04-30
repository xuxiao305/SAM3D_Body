#!/bin/bash
# 在丹炉上运行 SAM3D Body 推理测试
# 跳过 detector/segmentor/fov（因为 detectron2 是 CPU-only，且没下载 SAM2/MoGe2）
# 直接用整张图当 bbox

set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== Step 1: 安装 pyrootutils ==="
pip install pyrootutils python-dotenv 2>&1 | tail -3

echo ""
echo "=== Step 2: 检查环境 ==="
cd /root/SAM3D_Body/Main
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Step 3: 运行推理（无 detector/segmentor/fov）==="
export PYOPENGL_PLATFORM=egl

# 先只跑一张图试试
mkdir -p /root/SAM3D_Body/test_images_one
cp /root/SAM3D_Body/test_images/ComfyUI_00029_.png /root/SAM3D_Body/test_images_one/ 2>/dev/null || true
ls /root/SAM3D_Body/test_images_one/

cd /root/SAM3D_Body/Main
python demo.py \
    --image_folder /root/SAM3D_Body/test_images_one \
    --output_folder /root/SAM3D_Body/output_dinov3 \
    --checkpoint_path /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/assets \
    --detector_name "" \
    --segmentor_name "" \
    --fov_name "" \
    2>&1 | tail -50

echo ""
echo "=== Step 4: 输出结果 ==="
ls -lh /root/SAM3D_Body/output_dinov3/ 2>&1 || echo "无输出"

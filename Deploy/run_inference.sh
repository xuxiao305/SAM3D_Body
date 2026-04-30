#!/bin/bash
# Run SAM3D Body inference on DanLu server
# Usage: bash run_inference.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

cd /root/SAM3D_Body/Main

# Check pyrootutils
echo "=== 检查 pyrootutils ==="
python -c "import pyrootutils; print('pyrootutils OK:', pyrootutils.__file__)"

# Check key imports
echo "=== 检查关键导入 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Run inference with DINOv3 model (lighter)
echo ""
echo "=== 运行 SAM3D Body DINOv3 推理 ==="
python demo.py \
    --image_folder /root/SAM3D_Body/test_images \
    --output_folder /root/SAM3D_Body/output_dinov3 \
    --checkpoint_path /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    2>&1

echo ""
echo "=== 推理完成，检查输出 ==="
ls -lh /root/SAM3D_Body/output_dinov3/ 2>/dev/null || echo "无输出文件"

#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== Looking for CUDA ==="
find / -name nvcc -type f 2>/dev/null | head -5
echo ""
echo "=== CUDA libs ==="
find / -name libcudart.so* -type f 2>/dev/null | head -5
echo ""
echo "=== nvidia-smi ==="
nvidia-smi | head -5
echo ""
echo "=== ldconfig ==="
ldconfig -p 2>/dev/null | grep cuda | head -10

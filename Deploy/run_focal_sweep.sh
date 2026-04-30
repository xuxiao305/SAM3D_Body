#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

cd /root/SAM3D_Body/Main
export PYOPENGL_PLATFORM=egl

python /tmp/focal_sweep.py 2>&1

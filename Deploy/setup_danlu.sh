#!/bin/bash
# ============================================================
# SAM3D_Body 丹炉一键部署脚本
# 用途：在丹炉服务器上创建项目环境、安装依赖
# 注意：模型下载请执行 download_models.sh
# ============================================================

set -e

# ---- 配置 ----
PROJECT_DIR="/root/SAM3D_Body"
CONDA_ENV_NAME="sam3d_body"
PYTHON_VERSION="3.11"
CUDA_VERSION="cu121"  # 丹炉 Driver 550.54.15, CUDA 12.4, 使用 cu121 兼容

echo "============================================"
echo " SAM3D_Body 丹炉部署脚本"
echo " 项目目录: ${PROJECT_DIR}"
echo " Conda 环境: ${CONDA_ENV_NAME}"
echo " Python: ${PYTHON_VERSION}"
echo " CUDA: ${CUDA_VERSION}"
echo "============================================"

# ---- 1. 检查 Conda ----
echo ""
echo "[1/7] 检查 Conda ..."
if ! command -v conda &> /dev/null; then
    source /root/miniconda3/etc/profile.d/conda.sh
fi

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "⚠️  Conda 环境 '${CONDA_ENV_NAME}' 已存在，跳过创建"
else
    echo "创建 Conda 环境 '${CONDA_ENV_NAME}' ..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${CONDA_ENV_NAME}
echo "当前 Python: $(which python)"

# ---- 2. 创建项目目录 ----
echo ""
echo "[2/7] 创建项目目录 ..."
mkdir -p ${PROJECT_DIR}/checkpoints
mkdir -p ${PROJECT_DIR}/test_images
mkdir -p ${PROJECT_DIR}/output_dinov3
mkdir -p ${PROJECT_DIR}/output_vith
echo "✅ 目录结构已就绪"

# ---- 3. 检查是否已有项目代码 ----
echo ""
echo "[3/7] 检查项目代码 ..."
if [ -f "${PROJECT_DIR}/demo.py" ]; then
    echo "✅ 项目代码已存在，跳过"
else
    echo "⚠️  项目代码未找到！请先从本地上传 Main/ 目录："
    echo "   rsync -aL -e 'ssh -i /tmp/DanLu_key -p 44304' /mnt/d/AI/Prototypes/SAM3D_Body/Main/ root@apps-sl.danlu.netease.com:${PROJECT_DIR}/"
    echo ""
    echo "是否继续安装依赖？(y/n)"
    read -r CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        echo "部署中止。请先上传代码后重新运行。"
        exit 1
    fi
fi

# ---- 4. 安装 PyTorch ----
echo ""
echo "[4/7] 安装 PyTorch ..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo "⚠️  PyTorch 已安装: ${TORCH_VER}，跳过"
else
    echo "安装 PyTorch (CUDA ${CUDA_VERSION}) ..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
fi

# 验证 CUDA
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
"

# ---- 5. 安装 Python 依赖 ----
echo ""
echo "[5/7] 安装 Python 依赖 ..."

# 检查 pip 是否配置了镜像
pip config get global.index-url 2>/dev/null || true

echo "安装核心依赖 ..."
pip install \
    pytorch-lightning \
    pyrender \
    opencv-python \
    yacs \
    scikit-image \
    einops \
    timm \
    dill \
    pandas \
    rich \
    hydra-core \
    hydra-submitit-launcher \
    hydra-colorlog \
    pyrootutils \
    webdataset \
    chump \
    "networkx==3.2.1" \
    roma \
    joblib \
    seaborn \
    wandb \
    appdirs \
    appnope \
    ffmpeg \
    cython \
    jsonlines \
    pytest \
    xtcocotools \
    loguru \
    optree \
    fvcore \
    black \
    pycocotools \
    tensorboard \
    huggingface_hub

# ---- 6. 安装 Detectron2 ----
echo ""
echo "[6/7] 安装 Detectron2 ..."
if python -c "import detectron2" 2>/dev/null; then
    echo "⚠️  Detectron2 已安装，跳过"
else
    echo "安装 Detectron2 (指定 commit a1ce2f9) ..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps || {
        echo "⚠️  Detectron2 从 GitHub 安装失败，可能网络不通"
        echo "   请尝试离线安装："
        echo "   1. 本地: pip download detectron2 -f 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' -d ./offline_pkgs/"
        echo "   2. 上传: scp -r ./offline_pkgs/ root@server:${PROJECT_DIR}/"
        echo "   3. 服务器: pip install --no-index --find-links=${PROJECT_DIR}/ offline_pkgs/ detectron2"
    }
fi

# ---- 7. 安装 MoGe (可选) ----
echo ""
echo "[7/7] 安装 MoGe (可选，FOV 估计用) ..."
if python -c "import moge" 2>/dev/null; then
    echo "⚠️  MoGe 已安装，跳过"
else
    echo "安装 MoGe ..."
    pip install git+https://github.com/microsoft/MoGe.git || {
        echo "⚠️  MoGe 安装失败，可后续手动安装"
        echo "   不影响核心模型推理，仅影响 FOV 估计功能"
    }
fi

# ---- 完成 ----
echo ""
echo "============================================"
echo " ✅ 环境部署完成！"
echo ""
echo " 下一步："
echo "   1. 上传项目代码（如尚未上传）"
echo "   2. 执行 bash ${PROJECT_DIR}/Deploy/download_models.sh 下载模型"
echo "   3. 运行推理测试"
echo ""
echo " 激活环境："
echo "   conda activate ${CONDA_ENV_NAME}"
echo "============================================"

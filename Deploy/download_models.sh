#!/bin/bash
# ============================================================
# SAM3D_Body 模型下载脚本（丹炉服务器）
# 用途：从 HuggingFace 镜像下载两个模型权重
# ============================================================

set -e

PROJECT_DIR="/root/SAM3D_Body"
CHECKPOINTS_DIR="${PROJECT_DIR}/checkpoints"

echo "============================================"
echo " SAM3D_Body 模型下载脚本"
echo " 存储目录: ${CHECKPOINTS_DIR}"
echo "============================================"

# ---- 激活环境 ----
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

# ---- 设置 HuggingFace 镜像 ----
# 丹炉无法直连 HuggingFace，使用国内镜像
# 常用镜像：
#   hf-mirror.com  — 国内常用镜像
#   如果镜像也不通，需要从本地下载后 scp 上传（见 README §3.9）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
echo "HuggingFace 镜像: ${HF_ENDPOINT}"
echo ""

# ---- 检查 huggingface_hub ----
if ! python -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    echo "❌ huggingface_hub 未安装，请先运行 setup_danlu.sh"
    exit 1
fi

# ---- 模型 1: DINOv3 ----
DINOV3_DIR="${CHECKPOINTS_DIR}/sam-3d-body-dinov3"
if [ -f "${DINOV3_DIR}/model.ckpt" ]; then
    echo "✅ DINOv3 模型已存在，跳过下载: ${DINOV3_DIR}"
else
    echo ""
    echo "[1/2] 下载 DINOv3 模型 (jetjodh/sam-3d-body-dinov3, ~2.81 GB) ..."
    mkdir -p ${DINOV3_DIR}
    huggingface-cli download jetjodh/sam-3d-body-dinov3 --local-dir ${DINOV3_DIR} || {
        echo ""
        echo "❌ DINOv3 模型下载失败！"
        echo "   可能原因："
        echo "   1. 镜像不通 — 尝试更换镜像: export HF_ENDPOINT=https://另一个镜像地址"
        echo "   2. 镜像未同步该仓库 — 需从本地下载后 scp 上传"
        echo ""
        echo "   本地下载命令："
        echo "     huggingface-cli download jetjodh/sam-3d-body-dinov3 --local-dir D:\\AI\\Models\\sam-3d-body-dinov3"
        echo ""
        echo "   本地上传命令 (WSL2)："
        echo "     rsync -aL --info=progress2 -e 'ssh -i /tmp/DanLu_key -p 44304' /mnt/d/AI/Models/sam-3d-body-dinov3/ root@apps-sl.danlu.netease.com:${DINOV3_DIR}/"
        exit 1
    }
    echo "✅ DINOv3 模型下载完成"
fi

# ---- 模型 2: ViT-H ----
VITH_DIR="${CHECKPOINTS_DIR}/sam-3d-body-vith"
if [ -f "${VITH_DIR}/model.ckpt" ]; then
    echo "✅ ViT-H 模型已存在，跳过下载: ${VITH_DIR}"
else
    echo ""
    echo "[2/2] 下载 ViT-H 模型 (jetjodh/sam-3d-body-vith, ~2.39 GB) ..."
    mkdir -p ${VITH_DIR}
    huggingface-cli download jetjodh/sam-3d-body-vith --local-dir ${VITH_DIR} || {
        echo ""
        echo "❌ ViT-H 模型下载失败！"
        echo "   同上，请检查镜像或从本地下载后 scp 上传"
        echo ""
        echo "   本地下载命令："
        echo "     huggingface-cli download jetjodh/sam-3d-body-vith --local-dir D:\\AI\\Models\\sam-3d-body-vith"
        echo ""
        echo "   本地上传命令 (WSL2)："
        echo "     rsync -aL --info=progress2 -e 'ssh -i /tmp/DanLu_key -p 44304' /mnt/d/AI/Models/sam-3d-body-vith/ root@apps-sl.danlu.netease.com:${VITH_DIR}/"
        exit 1
    }
    echo "✅ ViT-H 模型下载完成"
fi

# ---- 验证 ----
echo ""
echo "============================================"
echo " 验证模型文件 ..."
echo "============================================"

ERRORS=0

for MODEL_DIR in "${DINOV3_DIR}" "${VITH_DIR}"; do
    MODEL_NAME=$(basename ${MODEL_DIR})
    echo ""
    echo "📦 ${MODEL_NAME}:"
    
    if [ -f "${MODEL_DIR}/model.ckpt" ]; then
        SIZE=$(du -h "${MODEL_DIR}/model.ckpt" | cut -f1)
        echo "  ✅ model.ckpt (${SIZE})"
    else
        echo "  ❌ model.ckpt 缺失"
        ERRORS=$((ERRORS + 1))
    fi
    
    if [ -f "${MODEL_DIR}/model_config.yaml" ]; then
        echo "  ✅ model_config.yaml"
    else
        echo "  ❌ model_config.yaml 缺失"
        ERRORS=$((ERRORS + 1))
    fi
    
    if [ -f "${MODEL_DIR}/assets/mhr_model.pt" ]; then
        SIZE=$(du -h "${MODEL_DIR}/assets/mhr_model.pt" | cut -f1)
        echo "  ✅ assets/mhr_model.pt (${SIZE})"
    else
        echo "  ❌ assets/mhr_model.pt 缺失"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
if [ ${ERRORS} -eq 0 ]; then
    echo "🎉 所有模型文件验证通过！"
    echo ""
    echo " 运行推理："
    echo "   cd ${PROJECT_DIR} && conda activate sam3d_body"
    echo "   python demo.py --image_folder ./test_images --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
else
    echo "⚠️  有 ${ERRORS} 个文件缺失，请检查下载日志"
fi

echo "============================================"

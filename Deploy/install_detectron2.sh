#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

echo "=== 尝试安装 Detectron2 ==="

# 方法1: 尝试从 GitHub 安装（可能因网络不通而失败）
echo "尝试从 GitHub 安装 detectron2 ..."
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 && {
    echo "✅ Detectron2 安装成功"
    python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
    exit 0
} || {
    echo "⚠️  GitHub 安装失败，尝试备选方案 ..."
}

# 方法2: 从 PyPI 安装（版本可能不完全匹配但可工作）
echo "尝试从 PyPI 安装 detectron2 ..."
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.5/index.html 2>&1 && {
    echo "✅ Detectron2 从预编译 wheel 安装成功"
    python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
    exit 0
} || {
    echo "⚠️  预编译 wheel 安装也失败"
}

# 方法3: 尝试 pip 直接安装
echo "尝试 pip install detectron2 ..."
pip install detectron2 2>&1 && {
    echo "✅ Detectron2 安装成功"
    python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"
    exit 0
} || {
    echo "❌ 所有 Detectron2 安装方式均失败"
    echo "   需要手动处理：本地编译后 scp 上传"
    exit 1
}

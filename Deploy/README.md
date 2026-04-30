# SAM3D_Body 丹炉部署指南

> 最后更新：2026-04-30  
> 适用：丹炉服务器 `apps-sl.danlu.netease.com:44304`

---

## 1. 部署概览

| 项目 | 值 |
|------|----|
| 服务器 | `apps-sl.danlu.netease.com:44304` |
| GPU | 2× NVIDIA A30 (24GB VRAM each) |
| 项目目录 | `/root/SAM3D_Body/` |
| Conda 环境 | `sam3d_body`（项目专用，不复用全局） |
| Python | 3.11 |
| 模型存储 | `/root/SAM3D_Body/checkpoints/` |

### 部署的两个模型

| 模型 | HF Repo | model.ckpt 大小 | mhr_model.pt 大小 | 总大小 |
|------|---------|-----------------|-------------------|--------|
| DINOv3 | `jetjodh/sam-3d-body-dinov3` | 2.11 GB | 696 MB | ~2.81 GB |
| ViT-H | `jetjodh/sam-3d-body-vith` | 1.69 GB | ~696 MB | ~2.39 GB |

> ⚠️ 两个模型共享相同的 `assets/mhr_model.pt`，但为保险起见各自保留一份独立副本。

---

## 2. 一键部署

### 2.1 上传部署脚本到服务器

从本地 PowerShell（通过 WSL2）执行：

```powershell
# 准备密钥
wsl -e bash -c "cp /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa /tmp/DanLu_key && chmod 600 /tmp/DanLu_key"

# 上传部署脚本
wsl -e bash -c "scp -i /tmp/DanLu_key -P 44304 /mnt/d/AI/Prototypes/SAM3D_Body/Deploy/setup_danlu.sh root@apps-sl.danlu.netease.com:/root/"
```

### 2.2 上传项目代码

```powershell
# 上传 Main/ 下的核心代码
wsl -e bash -c "rsync -aL --info=progress2 -e 'ssh -i /tmp/DanLu_key -p 44304' /mnt/d/AI/Prototypes/SAM3D_Body/Main/ root@apps-sl.danlu.netease.com:/root/SAM3D_Body/"
```

### 2.3 在服务器上执行部署脚本

```powershell
wsl -e bash -c "ssh -i /tmp/DanLu_key -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com 'bash /root/setup_danlu.sh'"
```

### 2.4 下载模型权重（需要在服务器上交互式执行）

```powershell
wsl -e bash -c "ssh -i /tmp/DanLu_key -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com 'bash /root/SAM3D_Body/Deploy/download_models.sh'"
```

---

## 3. 分步部署详解

如果一键部署出问题，可以按以下步骤逐个执行。

### 3.1 SSH 登录服务器

```bash
ssh -i /tmp/DanLu_key -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com
```

### 3.2 创建项目目录

```bash
mkdir -p /root/SAM3D_Body/checkpoints
```

### 3.3 创建 Conda 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n sam3d_body python=3.11 -y
conda activate sam3d_body
```

### 3.4 安装 PyTorch

```bash
# A30 是 Ampere 架构，支持 CUDA 11.8+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ 如果服务器已有 CUDA 12.x 驱动，可改用 `cu121`:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3.5 安装 Python 依赖

```bash
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub
```

### 3.6 安装 Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

> ⚠️ 丹炉可能无法访问 GitHub。如遇网络问题，参见 §3.8 离线安装方案。

### 3.7 安装 MoGe（可选，FOV 估计用）

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### 3.8 离线安装方案（网络受限时）

如果服务器无法访问 GitHub / PyPI：

```bash
# 在本地机器下载 whl 包，然后 scp 上传
pip download -d ./offline_packages/ detectron2 -f 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9'
scp -i /tmp/DanLu_key -P 44304 -r ./offline_packages/ root@apps-sl.danlu.netease.com:/root/SAM3D_Body/offline_packages/

# 在服务器上离线安装
pip install --no-index --find-links=/root/SAM3D_Body/offline_packages/ detectron2
```

### 3.9 下载模型权重

丹炉无法直连 HuggingFace，需使用国内镜像：

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 DINOv3 模型
huggingface-cli download jetjodh/sam-3d-body-dinov3 --local-dir /root/SAM3D_Body/checkpoints/sam-3d-body-dinov3

# 下载 ViT-H 模型
huggingface-cli download jetjodh/sam-3d-body-vith --local-dir /root/SAM3D_Body/checkpoints/sam-3d-body-vith
```

如果 `hf-mirror.com` 也不通，可从本地下载后 scp 上传：

```powershell
# 本地 Windows 下载（需要有网络的环境）
huggingface-cli download jetjodh/sam-3d-body-dinov3 --local-dir D:\AI\Models\sam-3d-body-dinov3
huggingface-cli download jetjodh/sam-3d-body-vith --local-dir D:\AI\Models\sam-3d-body-vith

# 然后通过 WSL2 上传
wsl -e bash -c "rsync -aL --info=progress2 -e 'ssh -i /tmp/DanLu_key -p 44304' /mnt/d/AI/Models/sam-3d-body-dinov3/ root@apps-sl.danlu.netease.com:/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/"
wsl -e bash -c "rsync -aL --info=progress2 -e 'ssh -i /tmp/DanLu_key -p 44304' /mnt/d/AI/Models/sam-3d-body-vith/ root@apps-sl.danlu.netease.com:/root/SAM3D_Body/checkpoints/sam-3d-body-vith/"
```

---

## 4. 验证部署

### 4.1 验证环境

```bash
conda activate sam3d_body
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
"
```

预期输出：
```
PyTorch: 2.x.x
CUDA available: True
GPU count: 2
  GPU 0: NVIDIA A30, 24.0 GB
  GPU 1: NVIDIA A30, 24.0 GB
```

### 4.2 验证模型加载

```bash
cd /root/SAM3D_Body
conda activate sam3d_body

python -c "
from sam_3d_body import load_sam_3d_body
import torch

# 测试 DINOv3 模型加载
print('Loading DINOv3 model...')
model, cfg = load_sam_3d_body(
    checkpoint_path='/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model.ckpt',
    device='cuda',
    mhr_path='/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
)
print('DINOv3 model loaded successfully!')

# 释放显存
del model
torch.cuda.empty_cache()

# 测试 ViT-H 模型加载
print('Loading ViT-H model...')
model, cfg = load_sam_3d_body(
    checkpoint_path='/root/SAM3D_Body/checkpoints/sam-3d-body-vith/model.ckpt',
    device='cuda',
    mhr_path='/root/SAM3D_Body/checkpoints/sam-3d-body-vith/assets/mhr_model.pt'
)
print('ViT-H model loaded successfully!')
"
```

### 4.3 运行 Demo 推理

```bash
cd /root/SAM3D_Body
conda activate sam3d_body

# 使用 DINOv3 模型
python demo.py \
    --image_folder ./test_images \
    --output_folder ./output_dinov3 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# 使用 ViT-H 模型
python demo.py \
    --image_folder ./test_images \
    --output_folder ./output_vith \
    --checkpoint_path ./checkpoints/sam-3d-body-vith/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-vith/assets/mhr_model.pt
```

### 4.4 上传测试图片

从本地上传测试图片到服务器：

```powershell
wsl -e bash -c "scp -i /tmp/DanLu_key -P 44304 /mnt/d/path/to/test_image.jpg root@apps-sl.danlu.netease.com:/root/SAM3D_Body/test_images/"
```

### 4.5 下载推理结果

```powershell
wsl -e bash -c "scp -i /tmp/DanLu_key -P 44304 -r root@apps-sl.danlu.netease.com:/root/SAM3D_Body/output_dinov3/ /mnt/d/AI/Prototypes/SAM3D_Body/outputs/"
```

---

## 5. 完整推理管线（含检测器/分割器/FOV）

### 5.1 仅核心模型推理（最简模式，无需额外模型）

```bash
python demo.py \
    --image_folder ./test_images \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --detector_name "" \
    --segmentor_name "" \
    --fov_name ""
```

> 此模式下：无人体检测（使用全图 bbox）、无分割 mask、无 FOV 估计（使用默认 FOV）。

### 5.2 含 ViTDet 检测器推理

ViTDet-H 的权重会自动从 `dl.fbaipublicfiles.com` 下载。如网络不通：

```bash
# 在本地下载 ViTDet 权重（约 2.5GB）
# URL: https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl

# 上传到服务器
wsl -e bash -c "scp -i /tmp/DanLu_key -P 44304 /mnt/d/path/to/model_final_f05665.pkl root@apps-sl.danlu.netease.com:/root/SAM3D_Body/checkpoints/vitdet/"

# 使用本地权重运行
python demo.py \
    --image_folder ./test_images \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --detector_name vitdet \
    --detector_path ./checkpoints/vitdet
```

### 5.3 含 SAM2.1 分割器推理

```bash
# 需要预先部署 SAM2.1
git clone https://github.com/facebookresearch/sam2.git /root/SAM3D_Body/external/sam2
cd /root/SAM3D_Body/external/sam2 && pip install -e .

# 下载 SAM2.1 checkpoint
# 需从本地下载后上传: sam2.1_hiera_large.pt (~900MB)

python demo.py \
    --image_folder ./test_images \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --detector_name vitdet \
    --segmentor_name sam2 \
    --segmentor_path /root/SAM3D_Body/external/sam2
```

### 5.4 含 MoGe2 FOV 估计器推理

```bash
pip install git+https://github.com/microsoft/MoGe.git

# MoGe2 权重会自动从 HuggingFace 下载 (Ruicheng/moge-2-vitl-normal)
# 丹炉需设置镜像：export HF_ENDPOINT=https://hf-mirror.com
# 或从本地下载后上传

python demo.py \
    --image_folder ./test_images \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --fov_name moge2 \
    --fov_path /root/SAM3D_Body/checkpoints/moge2
```

---

## 6. 显存预估

| 组件 | 显存占用 |
|------|---------|
| SAM3D-Body DINOv3 (推理) | ~6-8 GB |
| SAM3D-Body ViT-H (推理) | ~5-7 GB |
| ViTDet-H 检测器 | ~4-5 GB |
| SAM2.1 Hiera Large | ~2-3 GB |
| MoGe2 | ~1-2 GB |
| **完整管线（DINOv3）** | **~14-18 GB** |
| **完整管线（ViT-H）** | **~13-16 GB** |

> A30 有 24GB VRAM，完整管线可单卡运行。如果同时加载检测器+分割器+FOV+主模型，建议只用 1 张卡避免多卡冲突。

---

## 7. 服务器目录结构

```
/root/SAM3D_Body/
├── demo.py
├── sam_3d_body/              # 核心代码包
│   ├── __init__.py
│   ├── build_models.py
│   ├── sam_3d_body_estimator.py
│   ├── data/
│   ├── metadata/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tools/
│   ├── build_detector.py
│   ├── build_sam.py
│   ├── build_fov_estimator.py
│   └── cascade_mask_rcnn_vitdet_h_75ep.py
├── checkpoints/
│   ├── sam-3d-body-dinov3/
│   │   ├── model.ckpt              # 2.11 GB
│   │   ├── model_config.yaml
│   │   └── assets/
│   │       └── mhr_model.pt        # 696 MB
│   ├── sam-3d-body-vith/
│   │   ├── model.ckpt              # 1.69 GB
│   │   ├── model_config.yaml
│   │   └── assets/
│   │       └── mhr_model.pt        # ~696 MB
│   ├── vitdet/
│   │   └── model_final_f05665.pkl  # ~2.5 GB (可选)
│   └── moge2/                       # (可选)
├── external/                         # (可选)
│   └── sam2/
├── test_images/                      # 测试输入
├── output_dinov3/                    # DINOv3 推理输出
├── output_vith/                      # ViT-H 推理输出
└── Deploy/                           # 部署脚本
    ├── setup_danlu.sh
    └── download_models.sh
```

---

## 8. 常见问题

### Q: `huggingface-cli download` 失败
**A**: 丹炉无法直连 HuggingFace，设置镜像环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
如果镜像也不通，从本地下载后 scp 上传（见 §3.9）。

### Q: `pip install detectron2` 编译失败
**A**: 需要 GCC 和 CUDA 开发头文件。确认：
```bash
gcc --version          # 需要 >= 7
nvcc --version         # 需要 CUDA toolkit
```
如果缺少编译工具，可从本地编译好 whl 后上传安装。

### Q: `pyrender` 报错
**A**: 服务器通常无显示器，需设置离屏渲染：
```bash
export PYOPENGL_PLATFORM=osmesa
# 或
export PYOPENGL_PLATFORM=egl
```
可能还需安装 `sudo apt-get install libosmesa6-dev` 或 `libegl1-mesa-dev`。

### Q: 显存不足 OOM
**A**: 
1. 只加载核心模型，不加载检测器/分割器/FOV（见 §5.1）
2. 使用 `--fov_name ""` 跳过 FOV 估计
3. 确保没有其他进程占用 GPU：`nvidia-smi`

### Q: 误装包到全局环境
**A**: 确保每次操作前都 `conda activate sam3d_body`，验证：
```bash
which python  # 应该指向 /root/miniconda3/envs/sam3d_body/bin/python
```

# 丹炉（DanLu）开发基础指南

> 最后更新：2026-04-30  
> 适用项目：Inspatio WorldFM / SAM3D_Body

---

## 1. 服务器基本信息

| 项目 | 值 |
|------|----|
| 主机地址 | `apps-sl.danlu.netease.com` |
| SSH 端口 | `44304` |
| 登录用户 | `root` |
| GPU | 2× NVIDIA A30 (24GB VRAM each) |
| Conda 环境 | `inspatio_world` / `sam3d_body`（项目专用，不复用全局） |
| Python | 需要根据当前仓库需求自行部署虚拟环境，不要复用全局环境，以免搞乱版本 |

---

## 2. SSH 私钥

**Windows 路径**：`D:\AI\PrivateKeys\DanLu\xuxiao02_rsa`  
**WSL2 路径**：`/mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa`

> ⚠️ **重要**：WSL2 下使用 SSH 私钥时，**必须先确保权限为 600**，否则 SSH 会拒绝使用该密钥：
> ```bash
> chmod 600 /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa
> ```

---

## 3. 如何登录

### 3.1 从 Windows PowerShell 直接登录

```powershell
ssh -i "D:\AI\PrivateKeys\DanLu\xuxiao02_rsa" -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com
```

### 3.2 从 WSL2 登录（推荐，权限处理更简便）

```bash
ssh -i /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com
```

### 3.3 脚本中的通用写法（推荐模式）

将密钥临时复制到 `/tmp/DanLu_key` 再使用，避免路径权限问题：

```bash
cp /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa /tmp/DanLu_key
chmod 600 /tmp/DanLu_key
ssh -i /tmp/DanLu_key -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com
```

### 3.4 从 PowerShell 调起 WSL2 执行 SSH

```powershell
wsl -e bash -c "ssh -i /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa -p 44304 -o StrictHostKeyChecking=no root@apps-sl.danlu.netease.com"
```

---

## 4. 上传资产（SCP）

上传操作**必须在 WSL2 bash 环境**中进行（Windows 的 `scp` 不支持 `.pem`/`_rsa` 私钥格式的权限要求）。

### 4.1 上传单个文件

```bash
# 先准备密钥
cp /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa /tmp/DanLu_key
chmod 600 /tmp/DanLu_key

# 上传文件
scp -i /tmp/DanLu_key -P 44304 \
    /mnt/d/AI/Inspatio_WorldFM/gen_multiview_meta.py \
    root@apps-sl.danlu.netease.com:/root/Inspatio_WorldFM/gen_multiview_meta.py
```

### 4.2 上传整个目录（递归）

```bash
scp -i /tmp/DanLu_key -P 44304 -r \
    /mnt/d/AI/Inspatio_WorldFM/modules/ \
    root@apps-sl.danlu.netease.com:/root/Inspatio_WorldFM/modules/
```

### 4.3 上传输入图片

```bash
scp -i /tmp/DanLu_key -P 44304 \
    "/mnt/d/AI/Inspatio_WorldFM/Data/Test3/Pixel_Village.png" \
    "root@apps-sl.danlu.netease.com:/root/plan_c/Test3/"
```

### 4.4 使用 rsync（推荐用于大目录，支持断点续传）

rsync 还能正确处理符号链接（`-L` 解引用符号链接，避免上传的是空链接）：

```bash
rsync -aL --info=progress2 \
    -e "ssh -i /tmp/DanLu_key -p 44304" \
    /mnt/d/AI/Inspatio_WorldFM/modules/ \
    root@apps-sl.danlu.netease.com:/root/Inspatio_WorldFM/modules/
```

---

## 5. 下载输出结果（SCP）

```bash
# 下载单个输出文件
scp -i /tmp/DanLu_key -P 44304 \
    root@apps-sl.danlu.netease.com:/root/plan_c/Village/focused_quarter/worldmirror_output/gaussians.ply \
    /mnt/d/AI/Inspatio_WorldFM/outputs/

# 下载整个输出目录
scp -i /tmp/DanLu_key -P 44304 -r \
    root@apps-sl.danlu.netease.com:/root/plan_c/Village/ \
    /mnt/d/AI/Inspatio_WorldFM/outputs/
```

---

## 6. 在服务器上执行命令（非交互式）

适合在脚本中批量运行服务器命令，不需要手动登录：

```bash
ssh -i /tmp/DanLu_key -p 44304 -o StrictHostKeyChecking=no \
    root@apps-sl.danlu.netease.com << 'ENDSSH'

# 以下命令在服务器上执行
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inspatio_world

cd /root/Inspatio_WorldFM
python gen_multiview_meta.py --help

ENDSSH
```

---

## 7. 服务器关键目录

| 目录 | 说明 |
|------|------|
| `/root/plan_c/` | Plan C 测试数据和结果 |
| `/root/SAM3D_Body/` | SAM3D_Body 项目目录（代码 + checkpoints） |
| `/data/miniconda3/envs/inspatio_world/` | Inspatio WorldFM Python 环境 |
| `/data/miniconda3/envs/sam3d_body/` | SAM3D_Body Python 环境 |
| `/root/.cache/huggingface/` | HF 模型缓存（~34GB，含大量重复） |

---

## 8. 激活 Python 环境

### Inspatio WorldFM

登录服务器后：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inspatio_world

# 验证环境
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

### SAM3D_Body

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body

# 验证环境
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

---

## 9. 运行管线

### 完整 Plan C 管线（WorldFM → WorldMirror）

```bash
cd /root/Inspatio_WorldFM
conda activate inspatio_world

bash run_plan_c.sh /root/plan_c/Test3/Pixel_Village.png Village focused_quarter
```

### 仅运行 WorldFM

```bash
cd /root/Inspatio_WorldFM
conda activate inspatio_world

python run_pipeline.py --config a30_config.yaml --input /path/to/image.png --output /root/plan_c/output
```

### 仅运行 WorldMirror

```bash
cd /data/hy-world-2.0   # 必须在此目录（skyseg.onnx 相对路径）

torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
    --input_path <images_dir> \
    --prior_cam_path <cameras.json> \
    --strict_output_path <output_dir> \
    --pretrained_model_name_or_path /root/Inspatio_WorldFM/checkpoints/HY-WorldMirror-2.0 \
    --target_size 832 --use_fsdp --enable_bf16 \
    --apply_confidence_mask --apply_edge_mask --compress_pts --no_save_normal --no_interactive
```

> ⚠️ `--pretrained_model_name_or_path` 必须写本地路径，服务器无法访问 HuggingFace。

### SAM3D_Body 推理

```bash
cd /root/SAM3D_Body
conda activate sam3d_body

# DINOv3 模型（推荐，精度更高）
python demo.py \
    --image_folder ./test_images \
    --output_folder ./output_dinov3 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# ViT-H 模型
python demo.py \
    --image_folder ./test_images \
    --output_folder ./output_vith \
    --checkpoint_path ./checkpoints/sam-3d-body-vith/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-vith/assets/mhr_model.pt
```

> 详细部署步骤见 `Deploy/README.md`

---

## 10. 常用检查命令

```bash
# GPU 状态
nvidia-smi

# 磁盘使用
df -h /root /data

# 进程监控
watch -n 2 nvidia-smi

# 查看日志（管线输出）
tail -f /root/plan_c/Village/pipeline.log

# 查看输出内容
ls -lh /root/plan_c/Village/focused_quarter/worldmirror_output/
```

---

## 11. 注意事项

- **服务器无法访问 HuggingFace**，所有模型必须使用本地路径；下载模型时需设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`
- SCP 的 `-P`（大写）是指定端口，SSH 的 `-p`（小写）是指定端口
- 每次脚本运行完毕后可以 `rm -f /tmp/DanLu_key` 清理临时密钥
- **项目环境必须自包含**，每个项目使用独立 conda 环境，不搞乱全局环境

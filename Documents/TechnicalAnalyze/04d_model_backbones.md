# 4d. 核心模型深度分析 — Backbone & 位置编码

> 对应代码：
> - `models/backbones/vit.py` — ViT 系列实现
> - `models/backbones/dinov3.py` — DINOv3 backbone

## 4d.1 ViT 架构总览

代码提供了 5 种 ViT 变体：

| 变体 | embed_dim | depth | num_heads | patch_size | 图像尺寸 | 用途 |
|------|-----------|-------|-----------|------------|---------|------|
| `vit_h` | 1280 | 32 | 16 | 16 | 1024×1024 | 默认主 backbone |
| `vit_l` | 1024 | 24 | 16 | 16 | 1024×1024 | 轻量选项 |
| `vit_b` | 768 | 12 | 12 | 16 | 1024×1024 | 最轻量 |
| `vit256` | 1280 | 32 | 16 | 16 | 518×518 | 高分辨率输入 |
| `vit512_384` | 1280 | 32 | 16 | 16 | 512×384 | 宽屏输入 |

所有变体共享相同的 `ViT` 类，通过 `factory` 函数配置。

---

## 4d.2 PatchEmbed

```python
class PatchEmbed(nn.Module):
    # 卷积 patch 嵌入
    self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    # 特殊 padding: 4 + 2*(ratio//2 - 1)，ratio = img_size / patch_size
    self.padding = (pad_h, pad_w)
```

**Padding 策略**：
- 基础 padding = 4
- 根据 `ratio = img_size // patch_size` 调整
- 对于 1024/16 = 64: padding = 4 + 2*(64//2-1) = 66
- 确保输出 feature map 尺寸精确对齐

**输出**：`[B, embed_dim, H/ratio, W/ratio]`（经过 padding 后的实际尺寸）

---

## 4d.3 位置编码

### 绝对位置编码插值 `get_abs_pos()`

```python
def get_abs_pos(abs_pos, h, w):
    # 原始网格尺寸
    orig_size = int((abs_pos.shape[0]-1) ** 0.5)  # -1 排除 CLS token
    # 2D 网格插值
    pos = abs_pos.reshape(-1, orig_size, orig_size, dim)
    pos = F.interpolate(pos, size=(h, w), mode='bicubic')
    return pos.reshape(-1, dim)
```

**处理方式**：
1. 将 1D 位置编码重塑为 2D 网格（排除 CLS token）
2. 双三次插值到目标分辨率
3. 重塑回 1D

### `ViT.forward_features()`

```python
# 拼接位置编码
x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
# 如果有 extra_embed (CLIFF 条件等):
x = x + self.extra_embed
```

注意：将 CLS 位置编码广播到所有 patch（`pos_embed[:, :1]`），其余按空间顺序相加。

---

## 4d.4 Transformer Block

```python
class Block(nn.Module):
    # 标准 ViT Block
    self.norm1 = nn.LayerNorm(dim)
    self.attn = Attention(dim, num_heads)  # 或 FlashAttention
    self.norm2 = nn.LayerNorm(dim)
    self.mlp = Mlp(dim, mlp_ratio)
    self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
```

### Flash Attention

```python
class FlashAttention(nn.Module):
    def forward(self, x):
        qkv = self.qkv(x)  # [B, N, 3*C]
        q, k, v = rearrange(qkv, 'B N (three H D) -> three B H N D', three=3, H=self.num_heads)
        x = flash_attn_func(q, k, v, dropout_p=self.dropout_p)  # Flash Attention
        x = self.proj(x)
        return x
```

**条件启用**：如果 `flash_attn_func` 可用且输入在 CUDA 上，使用 Flash Attention；否则回退到标准 `nn.MultiheadAttention`。

---

## 4d.5 冻结策略 `_freeze_stages()`

```python
def _freeze_stages(self):
    if self.frozen_stages >= 0:
        self.patch_embed.requires_grad_(False)
    if self.frozen_stages >= 1:
        self.pos_embed.requires_grad_(False)
        self.extra_embed.requires_grad_(False)
        for i in range(self.frozen_stages):
            self.blocks[i].requires_grad_(False)
```

**典型配置**：`frozen_stages=24` — 冻结前 24 层（ViT-H 的一半），仅微调后 8 层 + head。

**checkpoint 策略**：使用 `checkpoint_seq` 对未冻结的 block 进行梯度检查点，降低显存占用。

---

## 4d.6 DINOv3 Backbone

`dinov3.py` 封装了 Meta 的 DINOv3 预训练模型：

```python
class DINOv3(nn.Module):
    # 基于 ViT-giant2 (1.1B 参数)
    # embed_dim=1536, depth=40, num_heads=24
    # patch_size=14
    # 默认图像尺寸: 518×518
```

**特点**：
- 更大的模型容量（1536 维 vs ViT-H 的 1280 维）
- patch_size=14 (而非 16)，提供更细粒度的 patch
- register tokens 支持
- 使用 `torch.hub.load('facebookresearch/dinov2')` 加载预训练权重

**在 SAM3DBody 中的角色**：作为可选 backbone，提供更强的视觉特征。但默认使用 ViT-H (SAM 预训练权重)。

---

## 4d.7 HybridEmbed

```python
class HybridEmbed(nn.Module):
    # 用于非标准输入（如早期 CNN 特征 → ViT）
    self.proj = nn.Linear(conv_dim, embed_dim)
    # 可选 cls_token
```

**在当前代码中未使用**，预留用于未来混合架构。

---

## 4d.8 输出格式

```python
# ViT.forward_features()
x = self.patch_embed(x)          # [B, D, Hp, Wp]
x = x.flatten(2).transpose(1,2) # [B, Hp*Wp, D]
x = x + pos_embed + extra_embed
for blk in self.blocks:
    x = blk(x)
x = self.last_norm(x)            # [B, Hp*Wp, D]
x = x.reshape(B, Hp, Wp, -1).permute(0,3,1,2)  # [B, D, Hp, Wp]
```

**输出**：4D feature map `[B, embed_dim, H/patch_size, W/patch_size]`

对于 ViT-H + 1024×1024 输入：`[B, 1280, 64, 64]`

---

## 4d.9 与 Fast 版本的关系

### Backbone 替换

Fast 版本的第一项加速方法是用 **YOLO11-Pose** 替代 ViTDet + ViT-H 的检测→分割→编码管线：

```
原始管线: ViTDet-H → SAM2.1 → FOV Estimator → ViT-H Encoder → Decoder
Fast 管线: YOLO11-Pose → (bbox + 2D kps + segmentation) → FOV Estimator → (skip encoder) → Decoder
```

**关键差异**：
1. **YOLO11-Pose 直接提供 2D 关键点**：不再需要 Keypoint Prompt Sampler
2. **空间先验解耦**：2D 检测和关键点估计在 YOLO 中一次完成
3. **跳过 ViT encoder**：YOLO 的 CNN backbone 特征不用于 3D 重建

### 可复用的组件

- `PatchEmbed` + `get_abs_pos`：可用于任何需要 ViT 特征的场景
- 冻结策略：`frozen_stages` 机制可以继续用于微调
- Flash Attention：推理时加速关键

### 不可复用的组件

- ViT-H 的 32 层 Transformer：Fast 版本不再使用此 backbone
- SAM2.1 分割：被 YOLO 的分割头替代

### 推理时加速路径

对于 Fast 版本，如果仍需 ViT 特征（如更高质量的 3D 重建），可以使用：
- **torch.compile**：将 ViT 编译为静态图
- **TensorRT**：将 ViT 导出为 TRT engine
- **量化**：INT8/FP8 量化 ViT 权重

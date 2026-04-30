# 4e. 核心模型深度分析 — 辅助模块

> 对应代码：
> - `models/modules/camera_embed.py` — CameraEncoder
> - `models/decoders/prompt_encoder.py` — PromptEncoder, KeypointSamplerV1
> - `models/modules/transformer.py` — Transformer 基础组件
> - `models/modules/geometry_utils.py` — 几何工具
> - `models/modules/mhr_utils.py` — MHR 工具函数

## 4e.1 CameraEncoder

### 架构

```python
class CameraEncoder(nn.Module):
    self.fourier = FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)
    # 输出: 99 维 (1 DC + 2×16×3 bands)
    self.conv1x1 = nn.Conv2d(embed_dim + 99, embed_dim, 1)
    self.norm = LayerNorm2d(embed_dim)
```

### FourierPositionEncoding

```
输入: rays [B, 3, H_r, W_r]  (方向向量, 归一化到 [-1,1])

编码:
  DC 项: 1 维 (常数 1)
  频带: 16 个频率 × sin/cos × 3 坐标 = 96 维
  
  频率 = 2^k × π, k = 0, 1, ..., 15
  
输出: [B, 99, H_r, W_r]
```

### 前向流程

```python
def forward(self, rays, features):
    # 1. 射线方向 Fourier 编码
    ray_feat = self.fourier(rays)          # [B, 99, H_r, W_r]
    
    # 2. 拼接到视觉特征
    feat = cat([features, ray_feat], dim=1)  # [B, D+99, H_r, W_r]
    
    # 3. 1×1 卷积融合
    feat = self.conv1x1(feat)               # [B, D, H_r, W_r]
    feat = self.norm(feat)                  # LayerNorm2d
    
    return feat
```

### 与 SAM3DBody 的集成

```python
# sam3d_body.py 中的调用:
ray_condition = self.get_ray_condition(cam_int, bbox_center, bbox_size, img_size, patch_size)
ray_features = self.ray_cond_emb(ray_condition, image_features)  # CameraEncoder
# ray_features 用于 decoder 的 cross-attention memory
```

**`get_ray_condition()`**：
```
1. 根据 cam_int 计算每个像素的射线方向
2. 裁剪到 bbox 区域
3. 下采样到 feature map 分辨率 (H/patch_size × W/patch_size)
4. 归一化到 [-1, 1]
```

---

## 4e.2 PromptEncoder

### 架构

```python
class PromptEncoder(nn.Module):
    self.pe = PositionEmbeddingRandom(num_pos_feats=embed_dim//2)
    # 随机 Fourier 位置编码
    
    self.point_embeddings = nn.ModuleList([
        nn.Embedding(1, embed_dim),  # 正样本点
        nn.Embedding(1, embed_dim),  # 负样本点
    ])
    
    self.per_joint_embeddings = nn.ModuleList([
        nn.Embedding(1, embed_dim) for _ in range(70)  # 70 个关节
    ])
    
    self.not_a_point_embed = nn.Embedding(1, embed_dim)
    self.mask_downscaling_v1 = nn.Sequential(...)  # v1 mask 下采样
    self.mask_downscaling_v2 = nn.Sequential(...)  # v2 mask 下采样
```

### 关键点 Prompt 编码

```python
def _embed_points(self, points, labels, pe_type):
    # 1. 位置编码
    point_embedding = self.pe(points)  # [B, N, D]
    
    # 2. 类型 embedding
    if pe_type == "keypoint":
        # 使用关节特定 embedding
        point_embedding += self.per_joint_embeddings[label](label)
    else:
        # 使用正/负样本 embedding
        point_embedding += self.point_embeddings[label](label)
    
    return point_embedding
```

### PositionEmbeddingRandom

```python
class PositionEmbeddingRandom(nn.Module):
    self.positional_encoding_gaussian_matrix = nn.Parameter(
        torch.randn(2, embed_dim//2) * 0.02, requires_grad=False
    )
    
    def forward(self, coords):
        # coords: [B, N, 2], 归一化到 [0,1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        # sin/cos 编码
        return cat([sin(2π×coords), cos(2π×coords)], dim=-1)  # [B, N, D]
```

### Mask 下采样

```
v1: Conv2d(1, embed_dim//4, 2, 2) → LayerNorm → Conv2d(//4, embed_dim, 2, 2) → GELU → Conv2d(embed_dim, embed_dim, 1)
v2: Conv2d(1, embed_dim//4, 2, 2) → LayerNorm → GELU → Conv2d(//4, embed_dim, 1) → LayerNorm
```

---

## 4e.3 KeypointSamplerV1

训练时迭代式关键点采样器：

```python
class KeypointSamplerV1:
    keybody_ratio: float = 0.8   # 80% 关键体部关节
    worst_ratio: float = 0.8     # 80% 最差关节
    
    def sample_keypoints(self, pred_kp2d, gt_kp2d, conf, kp_valid):
        # 1. 选择 keybody 关节 (80%)
        keybody_kps = select_keybody_joints(kp_valid, self.keybody_ratio)
        
        # 2. 从最差关节中选择 (80% worst)
        errors = compute_2d_error(pred_kp2d, gt_kp2d)
        worst_kps = select_worst_joints(errors, self.worst_ratio)
        
        # 3. 合并
        sampled_kps = cat([keybody_kps, worst_kps])
        
        return sampled_kps
```

**训练流程**：
1. 初始预测 → 计算误差 → 采样最差关节
2. 将采样关节作为 prompt 反馈 → 下一轮预测
3. 迭代 2-3 轮

**推理时不使用** — 推理时使用所有 70 个关节的 embedding。

---

## 4e.4 Transformer 基础组件

### FFN (Feed-Forward Network with Identity Connection)

```python
class FFN(nn.Module):
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.ls = LayerScale(dim, init_values) if layer_scale else nn.Identity()
    self.drop_path = DropPath(drop_path)
    # Identity connection: output = x + drop_path(ls(fc2(act(fc1(x)))))
```

### MLP (简单多层感知机)

```python
class MLP(nn.Module):
    # 3 层 MLP (标准 SAM 结构)
    layers = [Linear(input_dim, hidden_dim), ReLU, Linear(hidden_dim, hidden_dim), ReLU, Linear(hidden_dim, output_dim)]
```

### MultiheadAttention (自注意力)

```python
class MultiheadAttention(nn.Module):
    # 标准 MHA
    self.qkv = nn.Linear(dim, dim*3)
    self.proj = nn.Linear(dim, dim)
    # 使用 F.scaled_dot_product_attention (PyTorch 2.x)
```

### Attention (交叉注意力)

```python
class Attention(nn.Module):
    # 独立的 Q/K/V 投影
    self.q_proj = nn.Linear(dim, dim)
    self.k_proj = nn.Linear(dim, dim_kv)
    self.v_proj = nn.Linear(dim, dim_kv)
    self.proj = nn.Linear(dim_kv, dim)
```

### TransformerDecoderLayer

```python
class TransformerDecoderLayer(nn.Module):
    # 自注意力 + 交叉注意力 + FFN
    self.self_attn = MultiheadAttention(dim, num_heads)
    self.cross_attn = Attention(dim, dim_kv, num_heads)
    self.ffn = FFN(dim, hidden_dim)
    self.norm1 = LayerNorm32(dim)  # Pre-norm
    self.norm2 = LayerNorm32(dim)
    self.norm3 = LayerNorm32(dim)
    
    # 可选 two-way attention
    self.two_way_attn = Attention(dim, dim, num_heads)  # memory→query 反馈
```

---

## 4e.5 geometry_utils

### `rot6d_to_rotmat()`

```
6D rotation → 3×3 rotation matrix
  输入: [*, 6] → 输出: [*, 3, 3]
  
  col1 = normalize(d[:, :3])
  col2 = normalize(d[:, 3:6] - (col1·d[:,3:6])·col1)  # Gram-Schmidt
  col3 = cross(col1, col2)
```

### `aa_to_rotmat()`

```
axis-angle → rotation matrix (Rodrigues formula)
```

### `perspective_projection()`

```
标准透视投影:
  j2d = (focal_x × X / Z + cx, focal_y × Y / Z + cy)
  
输入:
  points: [B, N, 3]
  cam_int: [B, 3, 3]
  
输出: [B, N, 2]
```

### `cam_crop_to_full()`

```
裁剪坐标 → 全图坐标
  cam_t_full = cam_t_crop + [2(bbox_cx - img_cx)/f, 2(bbox_cy - img_cy)/f, 0]
```

---

## 4e.6 mhr_utils

### `fix_wrist_euler()`

```
腕关节 Euler 角约束 (IK 后):
  X ∈ [-2.2, 1.0]
  Z ∈ [-2.2, 1.5]
  Y ∈ [-1.2, 1.5]
  
  clamp each component to range
```

### `rotation_angle_difference()`

```
两个旋转矩阵之间的角度差:
  diff = R1 @ R2.T
  angle = arccos((trace(diff) - 1) / 2)
```

### `compact_cont_to_model_params_body()`

```
260维连续表示 → 130维 Euler 角:
  for each pair (sin_val, cos_val):
    θ = atan2(sin_val, cos_val)
```

### `compact_cont_to_model_params_hand()`

```
54维连续表示 → 27维 Euler 角
```

### `compact_model_params_to_cont_body()`

```
130维 Euler 角 → 260维连续表示:
  for each angle θ:
    (sin(θ), cos(θ))
```

### `batch6DFromXYZ()`

```
3D 位置 → 6D 旋转估计:
  用于从 3D 关键点位置估计初始旋转
```

### `mhr_param_hand_mask`

```
布尔掩码，标记 body_pose_params 中哪些属于手部关节
  用于 zero-out 手部参数（body forward 中）
```

---

## 4e.7 辅助模块

### `drop_path.py` — DropPath (Stochastic Depth)

```python
class DropPath(nn.Module):
    # 训练时随机丢弃整条路径，推理时不变
    # 用于 Transformer block 的正则化
```

### `layer_scale.py` — LayerScale

```python
class LayerScale(nn.Module):
    self.weight = nn.Parameter(init_values * torch.ones(dim))
    # output = weight * x  (可学习的逐通道缩放)
```

### `swiglu_ffn.py` — SwiGLU FFN

```python
class SwiGLUFFN(nn.Module):
    # Swish-Gated Linear Unit FFN
    # 用于 DINOv3 backbone
    self.w1 = nn.Linear(dim, hidden_dim)
    self.w2 = nn.Linear(hidden_dim, dim)
    self.w3 = nn.Linear(dim, hidden_dim)  # gate
    # output = w2(silu(w1(x)) * w3(x))
```

---

## 4e.8 与 Fast 版本的关系

### 可直接复用

1. **CameraEncoder** + **FourierPositionEncoding**: Fast 版本仍需要相机条件编码
2. **geometry_utils**: 所有几何变换函数通用
3. **mhr_utils**: MHR 工具函数在新管线中仍需要
4. **Transformer 基础组件**: FFN, MLP, LayerNorm 等是通用组件

### 需要修改

1. **PromptEncoder**: Fast 版本去掉迭代 prompting，可以简化
   - 移除 `per_joint_embeddings`（YOLO 直接提供关键点）
   - 移除 mask downscaling
   - 保留基本的点/位置编码

2. **KeypointSamplerV1**: 训练时不再需要，Fast 版本用 YOLO 替代

3. **Attention 模块**: Fast 的 Decoder Pruning 可能需要修改注意力计算
   - 支持层选择集合 S 的稀疏计算
   - 支持禁用 keypoint token cross-attention

### 需要新增

1. **MLP 投影器** (Neural Kinematic Projection):
   - 3 层 MLP: 4500 → 512 → 256 → 76
   - 输入: 子采样 MHR 顶点 (1,500/6,890)
   - 输出: SMPL 参数 (76 维)
   
2. **运动学先验 MLP**:
   - 输入: SMPL body pose (63 维)
   - 输出: 去噪后的 body pose
   - ~0.1ms 延迟

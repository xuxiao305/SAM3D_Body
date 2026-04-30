# 2. 推理管线架构分析 — 原始 SAM 3D Body 管线

> 对应论文 §3.1 Preliminaries: SAM 3D Body (3DB) Pipeline

## 2.1 原始 SAM 3D Body 管线总览

原始 3DB 管线是一个多阶段串行流程，从单张 RGB 图像出发，经过检测、编码、解码、合并和转换，最终输出全身 3D 人体网格。

```
输入图像 I ∈ R^{H×W×3}
    │
    ▼
[1] 人体检测 → {b_i} body bounding boxes
    │
    ▼
[2] FOV 估计 → K ∈ R^{3×3} 相机内参
    │
    ▼
[3] Body 裁剪 + 编码 → F_body = Enc(Crop(I, b_i)) ∈ R^{h×w×D}
    │   (可选: Ray Conditioning, Mask Conditioning)
    ▼
[4] Body Decoder → ̂θ_body (MHR 参数) + 手部 bbox
    │   (交叉注意力 + 中间预测 + 关键点 Token 更新)
    ▼
[5] 手部检测 → b_L, b_R (从 body decoder 输出推导)
    │
    ▼
[6] Hand 裁剪 + 编码 (×2, 串行)
    │   F_L = Enc(Crop(I, b_L))   ← CPU 裁剪
    │   F_R = Enc(Crop(I, b_R))   ← CPU 裁剪
    ▼
[7] Hand Decoder (×2, 串行)
    │   ̂θ_L = HandDec(F_L)
    │   ̂θ_R = HandDec(F_R)
    ▼
[8] 关键点提示精调 (可选)
    │   用手腕+肘部关键点重新运行 Body Decoder
    ▼
[9] 手-体合并 + 腕部 IK
    │   替换手部姿态/尺度/形状 + fix_wrist_euler()
    ▼
[10] MHR 前向运动学 → vertices, keypoints, joint_coords
    │
    ▼
[11] MHR→SMPL 迭代转换 (数百步优化) ← 论文 Eq.(1)
    │
    ▼
输出: SMPL 参数 ̂Θ_smpl
```

---

## 2.1.1 人体检测与分割

### 代码位置
- **检测器**：`tools/build_detector.py` → `HumanDetector`
- **分割器**：`tools/build_sam.py` → `HumanSegmentor`

### 检测器实现

`HumanDetector` 支持两种后端：

| 后端 | 实现 | 模型 | 特点 |
|------|------|------|------|
| `vitdet` | Detectron2 Cascade Mask R-CNN ViTDet-H | `cascade_mask_rcnn_vitdet_h_75ep.py` | 原始 3DB 默认检测器，仅输出 body bbox |
| `sam3` | SAM3 text-prompted detection | `sam3.model_builder` | 支持 "person" 文本提示，输出 body bbox |

**ViTDet-H 流程**：
```python
# load_detectron2_vitdet()
cfg = LazyConfig.load("cascade_mask_rcnn_vitdet_h_75ep.py")
cfg.train.init_checkpoint = "model_final_f05665.pkl"
detector = instantiate(cfg.model)
# 阈值设为 0.25
```

**关键点**：ViTDet-H 仅检测人体 bounding box，不提供手部 bbox。手部 bbox 需要等 body decoder 运行后从输出中推导（这是原始管线的串行瓶颈）。

### 分割器实现

`HumanSegmentor` 支持两种后端：

| 后端 | 实现 | 模型 |
|------|------|------|
| `sam2` | SAM2.1 Hiera Large | `sam2.1_hiera_large.pt` |
| `sam3` | SAM3 text-prompted | `sam3.model_builder` |

**SAM2 流程**：bbox → `sam_predictor.predict(box=boxes[[i]])` → mask + score
**SAM3 流程**：text prompt "person" → masks + boxes + scores

分割掩码作为可选的条件输入传递给 decoder，通过 `PromptEncoder.get_mask_embeddings()` 编码后加到图像特征上。

### 论文对应

- 论文 §3.1 提到使用 ViTDet-H [22] 进行检测
- Fast 版本用 YOLO11-Pose 替代，同时提供 body + hand bbox（见 §3.2 Decoupled Spatial Priors）
- 论文 Table 7 对比了两种检测器的性能

---

## 2.1.2 FOV 估计

### 代码位置
`tools/build_fov_estimator.py` → `FOVEstimator`

### 实现

唯一支持的后端是 MoGe2（Monocular Geometry Estimation v2）：

```python
# load_moge()
moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")

# run_moge()
moge_data = model.infer(input_image)
intrinsics = denormalize_f(moge_data["intrinsics"], H, W)
v_focal = intrinsics[1, 1]
intrinsics[0, 0] = v_focal  # 用垂直焦距覆盖水平焦距
```

**关键设计**：MoGe2 使用 v_focal 覆盖 h_focal（假设方形像素），输出 3×3 内参矩阵。

**无 FOV 估计器时的默认值**：`prepare_batch()` 中默认 `focal_length = √(H² + W²)`。

### 论文对应

- 论文 §3.2 FOV Estimator 消融（Table 5）验证了 smallest model at lowest resolution 即可
- Fast 版本选择 MoGe2 ViT-S（35M 参数）而非 ViT-L，因为 FOV 估计任务已饱和

---

## 2.1.3 图像编码（ViT/DINOv3 + Ray Conditioning）

### 代码位置
- **骨干网络**：`models/backbones/vit.py` + `models/backbones/dinov3.py`
- **Ray Conditioning**：`models/modules/camera_embed.py` → `CameraEncoder`

### 骨干网络选项

| 工厂函数 | 架构 | embed_dim | depth | 输入尺寸 | 说明 |
|---------|------|-----------|-------|---------|------|
| `vit()` | ViT-H | 1280 | 32 | 256×192 | 原始 3DB 默认 |
| `vit512_384()` | ViT-H | 1280 | 32 | 512×384 | 高分辨率变体 |
| `vit_l()` | ViT-L | 1024 | 24 | 256×192 | 轻量变体 |
| `vit_b()` | ViT-B | 768 | 12 | 256×192 | 最轻量 |
| `dinov3_vit*` | DINOv3 | varies | varies | varies | Torch Hub 加载 |

**自定义 ViT 特性**：
- 绝对位置编码 + `get_abs_pos()` 双三次插值（支持不同分辨率）
- `DropPath` 随机深度
- 可选 Flash Attention（`flash_attn_func`）
- 支持 `frozen_stages` 冻结前 N 层

**DINOv3 骨干**：
- 通过 `torch.hub.load("facebookresearch/dinov3")` 加载
- `get_layer_depth()` 支持分层学习率调度

### Ray Conditioning

`CameraEncoder` 将相机射线方向编码并注入图像特征：

```
输入: img_embeddings [B, C, H, W] + rays [B, 2, H, W]
  │
  ▼
rays → 下采样 (1/patch_size) → 补齐为 3D (x,y,1)
  │
  ▼
FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)
  │  → 99 维 Fourier 特征 (sin-cos + 原始坐标)
  ▼
concat([img_embeddings, rays_embeddings]) → Conv2d(C+99, C, 1×1) → LayerNorm2d
  │
  ▼
输出: ray-conditioned embeddings [B, C, H, W]
```

**Ray 计算流程** (`SAM3DBody.get_ray_condition()`)：
```python
meshgrid_xy = ...  # 像素坐标网格
# 逆仿射变换: 裁剪坐标 → 原图坐标
meshgrid_xy = (meshgrid_xy - affine_trans[0,1,2]) / affine_trans[0,0]
# 减去光心, 除以焦距 → 射线方向
meshgrid_xy = (meshgrid_xy - cam_int[0,1,2]) / cam_int[0,0,1,1]
```

### 论文对应

- 论文 §3.1 公式: `F = Enc(I_body) ∈ R^{h×w×D}`, "optionally fused with Fourier-encoded pixel rays derived from K"
- 代码实现与论文描述一致：Fourier 编码 + 1×1 Conv 融合

---

## 2.1.4 Prompt 编码（关键点 + 掩码）

### 代码位置
`models/decoders/prompt_encoder.py` → `PromptEncoder`

### 架构

```
输入: keypoints [B, N, 3]  (x, y, label)
  │
  ├── PositionEmbeddingRandom: (x,y) → Fourier 位置编码 [B, N, embed_dim]
  │
  ├── point_embeddings[label]: 每个关节一个 Embedding(1, embed_dim)
  │   label == -2 → invalid_point_embed
  │   label == -1 → not_a_point_embed (负样本)
  │   label == 0..69 → point_embeddings[label]
  │
  ▼
sparse_embeddings = pos_enc + label_embed  → [B, N, embed_dim]
```

**掩码编码**（可选）：
- `v1`: Conv2d 下采样 (1→4→16→embed_dim)
- `v2`: 更精细的下采样 (1→4→16→64→256→embed_dim)
- 最后一层 Conv 零初始化作为门控
- `no_mask_embed` 用于无掩码情况

### 关键点提示采样器

`KeypointSamplerV1` 用于训练时的迭代提示采样：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `keybody_ratio` | 0.8 | 优先采样关键体部关节的概率 |
| `worst_ratio` | 0.8 | 选择误差最大关键点的概率 |
| `negative_ratio` | 0.0 | 采样负样本提示的概率 |
| `dummy_ratio` | 0.1 | 采样空提示的概率 |

关键体部关节: `[5,6,7,8,9,10,11,12,13,14,41,62]`（肩/肘/腕/髋/膝/踝 + 手腕）

### 论文对应

- 论文 §3.1: "learnable query tokens representing the initial MHR state, spatial prompts, and keypoints"
- Fast 版本禁用了关键点提示精调（Table 6: KP Prompt OFF → FPS↑）

---

## 2.1.5 Body Decoder

### 代码位置
- `models/meta_arch/sam3d_body.py` → `SAM3DBody.forward_decoder()`
- `models/decoders/promptable_decoder.py` → `PromptableDecoder`

### Token 构建策略

Body Decoder 的输入由多种 Token 拼接而成：

```
token_embeddings = [
    MHR Token,          # 1 token: init_to_token_mhr(condition + init_pose + init_cam) → [B, 1, D]
    Previous Token,     # 1 token: prev_to_token_mhr(prev_pose + prev_cam) → [B, 1, D]
    Prompt Token,       # 1 token: prompt_encoder → prompt_to_token → [B, 1, D]
    Hand Detect Token,  # 2 tokens: hand_box_embedding (可选) → [B, 2, D]
    Keypoint2D Token,   # 70 tokens: keypoint_embedding → [B, 70, D]
    Keypoint3D Token,   # 70 tokens: keypoint3d_embedding → [B, 70, D]
]
# 总计: 1 + 1 + 1 + 2 + 70 + 70 = 145 tokens (D=1024 for default config)
```

其中 `D = cfg.MODEL.DECODER.DIM`，默认 1024。

### 中间预测与关键点更新循环

```
Layer 0:
  │  Cross-Attention(token → image_features)
  │  Self-Attention(token → token)
  │  FFN
  ▼
  if layer_idx ∈ S (do_interm_preds):
    token_to_pose_output_fn(norm(tokens))
      → pose_token → MHRHead → ̂θ → MHR FK → 3D joints J^(ℓ)
      → PerspectiveHead → ̂cam → projection → 2D kps ̂J_2d^(ℓ)
    keypoint_token_update_fn():
      2D update: pred_2d_kps → keypoint_posemb_linear → token_augment
                 + grid_sample(image_feat, pred_2d_kps) → keypoint_feat_linear → token_embeddings
      3D update: pred_3d_kps (pelvis-normalized) → keypoint3d_posemb_linear → token_augment
  │
  ▼
Layer 1: ... (重复)
  │
  ▼
Layer L-1 (最后一层):
  │  Cross-Attention + Self-Attention + FFN
  │  norm_final → token_to_pose_output_fn → 最终 pose_output
  ▼
输出: (normalized_tokens, all_pose_outputs)
```

### 论文对应

- 论文 §3.1: "After every layer ℓ, the output token t_{mhr}^ℓ decodes intermediate MHR parameters"
- 论文 §3.2: "We introduce a configurable layer selection set S ⊂ {0,...,L-1} that strictly gates this execution"
- 代码中 `do_interm_preds` 和 `keypoint_token_update` 控制此行为

---

## 2.1.6 Hand Decoder（手部检测 → 裁剪 → 独立解码）

### 代码位置
- `SAM3DBody.forward_decoder_hand()` — 手部解码器前向
- `SAM3DBody._get_hand_box()` — 从 body decoder 输出提取手部 bbox
- `SAM3DBodyEstimator.process_one_image()` — 手部裁剪与推理编排

### 手部检测流程

Body Decoder 的输出包含手部检测 tokens：

```python
# 在 forward_decoder() 中:
hand_box_embedding  # nn.Embedding(2, D) — 左右手各一个 token
hand_cls_embed      # nn.Linear(D, 2) — 分类: 是否有手
bbox_embed          # MLP(D, D, 4, 3) — 回归 bbox (x1,y1,w,h)

# 输出:
output["mhr"]["hand_box"]    # [B, 2, 4] sigmoid 归一化坐标
output["mhr"]["hand_logits"] # [B, 2, 2] 手部存在概率
```

`_get_hand_box()` 将归一化坐标转换回原图空间：
```
crop_coords → 逆仿射变换 → 原图 coords → xyxy 格式
```

### 手部解码流程（串行，原始管线瓶颈）

```
1. Body Decoder 完成 → 获取 left_xyxy, right_xyxy
2. 左手: 
   - 翻转图像 → prepare_batch(flipped_img, transform_hand, left_xyxy)
   - forward_step(batch_lhand, "hand") → lhand_output
   - 反翻转: scale/hand_pose/joint_rots 左右互换
3. 右手:
   - prepare_batch(img, transform_hand, right_xyxy)
   - forward_step(batch_rhand, "hand") → rhand_output
```

**关键瓶颈**：
- 手部裁剪在 CPU 上执行（`prepare_batch` 使用 NumPy/CV2）
- 3 次独立 backbone forward pass（body + 左手 + 右手）
- 2 次 hand decoder forward pass（左手 + 右手，串行）

### Hand Decoder 与 Body Decoder 的区别

| 特性 | Body Decoder | Hand Decoder |
|------|-------------|--------------|
| 位置编码 | `prompt_encoder.get_dense_pe()` | `hand_pe_layer()` (独立 PE) |
| Ray Conditioning | `ray_cond_emb` | `ray_cond_emb_hand` |
| MHR Head | `head_pose` (body) | `head_pose_hand` (enable_hand_model=True) |
| Camera Head | `head_camera` | `head_camera_hand` |
| 初始化 | `init_pose` / `init_camera` | `init_pose_hand` / `init_camera_hand` |
| Token 映射 | `init_to_token_mhr` | `init_to_token_mhr_hand` |
| 手部 PCA | 推理时设为单位矩阵 | 推理时设为单位矩阵 |
| 手部 bbox | 输出包含手部检测 tokens | 不输出手部检测 |

### 论文对应

- 论文 §3.1: "hand bounding boxes are predicted by a dedicated head. Each hand crop is independently encoded and decoded"
- Fast 版本将 3 次 backbone forward 合并为 1 次批处理（Eq.2）

---

## 2.1.7 手-体合并与腕部 IK

### 代码位置
`SAM3DBody.run_inference()` — Step 3: Replace + IK

### 手部有效性判断（4 条准则）

```python
# CRITERIA 1: 局部腕关节旋转差异 < thresh_wrist_angle (默认 1.4 rad)
angle_difference_valid_mask = rotation_angle_difference(
    ori_local_wrist_rotmat,     # body decoder 的局部腕部旋转
    fused_local_wrist_rotmat    # hand decoder 的全局腕部旋转 × 逆(前臂旋转×零位旋转)
) < thresh_wrist_angle

# CRITERIA 2: 手部 bbox 尺寸 > 64 像素
hand_box_size_valid_mask = (bbox_scale > 64).all(dim=1)

# CRITERIA 3: 所有关键点 2D 投影在裁剪范围内
hand_kps2d_valid_mask = pred_keypoints_2d_cropped.abs().amax() < 0.5

# CRITERIA 4: 手部预测腕部与 body 预测腕部的 2D 距离 < 0.25 (归一化)
hand_wrist_kps2d_valid_mask = dist < 0.25

# 综合:
hand_valid_mask = C1 & C2 & C3 & C4  # [B, 2] 左右手独立
```

### 合并策略

```python
# 1. 替换手部姿态参数
pose_output["mhr"]["hand"] = torch.where(valid, updated_hand_pose, body_hand_pose)

# 2. 替换手部尺度
pose_output["mhr"]["scale"][:, 8] = rhand_scale  # 右手
pose_output["mhr"]["scale"][:, 9] = lhand_scale  # 左手

# 3. 替换共享形状/尺度 (加权平均)
shared_scale = (lhand_scale * valid_left + rhand_scale * valid_right) / n_valid
shared_shape = (lhand_shape * valid_left + rhand_shape * valid_right) / n_valid
```

### 腕部 IK

```python
# 1. 用合并后的参数做一次 FK → 获取关节旋转
joint_rotations = mhr_forward(..., return_joint_rotations=True)

# 2. 计算手部 decoder 的全局腕部旋转 → 转为局部旋转
fused_local_wrist_rotmat = pred_global_wrist_rotmat × inv(lowarm_rot × joint_rotation[wrist])

# 3. 修正 Euler 角到关节限位内
wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat))
# 限位: X ∈ [-2.2, 1.0], Z ∈ [-2.2, 1.5], Y ∈ [-1.2, 1.5]

# 4. 仅在有效性检验通过时替换
updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
```

### 关键点提示精调

```python
# 组装关键点提示: 手部 decoder 的手腕 + body decoder 的肘部
keypoint_prompt = [right_wrist, left_wrist, right_elbow, left_elbow]
# 标签: [41, 62, 8, 7] (关节索引)

# 用这些提示重新运行 body decoder
pose_output = run_keypoint_prompt(batch, pose_output, keypoint_prompt)
```

### 论文对应

- 论文 §3.1: "the merged 2D keypoints are fed back as spatial prompts for a complete second forward pass"
- Fast 版本**禁用**了此精调步骤（§3.2: "Disabling keypoint-prompted refinement"）

---

## 2.1.8 迭代 MHR-to-SMPL 转换

### 论文描述

论文 Eq.(1) 定义了迭代优化：
```
Θ̂_smpl = argmin_Θ ‖V_mhr - V_smpl(Θ)‖² + R(Θ)
```
其中 `V_mhr` 是 MHR 预测的网格，`V_smpl(Θ)` 是 SMPL 网格，`R` 为解剖学正则化项。需要**数百步迭代**。

### 代码现状

当前代码**未实现** MHR-to-SMPL 迭代转换。代码仅输出 MHR 参数和 MHR 前向运动学结果：

```python
# mhr_head.py: mhr_forward()
verts, j3d, jcoords, mhr_model_params, joint_global_rots = self.head_pose.mhr_forward(
    global_rot=...,
    body_pose_params=...,
    hand_pose_params=...,
    scale_params=...,
    shape_params=...,
    ...)
# 输出: MHR 顶点 (18,439 个), 70 个关键点, 关节坐标
```

输出字典包含 `mhr_model_params`，可用于后续的 MHR-to-SMPL 转换，但转换本身不在代码库中。

### Fast 版本的替代方案

论文 Eq.(4) 用 MLP 替代迭代优化：
```
Θ̂_smpl = f_ω(x)
```
- 输入: MHR 顶点 → 重心坐标投影到 SMPL 表面 → 子采样 1,500 个顶点 → 去质心 → 展平为 4,500 维
- 架构: 3 层 MLP (4500 → 512 → 256 → 76)
- 输出: 76 维 SMPL 参数 (3 全局旋转 + 63 体部姿态 + 10 形状系数)
- 训练: 用原始迭代拟合生成训练对，损失为 `L1 顶点误差 + L2 参数误差`

**此模块在代码中完全缺失**。

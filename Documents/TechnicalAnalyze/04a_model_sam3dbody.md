# 4a. 核心模型深度分析 — SAM3DBody 主模型

> 对应代码：`models/meta_arch/sam3d_body.py` — `SAM3DBody` 类

## 4a.1 模型架构总览

`SAM3DBody` 继承自 `BaseModel`，是整个系统的核心模型类，协调 backbone、decoder、head 三大组件的交互。

```
SAM3DBody(BaseModel)
├── _initialze_model()          # 初始化所有子模块
├── forward_step()              # 单步前向 (body 或 hand)
├── run_inference()             # 完整推理流程
├── run_keypoint_prompt()       # 关键点提示精调
├── forward_decoder()           # Body 解码
├── forward_decoder_hand()      # Hand 解码
├── _get_hand_box()             # 手部 bbox 提取
├── _one_prompt_iter()          # 单轮迭代提示
├── keypoint_token_update_fn()         # 2D 关键点 token 更新 (body)
├── keypoint3d_token_update_fn()       # 3D 关键点 token 更新 (body)
├── keypoint_token_update_fn_hand()    # 2D 关键点 token 更新 (hand)
├── keypoint3d_token_update_fn_hand()  # 3D 关键点 token 更新 (hand)
├── camera_project()            # Body 相机投影
├── camera_project_hand()       # Hand 相机投影
├── get_ray_condition()         # 计算 ray 条件
└── _full_to_crop()             # 全图坐标→裁剪坐标
```

---

## 4a.2 子模块初始化详解

### 4a.2.1 Backbone

```python
self.backbone = create_backbone(cfg.MODEL.BACKBONE.TYPE, cfg)
```

| 配置项 | 工厂函数 | 架构 | embed_dim | depth |
|--------|---------|------|-----------|-------|
| `vit_hmr` | `vit()` | ViT-H | 1280 | 32 |
| `vit_hmr_512_384` | `vit512_384()` | ViT-H | 1280 | 32 |
| `vit_l` | `vit_l()` | ViT-L | 1024 | 24 |
| `vit_b` | `vit_b()` | ViT-B | 768 | 12 |

backbone 输出 `self.backbone.embed_dims` 和 `self.backbone.embed_dim`（同一值），被多个下游模块引用。

### 4a.2.2 Head 模块（4 个独立 Head）

| Head | 变量名 | 用途 | npose/ncam | 特殊说明 |
|------|--------|------|------------|---------|
| Body Pose | `head_pose` | Body MHR 参数回归 | 469 | `hand_pose_comps` 替换为单位矩阵 |
| Hand Pose | `head_pose_hand` | Hand MHR 参数回归 | 469 | `enable_hand_model=True` |
| Body Camera | `head_camera` | Body 相机平移 (s,tx,ty) | 3 | |
| Hand Camera | `head_camera_hand` | Hand 相机平移 | 3 | 可配置 `DEFAULT_SCALE_FACTOR_HAND` |

**关键设计**：推理时 `hand_pose_comps` 被替换为 `torch.eye(54)`，意味着手部 PCA 退化——直接输出 54 维手部参数而非 PCA 系数。原始 PCA 分量保存在 `hand_pose_comps_ori` 中。

### 4a.2.3 初始化 Embedding

```python
self.init_pose = nn.Embedding(1, 469)       # Body 零位姿态
self.init_pose_hand = nn.Embedding(1, 469)  # Hand 零位姿态
self.init_camera = nn.Embedding(1, 3)       # Body 相机初始化 (零初始化)
self.init_camera_hand = nn.Embedding(1, 3)  # Hand 相机初始化 (零初始化)
```

### 4a.2.4 Token 映射层

```python
# Body: condition(3) + pose(469) + cam(3) = 475 → D
self.init_to_token_mhr = nn.Linear(475, D)
# Body: pose(469) + cam(3) = 472 → D (无 condition)
self.prev_to_token_mhr = nn.Linear(472, D)
# Hand: 同结构，独立参数
self.init_to_token_mhr_hand = nn.Linear(475, D)
self.prev_to_token_mhr_hand = nn.Linear(472, D)
```

**condition_info 维度 = 3**：CLIFF 风格条件 `(cx-cx₀)/f, (cy-cy₀)/f, b/f`。

### 4a.2.5 Prompt 编码器

```python
self.prompt_encoder = PromptEncoder(
    embed_dim=self.backbone.embed_dims,  # 需匹配 backbone PE 维度
    num_body_joints=70,                   # MHR70 关键点
    frozen=cfg.MODEL.PROMPT_ENCODER.FROZEN,
    mask_embed_type=cfg.MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE,
)
self.prompt_to_token = nn.Linear(backbone_embed_dims, D)  # 1280 → 1024
```

**关键点采样器**：
```python
self.keypoint_prompt_sampler = build_keypoint_sampler(
    cfg.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER,
    prompt_keypoints=PROMPT_KEYPOINTS["mhr70"],  # {i: i for i in range(70)}
    keybody_idx=KEY_BODY,  # [5,6,7,8,9,10,11,12,13,14,41,62]
)
```

### 4a.2.6 Decoder

```python
self.decoder = build_decoder(cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims)
self.decoder_hand = build_decoder(cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims)
# 两个 decoder 共享配置，但各自有独立参数
```

### 4a.2.7 Token 集合

| Token | 数量 | 变量 | 条件 |
|-------|------|------|------|
| Hand Detect | 2 | `hand_box_embedding` | `DO_HAND_DETECT_TOKENS=True` |
| Keypoint 2D | 70 | `keypoint_embedding` | `DO_KEYPOINT_TOKENS=True` (必须) |
| Keypoint 3D | 70 | `keypoint3d_embedding` | `DO_KEYPOINT3D_TOKENS=True` |

**Token 更新函数**：

| 函数 | 输入 | 操作 |
|------|------|------|
| `keypoint_posemb_linear` | 2D 坐标 (2→D) | FFN: 2→D→D (2层) |
| `keypoint3d_posemb_linear` | 3D 坐标 (3→D) | FFN: 3→D→D (2层) |
| `keypoint_feat_linear` | 图像特征 (backbone_dim→D) | Linear: 1280→1024 |

### 4a.2.8 Ray Conditioning

```python
self.ray_cond_emb = CameraEncoder(backbone.embed_dim, backbone.patch_size)
self.ray_cond_emb_hand = CameraEncoder(backbone.embed_dim, backbone.patch_size)
```

两个独立的 `CameraEncoder`，分别用于 body 和 hand。

### 4a.2.9 手部检测头

```python
self.hand_cls_embed = nn.Linear(D, 2)     # 手部存在分类
self.bbox_embed = MLP(D, D, 4, 3)         # bbox 回归 (3层 MLP)
```

### 4a.2.10 FP16 支持

```python
if cfg.TRAIN.USE_FP16:
    self.convert_to_fp16()
    backbone_dtype = torch.float16 / torch.bfloat16
```

---

## 4a.3 `forward_decoder()` 详解

### Token 构建流程

```
1. init_estimate = [init_pose(469), init_camera(3)] → [B, 1, 472]
   + condition_info → [B, 1, 475]

2. token_embeddings = init_to_token_mhr(init_input) → [B, 1, D]

3. 如果有 prompt:
   a. prev_embeddings = prev_to_token_mhr(prev_estimate) → [B, 1, D]
   b. image_augment = prompt_encoder.get_dense_pe() 或 hand_pe_layer()
   c. image_embeddings = ray_cond_emb(image_embeddings, ray_cond)
   d. prompt_embeddings = prompt_to_token(prompt_encoder(keypoints)) → [B, 1, D]
   e. token_embeddings = [MHR(1), Prev(1), Prompt(1)] → [B, 3, D]

4. 可选: Hand Detect Tokens → [B, 5, D]
5. 必须: Keypoint2D Tokens → [B, 73, D]
6. 可选: Keypoint3D Tokens → [B, 143, D]
```

### 中间预测函数 `token_to_pose_output_fn`

在每层 decoder 后（如果 `do_interm_preds=True`），执行：

```
pose_token = tokens[:, 0]  # MHR token
  │
  ├── head_pose(pose_token, prev_pose) → MHR 参数
  │     ├── global_rot: 6D → rotmat → euler
  │     ├── body_pose: 260D cont → compact_cont_to_model_params → 130 euler
  │     ├── shape: 45D
  │     ├── scale: 28D
  │     ├── hand: 108D (54×2)
  │     └── face: 72D (×0, 置零)
  │
  ├── head_camera(pose_token, prev_cam) → (s, tx, ty)
  │
  ├── mhr_forward(global_rot, body_pose, hand_pose, scale, shape, face)
  │     → vertices, keypoints_3d, joint_coords, joint_global_rots
  │
  ├── camera_project(keypoints_3d, pred_cam, bbox_center, bbox_scale, cam_int)
  │     → keypoints_2d, cam_t, focal_length
  │
  └── _full_to_crop(keypoints_2d, affine_trans)
        → keypoints_2d_cropped (归一化到 [-0.5, 0.5])
```

### 关键点 Token 更新函数 `keypoint_token_update_fn`

**2D 更新**（`keypoint_token_update_fn`）：

```
1. 获取当前层预测的 2D 关键点 (crop 坐标, -0.5~0.5)
2. 计算无效掩码: 超出图像边界 或 深度 < 1e-5
3. 位置编码更新:
   token_augment[:, kps_start:kps_end] = keypoint_posemb_linear(pred_2d_kps) * ~invalid
4. 特征采样更新:
   grid_sample(image_embeddings, pred_2d_kps * 2) → pred_2d_kps_feats
   token_embeddings[:, kps_start:kps_end] += keypoint_feat_linear(pred_2d_kps_feats * ~invalid)
```

**3D 更新**（`keypoint3d_token_update_fn`）：

```
1. 获取当前层预测的 3D 关键点
2. 骨盆归一化: pred_3d_kps -= (left_hip + right_hip) / 2
3. 位置编码更新:
   token_augment[:, kps3d_start:kps3d_end] = keypoint3d_posemb_linear(pred_3d_kps)
```

**注意**：最后一层不执行更新（`if layer_idx == len(self.decoder.layers) - 1: return`）。

---

## 4a.4 `forward_decoder_hand()` 详解

结构与 `forward_decoder()` 高度对称，关键差异：

| 特性 | Body Decoder | Hand Decoder |
|------|-------------|--------------|
| 初始化 | `init_pose` / `init_camera` | `init_pose_hand` / `init_camera_hand` |
| Token 映射 | `init_to_token_mhr` | `init_to_token_mhr_hand` |
| 位置编码 | `prompt_encoder.get_dense_pe()` | `hand_pe_layer()` |
| Ray Conditioning | `ray_cond_emb` | `ray_cond_emb_hand` |
| MHR Head | `head_pose` | `head_pose_hand` (enable_hand_model=True) |
| Camera Head | `head_camera` | `head_camera_hand` |
| 关键点 Embedding | `keypoint_embedding` | `keypoint_embedding_hand` |
| 关键点 3D Embedding | `keypoint3d_embedding` | `keypoint3d_embedding_hand` |
| 位置编码线性层 | `keypoint_posemb_linear` | `keypoint_posemb_linear_hand` |
| 3D 位置编码线性层 | `keypoint3d_posemb_linear` | `keypoint3d_posemb_linear_hand` |
| 特征线性层 | `keypoint_feat_linear` | `keypoint_feat_linear_hand` |
| Decoder 实例 | `self.decoder` | `self.decoder_hand` |

**`head_pose_hand` 的特殊行为**（`enable_hand_model=True`）：
- 全局旋转从腕部坐标系转换到世界坐标系
- 全局平移根据腕部偏移修正
- 非手部参数被置零（`model_params[:, nonhand_param_idxs] = 0`）
- 仅输出右手关键点（`model_keypoints_pred[:, :21] = 0; model_keypoints_pred[:, 42:] = 0`）

---

## 4a.5 `run_inference()` 完整流程

```
输入: img, batch, inference_type, transform_hand, thresh_wrist_angle
│
├─ if inference_type == "body": forward_step("body") → return
├─ if inference_type == "hand": forward_step("hand") → return
│
├─ [Step 1] Body 推理
│  pose_output = forward_step(batch, "body")
│  left_xyxy, right_xyxy = _get_hand_box(pose_output, batch)
│  ori_local_wrist_rotmat = euler_to_rotmat(body wrist euler)
│
├─ [Step 2] 左手推理 (翻转)
│  flipped_img = img[:, ::-1]
│  batch_lhand = prepare_batch(flipped_img, transform_hand, left_xyxy)
│  lhand_output = forward_step(batch_lhand, "hand")
│  # 反翻转: scale/hand_pose/joint_rots 左右互换
│
├─ [Step 3] 右手推理
│  batch_rhand = prepare_batch(img, transform_hand, right_xyxy)
│  rhand_output = forward_step(batch_rhand, "hand")
│
├─ [Step 4] 手部有效性判断 (4 条准则)
│  hand_valid_mask = C1 & C2 & C3 & C4
│
├─ [Step 5] 关键点提示精调
│  keypoint_prompt = [right_wrist, left_wrist, right_elbow, left_elbow]
│  run_keypoint_prompt(batch, pose_output, keypoint_prompt)
│
├─ [Step 6] 手部参数替换
│  hand_pose, scale, shape ← lhand/rhand 对应分量
│
├─ [Step 7] 腕部 IK
│  mhr_forward → joint_rotations
│  fused_local_wrist = pred_global_wrist × inv(lowarm × joint_rotation)
│  wrist_xzy = fix_wrist_euler(euler)
│  body_pose ← where(valid, wrist_xzy, body_pose)
│
├─ [Step 8] 最终 FK + 投影
│  mhr_forward(return_keypoints, return_joint_coords, return_model_params, return_joint_rotations)
│  vertices, keypoints_3d, joint_coords ← ... (坐标系翻转 y,z *= -1)
│
└─ [Step 9] 手动 2D 投影
   # 不使用 camera_project，直接用焦距和光心计算
   pred_keypoints_2d = (keypoints_3d + cam_t) * focal + center * depth
```

---

## 4a.6 `_get_hand_box()` 手部边界框提取

```
1. 从 body decoder 输出获取归一化 bbox:
   hand_box[:, 0] → 左手 (x1, y1, w, h), 坐标范围 [0, IMAGE_SIZE]
   hand_box[:, 1] → 右手 (x1, y1, w, h)

2. 取 max(w, h) 使 bbox 为正方形

3. 逆仿射变换回原图坐标:
   scale_orig = scale_crop / affine_trans[0,0]
   center_orig = (center_crop - affine_trans[0,[0,1],2]) / affine_trans[0,0]

4. 转换为 xyxy 格式
```

---

## 4a.7 `forward_step()` 和 `forward_pose_branch()`

```python
def forward_step(batch, decoder_type="body"):
    # 设置 body/hand batch_idx
    if decoder_type == "body":
        body_batch_idx = range(B*N)
        hand_batch_idx = []
    else:
        body_batch_idx = []
        hand_batch_idx = range(B*N)
    
    return forward_pose_branch(batch)
```

`forward_pose_branch()` 执行：
1. 数据预处理 + backbone 编码
2. Ray conditioning 计算
3. 可选 Mask conditioning
4. CLIFF 条件信息计算
5. 构建 dummy prompt (label=-2)
6. 对 body_batch_idx 执行 `forward_decoder()`
7. 对 hand_batch_idx 执行 `forward_decoder_hand()`
8. 提取手部 bbox（如果 `DO_HAND_DETECT_TOKENS`）

---

## 4a.8 `get_ray_condition()` 射线条件计算

```
1. 构建像素网格: meshgrid(H, W)
2. 逆仿射变换: pixel → original image coords
3. 减去光心, 除以焦距 → 射线方向 (2D)
4. 输出: [B, N, 2, H, W]  (N=num_person)

注意: ViT 分辨率裁剪:
  - vit_hmr/vit/vit_b/vit_l: ray_cond[:, :, :, 32:-32]
  - vit_hmr_512_384: ray_cond[:, :, :, 64:-64]
```

---

## 4a.9 FP16 转换

```python
def convert_to_fp16(self):
    self.backbone = self.backbone.half()
    self.decoder = self.decoder.half()
    self.decoder_hand = self.decoder_hand.half()
    # head, prompt_encoder 等保持 fp32
```

backbone 和 decoder 使用 FP16，其余模块保持 FP32。推理时 backbone 输入会被转换为 `self.backbone_dtype`。

---

## 4a.10 条件信息计算 `_get_decoder_condition()`

CLIFF 风格条件（3D 感知的 2D 条件信息）：

```python
# condition_info = [(cx - cx₀)/f, (cy - cy₀)/f, b/f]
# cx, cy: bbox 中心 (原图坐标)
# cx₀, cy₀: 光心 (cam_int[0,2], cam_int[1,2])
# b: bbox 尺度
# f: 焦距 (cam_int[0,0])
```

可选 `USE_INTRIN_CENTER`：使用内参矩阵的光心而非图像中心。

---

## 4a.11 与 Fast 版本的差距

| 模块 | 当前实现 | Fast 需求 | 改动量 |
|------|---------|----------|--------|
| Body/Hand 独立编码 | 3 次 backbone forward | 批处理 1 次 | **大** |
| 串行 Hand Decoder | 2 次串行 forward | 批处理 1 次 | **大** |
| CPU 手部裁剪 | `prepare_batch()` (NumPy) | GPU-native `grid_sample` | **大** |
| 关键点提示精调 | 总是执行 | 禁用 | **小** |
| MHR-to-SMPL | 未实现 | MLP 投影 | **新模块** |
| 中间预测层选择 | 全量执行 | 可配置集合 S | **小** |
| 静态图编译 | 无 | TRT / torch.compile | **中** |

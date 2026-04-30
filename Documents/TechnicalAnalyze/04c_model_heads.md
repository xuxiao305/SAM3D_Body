# 4c. 核心模型深度分析 — MHRHead & PerspectiveHead

> 对应代码：
> - `models/heads/mhr_head.py` — `MHRHead` 类
> - `models/heads/camera_head.py` — `PerspectiveHead` 类

## 4c.1 MHRHead 架构总览

`MHRHead` 是 MHR (Momentum Human Representation) 参数回归头，将 decoder 输出的 pose token 转换为完整的 MHR 身体模型参数，并运行 MHR 前向运动学。

```
MHRHead(nn.Module)
├── proj: FFN (input_dim → npose)     # 参数回归
├── mhr: TorchScript / Momentum MHR   # 前向运动学引擎
├── 关节旋转: nn.Parameter(127, 3, 3) # MHR 骨架关节旋转
├── scale_mean: nn.Parameter(68)       # 尺度均值
├── scale_comps: nn.Parameter(28, 68)  # 尺度 PCA 分量
├── hand_pose_mean: nn.Parameter(54)   # 手部姿态均值
├── hand_pose_comps: nn.Parameter(54, 54)  # 手部 PCA 分量
├── keypoint_mapping: nn.Parameter(308, 18439+127)  # 顶点→关键点映射
└── (hand 版本额外参数)
    ├── right_wrist_coords: nn.Parameter(3)
    ├── root_coords: nn.Parameter(3)
    ├── local_to_world_wrist: nn.Parameter(3, 3)
    └── nonhand_param_idxs: nn.Parameter(145)
```

---

## 4c.2 参数空间详解

### npose = 469 维参数分解

```
npose = 6 (global_rot)           # 6D 旋转表示
       + 260 (body_cont)         # 紧凑连续体部姿态
       + 45 (shape)              # 形状系数
       + 28 (scale)              # 尺度系数
       + 54×2 (hand)             # 左右手各 54 维
       + 72 (face)               # 面部表情
       = 469
```

### 紧凑连续表示

**体部姿态**：260 维连续表示 → `compact_cont_to_model_params_body()` → 130 维 Euler 角（65 个关节 × 2 角，部分关节为 3 角）

**手部姿态**：54 维 PCA 系数 → `hand_pose_mean + hand_pose_comps × pca_coeffs` → 27 个关节旋转（每手）

**全局旋转**：6D 表示 → `rot6d_to_rotmat()` → 3×3 旋转矩阵 → `roma.rotmat_to_euler("ZYX")` → 3 维 Euler 角

### 推理时手部 PCA 退化

```python
# _initialze_model() 中:
self.head_pose.hand_pose_comps_ori = nn.Parameter(
    self.head_pose.hand_pose_comps.clone(), requires_grad=False
)
self.head_pose.hand_pose_comps.data = torch.eye(54).float()
```

推理时将 PCA 分量矩阵替换为单位矩阵，使得 54 维输入直接作为关节旋转参数（而非 PCA 系数）。原始 PCA 保存在 `hand_pose_comps_ori` 中。

---

## 4c.3 `forward()` 详解

```
输入: x [B, D] (pose token), init_estimate [B, 469]
  │
  ├── pred = proj(x) + init_estimate  # 残差回归
  │
  ├── [0:6] → global_rot_6d → rotmat → euler("ZYX") → global_rot [B, 3]
  │     global_trans = zeros  # 平移在 mhr_forward 中处理
  │
  ├── [6:266] → body_cont [B, 260]
  │     → compact_cont_to_model_params_body() → body_pose_euler [B, 130]
  │     → zero-out 手部关节 (mhr_param_hand_mask)
  │     → zero-out 下巴 ([-3:])
  │
  ├── [266:311] → shape [B, 45]
  ├── [311:339] → scale [B, 28]
  ├── [339:447] → hand [B, 108]  (54×2, 左右手)
  ├── [447:519] → face [B, 72]  (×0, 置零)
  │
  └── mhr_forward(global_trans, global_rot, body_pose, hand_pose, scale, shape, face)
        → verts [B, 18439, 3]
        → j3d [B, 70, 3]  (308→70 关键点)
        → jcoords [B, 127, 3]
        → mhr_model_params
        → joint_global_rots [B, 127, 3, 3]
```

**坐标系修正**：MHR 使用 OpenGL 风格坐标系，代码中翻转 Y 和 Z 轴：
```python
verts[..., [1, 2]] *= -1
j3d[..., [1, 2]] *= -1
jcoords[..., [1, 2]] *= -1
```

---

## 4c.4 `mhr_forward()` 详解

### 参数转换

```
1. scale: [B, 28] → scale_mean + scale @ scale_comps → [B, 68] 实际尺度
2. body_pose: [B, 130] euler → full_pose_params = [global_trans×10, global_rot, body_pose] → [B, 127]
3. hand_pose: 通过 replace_hands_in_pose() 替换 full_pose_params 中的手部关节
4. model_params = [full_pose_params, scales] → [B, 195]
```

### 手部替换 `replace_hands_in_pose()`

```
1. left/right_hand_params = split(hand_pose, [54, 54])
2. hand_params_model = compact_cont_to_model_params_hand(
     hand_pose_mean + hand_params @ hand_pose_comps
   )
3. full_pose_params[hand_joint_idxs_left] = left_hand_model_params
4. full_pose_params[hand_joint_idxs_right] = right_hand_model_params
```

### MHR 引擎

```python
# 两种引擎:
if MOMENTUM_ENABLED:  # 环境变量 MOMENTUM_ENABLED
    self.mhr = MHR.from_files(device, lod=1)
else:
    self.mhr = torch.jit.load(mhr_model_path)

# 前向:
curr_skinned_verts, curr_skel_state = self.mhr(shape_params, model_params, expr_params)
# curr_skinned_verts: [B, 18439, 3]  — 蒙皮顶点
# curr_skel_state: [B, 127, 8]  — (3 关节坐标 + 4 四元数 + 1)
```

### 关键点映射

```python
# 308 个关键点通过线性映射获得:
model_keypoints_pred = keypoint_mapping @ model_vert_joints.flatten(1,2)
# keypoint_mapping: [308, 18439+127]  — 稀疏映射矩阵
```

### Hand 版本特殊逻辑

当 `enable_hand_model=True` 时：
```python
# 全局旋转: 从腕部坐标系转到世界坐标系
global_rot = euler_to_rotmat("xyz", global_rot) @ local_to_world_wrist

# 全局平移: 补偿腕部偏移
global_trans = -(rotmat @ (right_wrist_coords - root_coords) + root_coords) + global_trans_ori

# 非手部参数置零
model_params[:, nonhand_param_idxs] = 0

# 仅输出右手关键点
model_keypoints_pred[:, :21] = 0  # 左手置零
model_keypoints_pred[:, 42:] = 0  # 其余置零
```

---

## 4c.5 PerspectiveHead 架构

```
PerspectiveHead(nn.Module)
├── proj: FFN (input_dim → 3)  # 相机参数回归
└── ncams = 3  # (s, tx, ty)
```

### `forward()`

```python
pred_cam = proj(x)  # [B, 3]
if init_estimate is not None:
    pred_cam = pred_cam + init_estimate  # 残差
return pred_cam  # (scale, tx, ty)
```

### `perspective_projection()`

```
输入:
  - points_3d: [B, K, 3]  — 3D 关键点
  - pred_cam: [B, 3]      — (s, tx, ty)
  - bbox_center: [B, 2]   — bbox 中心
  - bbox_size: [B]         — bbox 尺度
  - img_size: [B, 2]       — 原图尺寸
  - cam_int: [B, 3, 3]     — 相机内参

流程:
1. 坐标系翻转: pred_cam[0,2] *= -1
2. 计算深度: tz = 2 * focal_length / (bbox_size * s * default_scale_factor + ε)
3. 计算偏移:
   - cx = 2 * (bbox_center_x - img_w/2) / bs  (或使用内参光心)
   - cy = 2 * (bbox_center_y - img_h/2) / bs
4. 相机平移: cam_t = [tx + cx, ty + cy, tz]
5. 变换: j3d_cam = points_3d + cam_t
6. 投影: j2d = perspective_projection(j3d_cam, cam_int)

输出:
  - pred_keypoints_2d: [B, K, 2]
  - pred_cam_t: [B, 3]     — 完整相机平移
  - focal_length: [B]       — 焦距
  - pred_keypoints_2d_depth: [B, K]  — 深度值
```

### Zolly 模型（CLIFF/CameraHMR 相机模型）

核心思想：弱透视 + 焦距归一化

```
s = pred_cam[0]    # 尺度因子
tx = pred_cam[1]   # NDC 空间 x 偏移
ty = pred_cam[2]   # NDC 空间 y 偏移

# 深度估计 (Zolly 方案):
tz = 2f / (bbox_size × s)

# NDC 偏移:
cx = 2(bbox_center_x - img_center_x) / (bbox_size × s)
cy = 2(bbox_center_y - img_center_y) / (bbox_size × s)

# 完整平移:
cam_t = [tx + cx, ty + cy, tz]
```

**`default_scale_factor`**：Hand 版本可配置不同的默认尺度因子，适应手部裁剪的特殊尺度。

---

## 4c.6 MHR 参数的 compact 连续表示

### 编码：`compact_model_params_to_cont_body()`

将 130 维 Euler 角编码为 260 维连续表示：
- 每个 Euler 角 → sincos 编码 (sin(θ), cos(θ)) → 2 维
- 130 × 2 = 260 维

### 解码：`compact_cont_to_model_params_body()`

将 260 维连续表示解码为 130 维 Euler 角：
- 每对 (sin_val, cos_val) → atan2(sin_val, cos_val) → θ
- 260 / 2 = 130 维

### 手部：`compact_cont_to_model_params_hand()`

类似，但针对手部关节：
- 54 维 → 每对 sincos → 27 个 Euler 角

### 6D 旋转表示

全局旋转使用 6D 表示（Zhou et al.）：
```
6D → rot6d_to_rotmat() → 3×3 旋转矩阵
```
6D 表示的前两列直接来自预测，第三列通过叉积计算，保证正交性。

---

## 4c.7 关键点映射矩阵

`keypoint_mapping: [308, 18439 + 127]` 是一个稀疏线性映射，将 MHR 顶点 (18,439) 和关节坐标 (127) 映射到 308 个关键点。

```
model_vert_joints = cat([verts, joint_coords], dim=1)  # [B, 18566, 3]
model_keypoints_pred = (keypoint_mapping @ model_vert_joints.permute(1,0,2).flatten(1,2))
                        .reshape(-1, B, 3).permute(1,0,2)  # [B, 308, 3]
```

**推理时截断**：`j3d = j3d[:, :70]` — 仅使用前 70 个 MHR70 关键点。

---

## 4c.8 与 Fast 版本的关系

### MHR-to-SMPL 缺失

`mhr_forward()` 输出的 `verts [B, 18439, 3]` 是 MHR 网格顶点，Fast 版本需要将其转换为 SMPL 参数。

**Fast 需要新增的模块**：

1. **重心坐标投影** `B(V_mhr)`：
   - 预计算 SMPL→MHR 三角形映射
   - 单次矩阵乘法: `Ṽ = keypoint_mapping_similar @ V_mhr`
   - 输出: `Ṽ ∈ R^{6890×3}`

2. **子采样 + MLP**：
   - 从 6,890 顶点中子采样 1,500 个
   - 去质心 → 展平 → 4,500 维输入
   - 3 层 MLP: 4500 → 512 → 256 → 76
   - 输出: 76 维 SMPL 参数

3. **运动学先验 MLP**：
   - 输入: SMPL body pose (63 维)
   - 去噪自然姿态
   - ~0.1ms 延迟

### 当前可复用的输出

`mhr_forward()` 的输出已经包含 MLP 所需的所有信息：
- `verts`: MHR 网格 (18,439 顶点)
- `mhr_model_params`: 完整 MHR 参数
- `joint_global_rots`: 关节全局旋转

Fast 的 MLP 可以直接作为后处理模块添加，无需修改 `MHRHead` 本身。

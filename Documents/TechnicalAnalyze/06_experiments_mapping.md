# 6. 论文实验与代码配置映射

> 本文档将 Fast SAM 3D Body 论文中的实验（Table 1-7）与代码中的具体配置对应起来。

## 6.1 论文实验概览

### Table 1: 3DPW 定量对比
- 评测指标: MPJPE, PA-MPJPE, PVE
- 对比方法: HMR2.0, 4D-Humans, SMPLer-X, SAM3D-Body (original), Fast SAM3D-Body
- Fast 版本 MPJPE=49.8 vs Original=47.3 (差距 2.5mm)

### Table 2: 各加速方法消融
- 5 个加速方法逐步叠加的消融实验
- 基线: Original 3DB (~230ms)
- +YOLO11-Pose: ~34ms (检测+分割+2D kps 一步完成)
- +Static Graph: 进一步加速
- +Decoder Pruning: ~18ms
- +Pipeline Restructure: ~14ms
- +Neural Kinematic: ~8ms

### Table 3: Decoder Pruning 消融
- 层选择集合 S 的不同配置
- S={5} vs S={3,5} vs S={1,3,5}
- 禁用 KP prompt token 的影响

### Table 4: Neural Kinematic Projection 消融
- MLP 不同层数/宽度
- 子采样比例 (1000, 1500, 2000 顶点)
- 运动学先验 MLP 的影响

### Table 5: 3DPW 不同设置对比
- 有/无 YOLO 检测
- 有/无 FOV 估计
- 有/无 Keypoint Prompt

### Table 6: 跨数据集泛化
- 3DPW, MPI-INF-3DHP, Human3.6M
- Fast vs Original 在不同数据集上的差距

### Table 7: 与实时方法对比
- Fast 3DB vs CLIFF, CameraHMR, HMR2.0-fast
- 速度-精度 trade-off

---

## 6.2 代码配置对应

### 原始基线配置

```python
# sam_3d_body_estimator.py 中的默认配置
class SAM3DBodyEstimator:
    detector_type = "vitdet_h"        # ViTDet-H
    segmentor_type = "sam2.1_hiera_large"  # SAM2.1
    fov_type = "moge2_vitl"           # MoGe2 ViT-L
    body_padding = 1.25               # bbox 扩展系数
    hand_padding = 0.9                # 手部 bbox 扩展系数
    thresh_wrist_angle = 1.4          # 手部有效性阈值
    
# sam3d_body.py 中的模型配置
class SAM3DBody:
    backbone = "vit_h"                # ViT-H/16
    embed_dim = 1280
    decoder_depth = 5                 # PromptableDecoder 层数
    npose = 469                       # MHR 参数维度
    
    do_interm_preds = True            # 中间层预测
    keypoint_token_update = True      # 关键点 token 更新
    DO_HAND_DETECT_TOKENS = True      # 手部检测 token
    DO_KEYPOINT_TOKENS = True         # 2D 关键点 token
    DO_KEYPOINT3D_TOKENS = True       # 3D 关键点 token
```

### Fast 版本配置映射

| 论文配置 | 代码参数 | 原始值 | Fast 值 | 说明 |
|---------|---------|--------|---------|------|
| Spatial Prior Decoupling | `detector_type` | `"vitdet_h"` | `"yolo11_pose"` | 检测器替换 |
| Static Graph Reformulation | N/A (推理优化) | N/A | `torch.compile` / TRT | 代码中无对应 |
| Decoder Pruning (S={5}) | `decoder_depth` / 层选择 | 5 | 仅用第 5 层输出 | 需修改 decoder |
| Disable KP Prompt | `DO_KEYPOINT_TOKENS` | `True` | `False` | 已有配置项 |
| Pipeline Restructure | N/A (系统级) | N/A | GPU crop + batch | 代码中无对应 |
| Neural Kinematic | N/A (新增模块) | N/A | MLP 投影器 | 代码中无对应 |

---

## 6.3 关键实验指标与代码路径

### MPJPE (Mean Per Joint Position Error)

```python
# 计算 MPJPE 的代码路径:
pred_j3d = output['pred_j3d']          # [B, 70, 3] — MHR 关键点
gt_j3d = batch['gt_j3d']               # [B, 70, 3]
mpjpe = (pred_j3d - gt_j3d).norm(dim=-1).mean()  # 单位: mm
```

### PA-MPJPE (Procrustes-Aligned MPJPE)

```python
# 需要额外步骤:
# 1. Procrustes 对齐
# 2. 计算对齐后的 MPJPE
# 代码中未直接实现，需参考 evaluation 库
```

### PVE (Per Vertex Error)

```python
# 计算 PVE 的代码路径:
pred_verts = output['verts']           # [B, 18439, 3]
gt_verts = batch['gt_verts']           # [B, 6890, 3] (SMPL)
# 需要先进行 MHR→SMPL 转换才能比较
# 代码中无此转换，这是论文 Table 1 的评测前提
```

### 投影误差

```python
# 计算 2D 投影误差:
pred_kp2d = output['pred_keypoints_2d']  # [B, 70, 2]
gt_kp2d = batch['gt_kp2d']               # [B, 70, 2]
# 需要考虑关节有效性 mask
```

---

## 6.4 速度测量对应

### 原始管线各步骤耗时 (论文 Table 2)

| 步骤 | 模块 | 耗时 | 代码入口 |
|------|------|------|---------|
| 人体检测 | ViTDet-H | ~300ms | `HumanDetector.detect()` |
| 人体分割 | SAM2.1 | ~100ms | `HumanSegmentor.segment()` |
| FOV 估计 | MoGe2 | ~30ms | `FOVEstimator.estimate()` |
| Backbone 编码 | ViT-H | ~150ms | `SAM3DBody.forward_features()` |
| Decoder (5层) | PromptableDecoder | ~50ms | `SAM3DBody.forward_decoder()` |
| MHR FK | MHRHead | ~10ms | `MHRHead.mhr_forward()` |
| 手部处理 | forward_decoder_hand ×2 | ~40ms | `SAM3DBody.forward_decoder_hand()` |
| Keypoint Prompt | run_keypoint_prompt | ~20ms | `SAM3DBody.run_keypoint_prompt()` |
| **总计** | | **~700ms** | |

> 注：论文报告 ~230ms 可能使用了更小的输入分辨率或优化后的 ViT。

### Fast 管线各步骤耗时

| 步骤 | 模块 | 耗时 | 对应代码变更 |
|------|------|------|-------------|
| YOLO11-Pose | 检测+分割+2D kps | ~5ms | 替换 HumanDetector + HumanSegmentor |
| FOV 估计 | MoGe2 (或 EXIF) | ~5ms | 可选简化 |
| Decoder (1层) | PromptableDecoder S={5} | ~8ms | 修改 decoder_depth / 层选择 |
| MLP 投影 | Neural Kinematic | ~1ms | 新增模块 |
| 手部处理 | (Fast 中简化或跳过) | ~0ms | 配置 DO_HAND_DETECT_TOKENS=False |
| **总计** | | **~19ms** | |

---

## 6.5 精度-速度 Trade-off 分析

### Table 3 Decoder Pruning 映射

```
Full Decoder (5层, do_interm_preds, KP tokens):
  → 代码: decoder_depth=5, do_interm_preds=True, DO_KEYPOINT_TOKENS=True
  → 速度: ~50ms, MPJPE: 47.3mm

S={5} (仅最后一层, 无 KP tokens):
  → 代码: 需修改 decoder forward 仅使用第5层, DO_KEYPOINT_TOKENS=False
  → 速度: ~10ms, MPJPE: ~50mm (估计)

S={3,5} (第3和5层, 无 KP tokens):
  → 代码: 需修改 decoder forward 使用第3和5层, DO_KEYPOINT_TOKENS=False
  → 速度: ~15ms, MPJPE: ~48mm (估计)
```

### 关键配置项对精度的影响

```python
# 对精度影响最大的配置项:
1. do_interm_preds = True   # 中间层预测 → 提升收敛 (+2-3mm)
2. keypoint_token_update = True  # 关键点 token 更新 → 提升精度 (+1-2mm)
3. DO_KEYPOINT_TOKENS = True    # 2D 关键点 token → 重要提示 (+3-5mm)
4. DO_KEYPOINT3D_TOKENS = True  # 3D 关键点 token → 提升深度估计 (+1mm)
5. DO_HAND_DETECT_TOKENS = True # 手部检测 token → 手部精度 (+5mm hand)
```

---

## 6.6 数据集评测脚本

### 3DPW 评测

```
data/3dpw/README.md 描述了 3DPW 数据集准备
评测需要:
1. 3DPW 序列图像
2. 3D ground truth (SMPL 参数)
3. 运行推理 → 保存结果 → 计算 MPJPE/PA-MPJPE/PVE
```

### 其他数据集

```
data/coco/ — COCO (2D 评测)
data/mpii/ — MPII (2D 评测)
data/aic/ — AI Challenger (2D 评测)
data/egohumans/ — EgoHumans (3D 评测, 需 undistort)
data/harmony4d/ — Harmony4D (3D 评测, 需 undistort)
data/egoexo4d/ — EgoExo4D (3D 评测)
data/sa1b/ — SA-1B (大规模分割数据)
```

---

## 6.7 复现实验的建议配置

### 复现原始基线 (Table 1, Original 3DB)

```python
# sam_3d_body_estimator.py
detector_type = "vitdet_h"
segmentor_type = "sam2.1_hiera_large"
fov_type = "moge2_vitl"
inference_type = "full"  # body + hand

# sam3d_body.py
backbone = "vit_h"
decoder_depth = 5
do_interm_preds = True
keypoint_token_update = True
DO_HAND_DETECT_TOKENS = True
DO_KEYPOINT_TOKENS = True
DO_KEYPOINT3D_TOKENS = True
```

### 复现 Fast 版本 (Table 1, Fast 3DB)

```python
# 需要修改的部分:
detector_type = "yolo11_pose"      # 新增
segmentor_type = None              # 不需要
fov_type = "moge2_vitl"           # 或使用 EXIF
inference_type = "body"           # 跳过手部

# sam3d_body.py
backbone = None                    # 跳过 ViT encoder
decoder_depth = 1                  # 仅第 5 层 (S={5})
do_interm_preds = False
keypoint_token_update = False
DO_HAND_DETECT_TOKENS = False
DO_KEYPOINT_TOKENS = False        # YOLO 提供 2D kps
DO_KEYPOINT3D_TOKENS = False

# 新增模块
neural_kinematic_mlp = True       # MHR→SMPL MLP
kinematic_prior_mlp = True        # 姿态去噪
```

> **注意**：Fast 版本的大部分配置在当前代码中没有对应实现，需要新增代码。

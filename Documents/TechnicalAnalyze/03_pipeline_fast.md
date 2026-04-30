# 3. 推理管线架构分析 — Fast SAM 3D Body 加速管线

> 对应论文 §3.2 Acceleration

## 3.1 Fast SAM 3D Body 加速管线总览

Fast 版本通过五个维度的算法重构，将原始 3DB 管线从串行执行转变为批处理并行执行：

```
原始 3DB:                          Fast 3DB:
                                  
Detect → body bbox only           Detect → body + hand bbox (YOLO11-Pose)
       ↓                                    ↓
Enc(body) ×1                     Enc([body, L_hand, R_hand]) ×1 batched
       ↓                                    ↓
BodyDec → hand bbox              BodyDec (pruned interm preds)
       ↓                                    ↓
Enc(L_hand) ×1 (CPU crop)        HandDec([L_feat, R_feat]) ×1 batched
Enc(R_hand) ×1 (CPU crop)                
       ↓                                    ↓
HandDec(L) ×1                    Merge
HandDec(R) ×1                           ↓
       ↓                               (skip keypoint prompt refinement)
Merge                                   ↓
       ↓                            MHR FK
Refine (2nd BodyDec pass)               ↓
       ↓                            f_ω(x): MLP → SMPL (if needed)
MHR FK
       ↓
IterFit → SMPL (hundreds of steps)
```

---

## 3.2.1 空间先验解耦（Decoupled Spatial Priors）

### 论文方案

用 YOLO11-Pose 替代串行手部检测：

```python
# 论文 §3.2: 手部 bbox 从粗略 2D 关键点解析推导
b_hand = (x_w - s/2, y_w - s/2, x_w + s/2, y_w + s/2)
# (x_w, y_w) = 预测的腕部关键点位置
# s = min(w_body, h_body) / α  (α 为缩放因子)
```

**核心思想**：高容量的下游 decoder 对小幅空间偏移具有鲁棒性，粗略关键点只需提供可靠的边界区域即可。

### 代码现状

- ✅ `tools/build_detector.py` 已支持 `sam3` 后端，但**未实现** YOLO11-Pose
- ❌ 无从 2D 关键点解析推导手部 bbox 的逻辑
- ❌ 当前手部 bbox 完全依赖 body decoder 的 `hand_box_embedding` + `bbox_embed` 输出

### 实现路径评估

需要新增：
1. YOLO11-Pose 集成（`ultralytics` 包）
2. 腕部关键点 → 手部 bbox 的解析推导函数
3. 修改 `SAM3DBodyEstimator.process_one_image()` 以同时获取 body + hand bbox

---

## 3.2.2 静态图重整（Static Graph Reformulation）

### 论文方案

将动态执行图重构为确定性静态图，支持两种编译策略：

| 策略 | 适用场景 | 实现 |
|------|---------|------|
| TensorRT FP16 | NVIDIA GPU | backbone + FOV estimator 编译为 TRT engine |
| torch.compile + CUDA Graph | 通用 PyTorch | 捕获 forward pass 为 CUDA Graph，确定性回放 |

### 代码现状

- ❌ 无 TensorRT 集成
- ❌ 无 torch.compile 调用
- ❌ 无 CUDA Graph 捕获
- ⚠️ 代码中存在动态控制流（`do_interm_preds` 分支），需要固定为静态才能编译

### 实现路径评估

需要：
1. 固定 `do_interm_preds` 层选择集合 S
2. 移除数据依赖的条件分支
3. 为 backbone 添加 TensorRT 导出脚本
4. 为 decoder 添加 `torch.compile` 包装

---

## 3.2.3 计算感知解码器剪枝（Compute-Aware Decoder Pruning）

### 论文方案

三项剪枝策略：

**A. 中间预测层选择**（Table 3）：

| 配置 | MPJPE↓ | PA↓ | PVE↓ | FPS↑ |
|------|--------|-----|------|------|
| {0,1,2} (推荐) | 58.96 | 31.33 | 69.28 | 7.18 |
| {0,1,2,3,4} (全) | 58.89 | 31.21 | 69.21 | 6.88 |
| ∅ (无) | 60.73 | 33.38 | 72.41 | 8.19 |

**B. 禁用关键点提示精调**（Table 6）：

| 配置 | MPJPE↓ | FPS↑ |
|------|--------|------|
| KP Prompt OFF | 58.96 | 7.18 |
| KP Prompt ON | 58.92 | 6.88 |

**C. 禁用姿态修正**（Table 6）：

| 配置 | MPJPE↓ | FPS↑ |
|------|--------|------|
| Correctives OFF | 58.96 | 7.18 |
| Correctives ON | 58.04 | 6.88 |

### 代码现状

- ✅ `PromptableDecoder` 已有 `do_interm_preds` 和 `keypoint_token_update` 开关
- ✅ `SAM3DBodyEstimator` 中 `inference_type` 参数可控制是否运行 hand decoder
- ⚠️ 中间预测层选择集合 S 未作为配置项暴露（当前是全量执行）
- ❌ 无"禁用关键点提示精调"的配置开关（`run_keypoint_prompt()` 在 `inference_type="full"` 时总是执行）
- ❌ "Correctives" 功能在代码中未见独立开关

### 实现路径评估

需要修改：
1. 在 `PromptableDecoder.forward()` 中添加 `interm_pred_layers: Set[int]` 参数
2. 在 `SAM3DBodyEstimator` 中添加 `skip_kp_prompt: bool` 参数
3. 在 `MHRHead.mhr_forward()` 中定位并隔离 correctives 逻辑

---

## 3.2.4 管线重构（Pipeline Restructuring）

### 论文方案

**A. GPU 原生手部裁剪**：
- 构造不同iable sampling grid 从解析推导的 bbox 坐标
- 双线性插值在 GPU 上一次完成所有裁剪
- 消除 GPU→CPU→GPU 数据传输

**B. Body-Hand 批处理**（Eq.2）：
```
[F_body, F_L, F_R] = Enc([I_body, I_L, I_R])
```
3 次 backbone forward → 1 次批处理 forward

**C. Hand Decoder 批处理**：
```
[θ̂_L, θ̂_R] = HandDec([F_L, F_R])
```
2 次 hand decoder forward → 1 次批处理

**D. 算子级优化**：
- 替换通用库调用为图可追踪的专用算子
- 向量化逐关节循环
- 缓存频繁访问的参数

### 代码现状

- ❌ 手部裁剪使用 `prepare_batch()` (NumPy/CV2)，在 CPU 上执行
- ❌ Body 和 Hand 使用独立 backbone forward pass
- ❌ 左右手 Hand Decoder 串行执行
- ⚠️ 代码中 `forward_step()` 已支持 `decoder_type` 切换，但未支持批处理模式

### 实现路径评估

需要：
1. 实现 GPU-native crop: `F.affine_grid()` + `F.grid_sample()`
2. 修改 `SAM3DBodyEstimator.process_one_image()` 为批处理模式
3. 合并 body + hand 为单次 backbone forward
4. 合并左右手为单次 hand decoder forward

---

## 3.2.5 神经运动学投影（Neural Kinematic Projection）

### 论文方案

Eq.(4): `Θ̂_smpl = f_ω(x)`

**架构**：
```
输入: MHR 顶点 V_mhr (18,439 个)
  │
  ▼
重心坐标投影: Ṽ = B(V_mhr)  → R^{6890×3}  (预计算重心坐标, 单次矩阵乘法)
  │
  ▼
子采样: 1,500 个顶点 → 去质心 → 展平 → x ∈ R^{4500}
  │
  ▼
MLP: 4500 → 512 → 256 → 76  (ReLU, ~2.5M 参数)
  │
  ▼
输出: Θ̂_smpl = (global_rot:3, body_pose:63, shape:10)
```

**训练**：
- 训练对: `{(x_i, Θ*_i)}` 由原始迭代拟合生成
- 损失: `L_convert = λ_v ‖V̂_smpl - Ṽ‖₁ + λ_reg ‖Θ̂ - Θ*‖₂²`

**运动学先验精调 MLP**：
- 在 AMASS 干净动作捕捉序列上训练
- 输入: 预测的 SMPL body pose → 输出: 去噪后的自然姿态
- 延迟: ~0.1ms
- 用于消除解剖学上不可行的姿态

### 代码现状

- ❌ MHR-to-SMPL 转换完全未实现（无论迭代还是 MLP）
- ❌ 无重心坐标投影
- ❌ 无神经运动学投影 MLP
- ❌ 无运动学先验精调 MLP

### 实现路径评估

需要全新实现：
1. **重心坐标投影模块**：预计算 SMPL→MHR 三角形映射 + 重心权重
2. **MHR-to-SMPL MLP**：3 层 MLP + 训练脚本
3. **训练数据生成**：运行原始迭代拟合在大量 3DB 预测上
4. **运动学先验 MLP**：在 AMASS 上训练去噪网络

---

## 加速效果汇总

| 加速技术 | 加速倍数 | 代码状态 |
|---------|---------|---------|
| 空间先验解耦 (消除串行依赖) | ~2× (3 backbone → 1) | ❌ 未实现 |
| 静态图编译 (TRT/compile) | ~1.5-2× | ❌ 未实现 |
| 解码器剪枝 (IntermPred + KP prompt) | ~1.2× | ⚠️ 部分可配置 |
| 管线重构 (GPU crop + batch) | ~1.5× | ❌ 未实现 |
| 神经运动学投影 (MLP 替代迭代) | ~10,000× (转换阶段) | ❌ 未实现 |
| **端到端** | **~8-11×** | |

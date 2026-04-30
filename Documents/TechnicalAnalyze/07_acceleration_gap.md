# 7. 加速差距分析

> 本文档详细分析当前代码实现与 Fast SAM 3D Body 论文之间的差距，包括缺失的模块、需要修改的模块、以及实现建议。

## 7.1 差距总览

| 加速方法 | 论文加速比 | 代码实现状态 | 实现难度 | 预计工作量 |
|---------|-----------|-------------|---------|-----------|
| ① Spatial Prior Decoupling | ~8× (检测) | ❌ 未实现 | ⭐⭐ 中等 | 2-3 天 |
| ② Static Graph Reformulation | ~1.5× (推理) | ❌ 未实现 | ⭐ 简单 | 1 天 |
| ③ Compute-Aware Decoder Pruning | ~2.5× (decoder) | ⚠️ 部分支持 | ⭐⭐ 中等 | 2-3 天 |
| ④ Pipeline Restructuring | ~1.3× (系统) | ❌ 未实现 | ⭐⭐⭐ 困难 | 5-7 天 |
| ⑤ Neural Kinematic Projection | ~60× (FK→MLP) | ❌ 未实现 | ⭐⭐ 中等 | 3-5 天 |

**总计差距**：当前代码无法复现任何 Fast 版本的加速效果。

---

## 7.2 方法①：Spatial Prior Decoupling

### 论文描述

用 YOLO11-Pose 替代 ViTDet-H + SAM2.1 + 手动关键点采样：
- ViTDet-H (~300ms) + SAM2.1 (~100ms) → YOLO11-Pose (~5ms)
- YOLO 同时输出 bbox + 2D keypoints + segmentation mask
- 2D 关键点直接替代 KeypointSamplerV1 的迭代采样

### 当前代码差距

1. **无 YOLO11-Pose 集成**
   - `tools/build_detector.py` 仅支持 ViTDet-H 和 SAM3
   - 需要新增 YOLO 后端

2. **关键点来源不同**
   - 当前：Decoder 迭代预测 → 反馈 → 重采样 (KeypointSamplerV1)
   - Fast：YOLO 直接输出 2D 关键点 → 作为固定输入

3. **分割来源不同**
   - 当前：SAM2.1 输出精细 mask → 用于条件编码
   - Fast：YOLO 输出粗分割 → 需要评估是否足够

### 实现路径

```
1. 新增 YOLO11-Pose 推理模块 (tools/build_yolo.py)
2. 修改 SAM3DBodyEstimator 支持 YOLO 检测模式
3. 将 YOLO 输出的 2D keypoints 转换为 PromptEncoder 输入格式
4. 评估 YOLO 分割 vs SAM2.1 分割对精度的影响
5. 修改 forward_decoder 跳过 KeypointSamplerV1
```

### 关键风险

- YOLO 2D 关键点精度 < Decoder 迭代优化精度 → 可能影响 3D 精度
- YOLO 分割 mask 质量 < SAM2.1 → 可能影响条件编码质量
- YOLO 不输出 FOV → 仍需 MoGe2 或 EXIF

---

## 7.3 方法②：Static Graph Reformulation

### 论文描述

将动态计算图转为静态图，消除 Python 开销：
- `torch.compile` (PyTorch 2.x)
- TensorRT 导出
- 消除动态 shape 和控制流

### 当前代码差距

1. **动态 shape**
   - `forward_decoder()` 中 token 数量取决于配置
   - 手部 token 有条件添加
   - 关键点 token 更新函数有条件调用

2. **控制流**
   - `run_inference()` 的 9 步管线中有多个 if-else
   - 手部有效性检查 (4 个条件)
   - Wrist IK 有条件执行

3. **数据依赖的 shape**
   - 不同图像尺寸 → 不同 feature map 尺寸
   - 不同 bbox → 不同裁剪尺寸

### 实现路径

```
1. 固定输入尺寸 (1024×1024 或 518×518)
2. 固定 token 序列 (移除条件分支)
3. 将 hand 处理分离为独立模型
4. 使用 torch.compile 编译主干
5. (可选) 导出 TensorRT engine
```

### 关键风险

- `torch.compile` 与 TorchScript MHR 模型的兼容性
- Flash Attention 在 TRT 中的支持
- 固定 token 序列可能增加无用计算

---

## 7.4 方法③：Compute-Aware Decoder Pruning

### 论文描述

1. 层选择集合 S：不运行所有 5 层 decoder，只运行 S 中的层
2. 禁用 KP prompt token：移除 70×2=140 个关键点 token
3. 两者组合：减少计算量 ~2.5×

### 当前代码支持

⚠️ **部分支持**：

```python
# PromptableDecoder 中已有的配置:
do_interm_preds = True     # 控制中间层预测
keypoint_token_update = True  # 控制关键点 token 更新

# SAM3DBody 中已有的配置:
DO_HAND_DETECT_TOKENS = True   # 控制手部检测 token
DO_KEYPOINT_TOKENS = True      # 控制关键点 token
DO_KEYPOINT3D_TOKENS = True    # 控制 3D 关键点 token
```

### 缺失部分

1. **层选择集合 S**
   - 当前 `PromptableDecoder` 按顺序运行所有层
   - 需要修改为支持跳层计算
   - 例如 S={5}: 直接用 init_embedding 作为前 4 层的输入，仅运行第 5 层

2. **中间层预测的灵活控制**
   - 当前 `do_interm_preds` 是全局开关
   - 需要按层控制：仅 S 中的层执行预测

3. **Token 数量优化**
   - 当前即使 `DO_KEYPOINT_TOKENS=False`，token 序列结构仍预留位置
   - 需要彻底移除未使用的 token

### 实现路径

```
1. 修改 PromptableDecoder.forward() 支持层选择集合 S
2. 修改 token 构建逻辑，完全移除未使用的 token
3. 添加配置: decoder_layer_set = [5]  # 或 [3, 5]
4. 修改 do_interm_preds 为按层配置
5. 评估不同 S 配置的精度-速度 trade-off
```

### 关键风险

- 仅使用最后一层 (S={5}) 精度下降较大 (~2-5mm MPJPE)
- 跳层后 cross-attention 的 KV cache 需要重新处理
- 中间层预测用于 keypoint token 更新，跳层后需要替代方案

---

## 7.5 方法④：Pipeline Restructuring

### 论文描述

1. GPU 上执行裁剪和缩放（避免 CPU↔GPU 数据传输）
2. 多人场景批量推理
3. 去除串行依赖，并行化独立步骤

### 当前代码差距

1. **CPU 裁剪**
   - `prepare_batch()` 在 CPU 上执行裁剪和缩放
   - 图像数据在 CPU↔GPU 间多次传输

2. **单人处理**
   - `process_one_image()` 每次仅处理一个人
   - 多人场景串行处理

3. **串行手部处理**
   - 左手 → 右手，串行执行
   - `forward_decoder_hand()` 调用两次

4. **串行 Keypoint Prompt**
   - `run_keypoint_prompt()` 在 body 解码后串行执行

### 实现路径

```
1. GPU Crop: 实现 CUDA kernel 或使用 kornia 进行 GPU 仿射变换
2. Batch 推理: 修改 SAM3DBody 支持批量输入
3. 并行手部: 将左手和右手 forward_decoder_hand 合并为单次 batch 调用
4. 移除串行 Keypoint Prompt (与方法①协同)
```

### 关键风险

- GPU crop 需要处理非对齐的 bbox，实现复杂
- Batch 推理需要统一 feature map 尺寸
- 并行手部处理需要不同的 bbox 处理逻辑
- 最大加速来自整体架构重组，局部优化收益有限

---

## 7.6 方法⑤：Neural Kinematic Projection

### 论文描述

用 MLP 替代迭代 MHR-to-SMPL 转换：
- 从 MHR 顶点 (18,439) 子采样 1,500 个
- 去质心 → 展平 → 4,500 维输入
- 3 层 MLP: 4500 → 512 → 256 → 76 (SMPL 参数)
- 附加运动学先验 MLP: 63 维 → 63 维

### 当前代码差距

1. **MHR-to-SMPL 完全缺失**
   - 当前代码不执行 MHR→SMPL 转换
   - 直接使用 MHR 顶点和关键点

2. **无 MLP 投影模块**
   - 需要从零实现

3. **无训练数据**
   - MLP 需要大量 (MHR params, SMPL params) 配对训练
   - 需要生成训练数据

### 实现路径

```
1. 数据准备:
   - 收集/生成大量 (MHR→FK→verts, SMPL params) 配对
   - 子采样 MHR 顶点，计算重心坐标映射
   
2. 模型实现:
   - NeuralKinematicProjection(nn.Module):
     - sub_sampler: 预计算子采样索引
     - mlp: Linear(4500, 512) → GELU → Linear(512, 256) → GELU → Linear(256, 76)
     - kinematic_prior: Linear(63, 128) → GELU → Linear(128, 63)
   
3. 训练:
   - Loss: L2 on SMPL params + L2 on resulting vertices
   - 数据增强: 不同姿态、形状、尺度
   
4. 集成:
   - 在 mhr_forward() 后调用 MLP
   - 输出: SMPL body_pose (63) + shape (10) + scale (3)
```

### 关键风险

- MLP 精度取决于训练数据量和质量
- 子采样 1,500/18,439 顶点可能丢失细节
- 运动学先验 MLP 可能过度平滑姿态
- 需要确保 MLP 输出的 SMPL 参数的有效性

---

## 7.7 实现优先级建议

### Phase 1: 低成本高收益 (1-2 周)

1. **方法② Static Graph**: `torch.compile` 最容易实现，且不影响精度
2. **方法③ Decoder Pruning**: 已有部分配置支持，修改量较小

### Phase 2: 中等成本中等收益 (2-3 周)

3. **方法① YOLO11-Pose**: 需要新增模块，但开源 YOLO 实现丰富
4. **方法⑤ Neural Kinematic**: 需要训练 MLP，但模块独立

### Phase 3: 高成本高收益 (3-4 周)

5. **方法④ Pipeline Restructure**: 需要重构整个推理管线

### 并行开发建议

- 方法① 和 方法⑤ 可以并行开发（无依赖）
- 方法② 应先于 方法④（静态图是管线重构的前提）
- 方法③ 应先于 方法④（Decoder 优化影响管线设计）

---

## 7.8 预期加速效果

### 各方法叠加 (基于论文数据)

```
原始管线:                    ~700ms (代码实测可能更高)
+ 方法② Static Graph:       ~470ms (-33%)
+ 方法③ Decoder Pruning:    ~350ms (-25%)
+ 方法① YOLO11-Pose:        ~50ms  (-86%)
+ 方法④ Pipeline Restructure: ~38ms (-24%)
+ 方法⑤ Neural Kinematic:    ~19ms  (-50%)
                              ------
最终:                        ~19ms  (36.8× 加速)
```

> 注：论文报告 10.9× 加速（从 ~230ms 到 ~21ms），差距来自原始基线的计时方式不同。

### 精度影响

```
方法②: 无精度损失
方法③: ~1-3mm MPJPE 增加
方法①: ~2-5mm MPJPE 增加 (YOLO 检测精度)
方法④: 无精度损失
方法⑤: ~0.5-1mm MPJPE 增加 (MLP 近似误差)
总计:  ~3-9mm MPJPE 增加
```

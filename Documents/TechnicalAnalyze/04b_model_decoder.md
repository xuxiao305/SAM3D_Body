# 4b. 核心模型深度分析 — PromptableDecoder

> 对应代码：`models/decoders/promptable_decoder.py` — `PromptableDecoder` 类

## 4b.1 架构总览

`PromptableDecoder` 是一个基于交叉注意力的 Transformer 解码器，借鉴自 SAM (Segment Anything Model) 的解码器设计。

```
PromptableDecoder(nn.Module)
├── layers: nn.ModuleList[TransformerDecoderLayer] × depth
├── norm_final: LayerNorm
├── do_interm_preds: bool      # 是否执行中间预测
├── do_keypoint_tokens: bool   # 是否有关键点 tokens
├── keypoint_token_update: bool # 是否更新关键点 tokens
└── frozen: bool
```

---

## 4b.2 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `dims` | int | Token 投影维度（默认 1024） |
| `context_dims` | int | 图像上下文特征维度（= backbone embed_dims = 1280） |
| `depth` | int | Transformer 层数（默认 5） |
| `num_heads` | int | 注意力头数（默认 8） |
| `head_dims` | int | 每头维度（默认 64） |
| `mlp_dims` | int | FFN 隐层维度（默认 1024） |
| `layer_scale_init_value` | float | LayerScale 初始值（默认 0） |
| `drop_rate` | float | Dropout 概率 |
| `attn_drop_rate` | float | 注意力 Dropout |
| `drop_path_rate` | float | DropPath 概率 |
| `ffn_type` | str | FFN 类型: "origin" 或 "swiglu_fused" |
| `enable_twoway` | bool | 双向 Transformer（SAM 风格） |
| `repeat_pe` | bool | 每层重复 PE（LaPE 风格） |
| `frozen` | bool | 冻结所有参数 |
| `do_interm_preds` | bool | 中间预测开关 |
| `do_keypoint_tokens` | bool | 关键点 tokens 开关 |
| `keypoint_token_update` | bool | 关键点 token 更新开关 |

---

## 4b.3 Forward 流程详解

```python
def forward(
    token_embedding,      # [B, N, D] — 所有输入 tokens
    image_embedding,      # [B, C, H, W] — 图像特征
    token_augment,        # [B, N, D] — token 位置增强
    image_augment,        # [B, C, H, W] — 图像位置增强
    token_mask,           # [B, N] — token 掩码
    token_to_pose_output_fn,  # 中间预测回调
    keypoint_token_update_fn, # 关键点更新回调
):
```

### 核心循环

```
# 图像特征 reshape: [B, C, H, W] → [B, H*W, C]
image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
image_augment = image_augment.flatten(2).permute(0, 2, 1)

for layer_idx, layer in enumerate(self.layers):
    # 1. Transformer 层前向
    token_embedding, image_embedding = layer(
        token_embedding, image_embedding,
        token_augment, image_augment, token_mask
    )
    
    # 2. 中间预测 (如果 do_interm_preds 且不是最后一层)
    if do_interm_preds and layer_idx < depth - 1:
        curr_pose_output = token_to_pose_output_fn(
            norm_final(token_embedding),
            prev_pose_output,
            layer_idx
        )
        all_pose_outputs.append(curr_pose_output)
        
        # 3. 关键点 token 更新 (如果 keypoint_token_update)
        if keypoint_token_update:
            token_embedding, token_augment, _, _ = \
                keypoint_token_update_fn(token_embedding, token_augment, 
                                         curr_pose_output, layer_idx)

# 最终归一化 + 预测
out = norm_final(token_embedding)
final_pose_output = token_to_pose_output_fn(out, all_pose_outputs[-1], depth-1)
all_pose_outputs.append(final_pose_output)

return out, all_pose_outputs
```

### 关键设计决策

1. **中间预测在归一化后执行**：`norm_final(token_embedding)` → 预测，而非直接用未归一化的 token
2. **中间预测跳过最后一层**：`layer_idx < depth - 1`，最后一层在循环外单独处理
3. **关键点更新在中间预测后执行**：先预测，再基于预测结果更新 token

---

## 4b.4 TransformerDecoderLayer 结构

每层包含三个子模块：

```
TransformerDecoderLayer
├── Self-Attention: token → token (带可选 PE)
│   ln1 → q=k=v=ln1(x) [+ x_pe] → self_attn → residual
│
├── Cross-Attention: token → image (带可选 PE)
│   ln2_1 → q=ln2_1(x) [+ x_pe]
│   ln2_2 → k=v=ln2_2(context) [+ context_pe]
│   → cross_attn → residual
│
├── FFN: token → token
│   ln3 → ffn → residual
│
└── [Optional] Two-way Attention: image → token
    ln4_1 → q=ln4_1(context) [+ context_pe]
    ln4_2 → k=v=ln4_2(x) [+ x_pe]
    → cross_attn_2 → residual (on context)
```

### 位置编码策略

| 策略 | 说明 |
|------|------|
| `repeat_pe=False` (默认) | PE 仅在第一层添加 (`skip_first_pe=True`) |
| `repeat_pe=True` (LaPE) | PE 在每层重新添加，通过 `ln_pe_1` / `ln_pe_2` 处理 |

### Token Mask 处理

```python
# 构造注意力掩码
if x_mask is not None:
    attn_mask = x_mask[:, :, None] @ x_mask[:, None, :]
    attn_mask.diagonal(dim1=1, dim2=2).fill_(1)  # 防止 NaN
    attn_mask = attn_mask > 0
```

掩码用于 self-attention，防止无效 token 参与注意力计算。

---

## 4b.5 与 Fast 版本的关系

### 可直接利用的配置开关

| 开关 | 当前作用 | Fast 对应 |
|------|---------|----------|
| `do_interm_preds` | 控制中间预测 | 需增加 `interm_pred_layers: Set[int]` |
| `keypoint_token_update` | 控制关键点更新 | 保持开启（在选定层内） |
| `frozen` | 冻结参数 | Fast 为 training-free |

### 需要修改的部分

1. **中间预测层选择**：当前是全量执行（所有层都做 IntermPred），Fast 版本需要 `S = {0, 1, 2}`（仅前 3 层）
   - 修改点：`if self.do_interm_preds and layer_idx < len(self.layers) - 1:` → `if layer_idx in self.interm_pred_layers:`

2. **静态图兼容**：`keypoint_token_update_fn` 中的条件分支和动态更新需要固定
   - `if layer_idx == len(self.decoder.layers) - 1: return` 这类数据依赖控制流需要消除

3. **手部特征拼接**：当前 `forward()` 支持 `hand_embeddings` / `hand_augment` 参数，可将手部特征与图像特征拼接
   - 已有基础设施，但未被 `forward_decoder()` 使用

---

## 4b.6 参数量估算

以默认配置（depth=5, dims=1024, context_dims=1280）为例：

| 组件 | 参数量估算 |
|------|----------|
| Self-Attention (×5) | ~5 × 3 × 1024² ≈ 15.7M |
| Cross-Attention (×5) | ~5 × (1024×1280 + 1280×1024 + 1024×1024) ≈ 33.5M |
| FFN (×5) | ~5 × 2 × 1024 × 1024 ≈ 10.5M |
| norm_final | ~2K |
| **总计** | **~60M** |

---

## 4b.7 Attention 模块详解

`TransformerDecoderLayer` 使用的 `Attention` 模块（来自 `transformer.py`）：

```python
class Attention:
    q_proj = nn.Linear(query_dims, embed_dims, bias=qkv_bias)
    k_proj = nn.Linear(key_dims, embed_dims, bias=qkv_bias)
    v_proj = nn.Linear(value_dims, embed_dims, bias=qkv_bias)
    proj = nn.Linear(embed_dims, query_dims, bias=proj_bias)
    # 使用 F.scaled_dot_product_attention (PyTorch 2.0+)
```

**与 ViT Attention 的区别**：
- ViT 使用单一 `qkv` 线性层（自注意力），Decoder 使用独立的 `q/k/v` 投影（交叉注意力）
- Decoder 支持 `query_dims ≠ key_dims ≠ value_dims` 的非对称维度

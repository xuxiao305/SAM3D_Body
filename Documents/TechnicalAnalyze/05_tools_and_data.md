# 5. 工具链与数据处理

> 对应代码：
> - `tools/build_detector.py` — HumanDetector
> - `tools/build_sam.py` — HumanSegmentor
> - `tools/build_fov_estimator.py` — FOVEstimator
> - `sam_3d_body/data/` — 数据变换与批处理
> - `sam_3d_body/visualization/` — 可视化模块
> - `sam_3d_body/utils/` — 工具函数

## 5.1 HumanDetector

### 架构

```python
class HumanDetector:
    """人体检测器，支持两种后端"""
    
    def __init__(self, detector_type="vitdet_h", device="cuda"):
        if detector_type == "vitdet_h":
            # ViTDet-H via Detectron2
            self.cfg = cascade_mask_rcnn_vitdet_h_75ep()
            self.predictor = DefaultPredictor(self.cfg)
        elif detector_type == "sam3":
            # SAM3 text-prompted detection
            self.sam3 = SAM3Model()
    
    def detect(self, image):
        """返回: List[Dict] with 'bbox', 'score', 'category_id'"""
```

### ViTDet-H 配置

来自 `cascade_mask_rcnn_vitdet_h_75ep.py`：

```
Backbone: ViT-H/16 (3.1B 参数)
检测头: Cascade Mask R-CNN (3 stages)
  - Stage 1: IoU threshold 0.5
  - Stage 2: IoU threshold 0.6
  - Stage 3: IoU threshold 0.7
训练: 75 epochs on COCO
输入尺寸: 1024×1024
```

### SAM3 文本检测

```python
# 使用 SAM3 的文本提示功能
# prompt = "person"
# 直接输出分割 mask + bbox
```

---

## 5.2 HumanSegmentor

### 架构

```python
class HumanSegmentor:
    """人体分割器"""
    
    def __init__(self, segmentor_type="sam2.1_hiera_large", device="cuda"):
        if segmentor_type == "sam2.1_hiera_large":
            self.model = SAM2Predictor(...)
        elif segmentor_type == "sam3":
            self.model = SAM3Model(...)
    
    def segment(self, image, bboxes):
        """输入: image + bboxes → 输出: masks"""
```

### SAM2.1 Hiera Large

```
Image Encoder: Hiera Large (196M 参数)
  - Multi-scale features
  - 输入: 1024×1024
  - 输出: 多尺度 feature maps

Memory Attention: 用于视频分割
Prompt Encoder: 点/框/mask 提示
Mask Decoder: 4 层 Transformer + MLP
```

### 分割流程

```
1. image → SAM2.1 encoder → feature map
2. bbox → prompt encoder → prompt embedding
3. feature + prompt → mask decoder → mask
4. mask → largest connected component → 最终 mask
```

---

## 5.3 FOVEstimator

### 架构

```python
class FOVEstimator:
    """视场角估计器"""
    
    def __init__(self, fov_type="moge2_vitl", device="cuda"):
        if fov_type == "moge2_vitl":
            self.model = MoGe2(...)
    
    def estimate(self, image):
        """返回: focal_length (h_focal, v_focal)"""
```

### MoGe2 ViT-L

```
Backbone: ViT-Large
任务: 单目深度 + FOV 估计
输出:
  - depth_map: [H, W]
  - focal_length: (h_focal, v_focal)
  - camera_intrinsics: [3, 3]
```

### FOV 在管线中的作用

```
FOV → cam_int (相机内参矩阵)
    → CameraEncoder (射线方向)
    → perspective_projection (3D→2D 投影)
    → condition_info (CLIFF 条件)
```

**关键**：FOV 的准确性直接影响 3D→2D 投影的精度，进而影响 MHR 参数优化的收敛。

---

## 5.4 数据变换

### `transforms/common.py`

```python
class Compose:
    """组合多个变换"""

class ToTensor:
    """HWC uint8 → CHW float32"""

class Normalize:
    """ImageNet 标准化: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]"""
```

### `transforms/bbox_utils.py`

```python
def bbox_to_mask(bbox, H, W):
    """bbox → 二值 mask"""

def expand_bbox(bbox, scale, H, W):
    """扩展 bbox: (x1,y1,x2,y2) → 放大 scale 倍"""

def square_bbox(bbox):
    """将 bbox 变为正方形 (取较长边)"""
```

---

## 5.5 批处理

### `data/utils/prepare_batch.py`

```python
def prepare_batch(image, bbox, mask, cam_int, ...):
    """将检测/分割结果打包为模型输入 batch
    
    流程:
    1. 从 bbox 计算裁剪区域
    2. 裁剪图像 + mask
    3. 计算 condition_info (CLIFF)
    4. 计算射线条件
    5. 打包为 dict
    """
```

### `data/utils/io.py`

```python
def load_image(path):
    """加载图像 → float32 [0, 1]"""

def load_keypoints(path):
    """加载关键点标注 → [N, 3] (x, y, conf)"""

def load_segmentation(path):
    """加载分割 mask → bool [H, W]"""
```

---

## 5.6 可视化模块

### `visualization/renderer.py`

```python
class Renderer:
    """3D 渲染器 (基于 pyrender 或 Open3D)"""
    
    def render(self, vertices, faces, image, camera):
        """将 3D mesh 渲染到 2D 图像上"""
    
    def render_skeleton(self, joints, image):
        """渲染骨架"""
```

### `visualization/skeleton_visualizer.py`

```python
class SkeletonVisualizer:
    """2D/3D 骨架可视化"""
    
    # MHR70 关节连接定义
    MHR70_SKELETON = [
        (0, 1), (1, 2), ...  # 70 个关节的连接关系
    ]
    
    def draw_skeleton_2d(self, image, keypoints_2d, confidence):
        """在 2D 图像上绘制骨架"""
    
    def draw_skeleton_3d(self, keypoints_3d):
        """3D 骨架可视化"""
```

### `visualization/utils.py`

```python
def get_different_colors(n):
    """生成 n 个可区分的颜色"""

def draw_bbox(image, bbox, color, thickness):
    """绘制 bbox"""

def overlay_mask(image, mask, color, alpha):
    """半透明 mask 叠加"""
```

---

## 5.7 工具函数

### `utils/checkpoint.py`

```python
def load_checkpoint(model, checkpoint_path, device):
    """加载检查点，处理 key 前缀不匹配"""

def strip_prefix(state_dict, prefix):
    """移除 state_dict 中的前缀 (如 'module.')"""
```

### `utils/config.py`

```python
class Config:
    """配置管理，支持 YAML 和命令行覆盖"""
    
    def merge_from_list(self, cfg_list):
        """从命令行参数列表合并配置"""
```

### `utils/dist.py`

```python
def setup_distributed():
    """初始化分布式训练"""

def is_main_process():
    """是否为主进程"""
```

### `utils/logging.py`

```python
def setup_logging(name, output_dir):
    """配置日志"""
```

### `optim/fp16_utils.py`

```python
class NativeScalerWithGradNormCount:
    """AMP (混合精度) 梯度缩放器"""
    
    def __call__(self, loss, optimizer, clip_grad=None):
        """混合精度反向传播 + 梯度裁剪"""
```

---

## 5.8 与 Fast 版本的关系

### 被替代的模块

1. **HumanDetector (ViTDet-H)**: 被 YOLO11-Pose 替代
   - ViTDet-H: 3.1B 参数，~300ms 推理
   - YOLO11-Pose: ~50M 参数，~5ms 推理
   - YOLO 同时提供 bbox + 2D 关键点 + 分割 mask

2. **HumanSegmentor (SAM2.1)**: 被 YOLO11-Pose 的分割头替代
   - SAM2.1 Hiera Large: 196M 参数，~100ms 推理
   - YOLO 分割头: 内置于检测模型，~0ms 额外开销

3. **KeypointSamplerV1**: 不再需要迭代采样
   - YOLO 直接输出 2D 关键点
   - 不需要 decoder 反馈 → 重采样

### 可复用的模块

1. **FOVEstimator (MoGe2)**: Fast 版本仍需要 FOV 估计
   - 但 Fast 论文提到可以使用 EXIF 信息替代
   - 或训练一个轻量级 FOV 估计器

2. **数据变换**: bbox_utils, common transforms 通用
3. **可视化**: 渲染和骨架可视化通用
4. **工具函数**: checkpoint, config, logging 通用
5. **FP16 工具**: 混合精度训练/推理通用

### 需要新增的模块

1. **YOLO11-Pose 包装器**:
   - 输入: image
   - 输出: bbox + 2D keypoints + segmentation mask
   - 需要处理 YOLO 输出格式 → SAM3DBody 输入格式的转换

2. **GPU Crop Pipeline**:
   - 在 GPU 上执行裁剪和缩放
   - 避免数据在 CPU↔GPU 间来回传输
   - 与 YOLO 输出直接对接

3. **Batch Pipeline**:
   - 多人场景的批量推理
   - 当前代码仅支持单人的 `process_one_image()`

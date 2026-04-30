"""焦距扫描测试：对每张图用不同焦距推理，对比侧视图差异"""

import os
import sys
import numpy as np
import cv2
import torch

sys.path.insert(0, "/root/SAM3D_Body/Main")
os.chdir("/root/SAM3D_Body/Main")
os.environ["PYOPENGL_PLATFORM"] = "egl"

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together

CHECKPOINT_PATH = "/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR_PATH = "/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
IMAGE_FOLDER = "/root/SAM3D_Body/test_images"
OUTPUT_FOLDER = "/root/SAM3D_Body/output_focal_sweep"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}")

print("\n=== Loading SAM3D Body model ===")
model, model_cfg = load_sam_3d_body(CHECKPOINT_PATH, device=device, mhr_path=MHR_PATH)

estimator = SAM3DBodyEstimator(
    sam_3d_body_model=model,
    model_cfg=model_cfg,
    human_detector=None,
    human_segmentor=None,
    fov_estimator=None,
)

# 焦距倍数：相对于 max(H,W)
# 0.5 = 极广角 (20mm)
# 1.0 = 标准 (50mm)，ComfyUI 默认
# 1.4 = SAM3D 默认 √(H²+W²)
# 2.0 = 长焦 (100mm)
# 4.0 = 接近正交投影
FOCAL_MULTIPLIERS = [0.5, 1.0, 1.4, 2.0, 4.0]

images = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
print(f"\n=== {len(images)} images, {len(FOCAL_MULTIPLIERS)} focal values each ===")


def make_cam_int(focal, W, H):
    """构造 (1, 3, 3) 相机内参矩阵"""
    return torch.tensor([[
        [focal, 0, W / 2.0],
        [0, focal, H / 2.0],
        [0, 0, 1.0],
    ]], dtype=torch.float32)


for img_path in images:
    img_name = os.path.basename(img_path).rsplit(".", 1)[0]
    print(f"\n--- {img_name} ---")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"  Size: {w}x{h}")

    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
    rendered_per_focal = []

    for mult in FOCAL_MULTIPLIERS:
        focal = mult * max(w, h)
        cam_int = make_cam_int(focal, w, h)

        outputs = estimator.process_one_image(
            img_path,
            bboxes=bbox,
            cam_int=cam_int,
            use_mask=False,
        )

        rend = visualize_sample_together(img, outputs, estimator.faces)
        rend = rend.astype(np.uint8)

        # 在渲染图顶部打个标签
        label = f"focal={mult:.1f}xMax (px={int(focal)})"
        labeled = cv2.copyMakeBorder(rend, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(labeled, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 0), 3, cv2.LINE_AA)
        rendered_per_focal.append(labeled)
        print(f"  focal mult={mult}: ok")

    # 纵向拼接所有焦距的结果
    target_w = min(r.shape[1] for r in rendered_per_focal)
    resized = []
    for r in rendered_per_focal:
        if r.shape[1] != target_w:
            scale = target_w / r.shape[1]
            new_h = int(r.shape[0] * scale)
            r = cv2.resize(r, (target_w, new_h))
        resized.append(r)
    grid = np.vstack(resized)

    out_path = os.path.join(OUTPUT_FOLDER, f"{img_name}_focal_sweep.jpg")
    cv2.imwrite(out_path, grid)
    print(f"  Saved -> {out_path} (shape={grid.shape})")

print("\n=== DONE ===")

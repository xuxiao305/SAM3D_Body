"""最小推理测试：跳过 detector/segmentor/fov，用整张图 bbox 测试 SAM3D 主干 + 渲染"""

import os
import sys
import numpy as np
import cv2
import torch

# 设置路径
sys.path.insert(0, "/root/SAM3D_Body/Main")
os.chdir("/root/SAM3D_Body/Main")

# 需要在 import pyrender 之前设置
os.environ["PYOPENGL_PLATFORM"] = "egl"

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together

CHECKPOINT_PATH = "/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR_PATH = "/root/SAM3D_Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
IMAGE_FOLDER = "/root/SAM3D_Body/test_images"
OUTPUT_FOLDER = "/root/SAM3D_Body/output_dinov3_minimal"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n=== Loading SAM3D Body model ===")
model, model_cfg = load_sam_3d_body(CHECKPOINT_PATH, device=device, mhr_path=MHR_PATH)
print("Model loaded.")

# 不传 detector/segmentor/fov（保持 None）
estimator = SAM3DBodyEstimator(
    sam_3d_body_model=model,
    model_cfg=model_cfg,
    human_detector=None,
    human_segmentor=None,
    fov_estimator=None,
)

# 找到所有图
images = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
print(f"\n=== Found {len(images)} images ===")

for img_path in images:
    print(f"\n--- Processing {os.path.basename(img_path)} ---")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"Size: {w}x{h}")

    # 用整张图作为 bbox
    bbox = np.array([[0, 0, w, h]], dtype=np.float32)

    outputs = estimator.process_one_image(
        img_path,
        bboxes=bbox,
        use_mask=False,
    )
    print(f"Outputs: {len(outputs)} person(s)")

    # 渲染并保存
    rend = visualize_sample_together(img, outputs, estimator.faces)
    out_path = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(img_path).rsplit(".", 1)[0] + ".jpg"
    )
    cv2.imwrite(out_path, rend.astype(np.uint8))
    print(f"Saved -> {out_path}")

print("\n=== DONE ===")
print(f"Outputs in: {OUTPUT_FOLDER}")

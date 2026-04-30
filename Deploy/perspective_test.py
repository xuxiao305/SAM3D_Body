"""带透视的 3D 模型图焦距测试 - 已知 50mm 等效焦距"""

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
IMAGE_FOLDER = "/root/SAM3D_Body/test_images_persp"
OUTPUT_FOLDER = "/root/SAM3D_Body/output_perspective"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda")
model, model_cfg = load_sam_3d_body(CHECKPOINT_PATH, device=device, mhr_path=MHR_PATH)

estimator = SAM3DBodyEstimator(
    sam_3d_body_model=model,
    model_cfg=model_cfg,
    human_detector=None,
    human_segmentor=None,
    fov_estimator=None,
)


def make_cam_int(focal_px, W, H):
    return torch.tensor([[
        [focal_px, 0, W / 2.0],
        [0, focal_px, H / 2.0],
        [0, 0, 1.0],
    ]], dtype=torch.float32)


images = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for img_path in images:
    img_name = os.path.basename(img_path).rsplit(".", 1)[0]
    print(f"\n--- {img_name} ---")
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    print(f"  Size: {W}x{H}")

    bbox = np.array([[0, 0, W, H]], dtype=np.float32)

    # 50mm 等效焦距 = 50/36 × W ≈ 1.389 × W
    focal_50mm = 50.0 / 36.0 * W
    default_focal = (W**2 + H**2) ** 0.5

    test_focals = [
        ("50mm_correct", focal_50mm),
        ("default_sam3d", default_focal),
        ("0.5xW_wide", 0.5 * W),
        ("1.0xW", 1.0 * W),
        ("2.0xW_tele", 2.0 * W),
    ]

    rendered = []
    for label_short, focal in test_focals:
        cam_int = make_cam_int(focal, W, H)
        outputs = estimator.process_one_image(
            img_path, bboxes=bbox, cam_int=cam_int, use_mask=False
        )
        rend = visualize_sample_together(img, outputs, estimator.faces).astype(np.uint8)
        label = f"{label_short}: focal={int(focal)}px"
        labeled = cv2.copyMakeBorder(rend, 70, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(labeled, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 0, 0), 4, cv2.LINE_AA)
        rendered.append(labeled)
        print(f"  {label}: ok")

    target_w = min(r.shape[1] for r in rendered)
    resized = []
    for r in rendered:
        if r.shape[1] != target_w:
            scale = target_w / r.shape[1]
            new_h = int(r.shape[0] * scale)
            r = cv2.resize(r, (target_w, new_h))
        resized.append(r)
    grid = np.vstack(resized)

    out_path = os.path.join(OUTPUT_FOLDER, f"{img_name}_perspective_test.jpg")
    cv2.imwrite(out_path, grid)
    print(f"  Saved -> {out_path}")

print("\n=== DONE ===")

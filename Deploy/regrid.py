"""把垂直拼接的对比图重排为 3x2 网格便于查看"""
import cv2
import numpy as np
import os

src = "/root/SAM3D_Body/output_perspective/perspective_3d_perspective_test.jpg"
dst = "/root/SAM3D_Body/output_perspective/perspective_3d_grid.jpg"

img = cv2.imread(src)
H, W = img.shape[:2]
print(f"Original: {W}x{H}")
single_h = H // 5
parts = [img[i*single_h:(i+1)*single_h] for i in range(5)]
scale = 0.5
new_w = int(W * scale)
new_h = int(single_h * scale)
parts_s = [cv2.resize(p, (new_w, new_h)) for p in parts]
blank = np.full_like(parts_s[0], 255)
row1 = cv2.hconcat(parts_s[:3])
row2 = cv2.hconcat([parts_s[3], parts_s[4], blank])
out = cv2.vconcat([row1, row2])
cv2.imwrite(dst, out)
print(f"Saved: {dst}, shape={out.shape}")

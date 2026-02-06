"""1536x1024 템플릿을 200px 폭으로 리사이즈한 뒤 TM/Canny 점수 비교"""
import cv2
import numpy as np
import os
from detect_logos import detect_logo_template, detect_logo_canny

scene = cv2.imread("box_raw.png")

# 1536x1024 크기 템플릿만 추출
large_templates = {}
for f in sorted(os.listdir(".")):
    if not f.lower().endswith(".png"):
        continue
    if "box_raw" in f or "result_" in f:
        continue
    img = cv2.imread(f)
    if img is not None and max(img.shape[:2]) > 500:
        name = os.path.splitext(f)[0]
        large_templates[name] = img

TARGET_W = 200

print(f"{'Template':<20} {'Size':>12} {'TM':>8} {'TM_R':>8} {'Canny':>8} {'Can_R':>8}")
print("=" * 80)

for name, img in large_templates.items():
    # 원본 점수
    _, tm_score, _ = detect_logo_template(scene, img, threshold=0.0)
    _, cn_score, _ = detect_logo_canny(scene, img, threshold=0.0)

    # 200px 리사이즈 점수
    h, w = img.shape[:2]
    scale = TARGET_W / w
    resized = cv2.resize(img, (TARGET_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, tm_r_score, _ = detect_logo_template(scene, resized, threshold=0.0)
    _, cn_r_score, _ = detect_logo_canny(scene, resized, threshold=0.0)

    sz_orig = f"{w}x{h}"
    sz_resz = f"{resized.shape[1]}x{resized.shape[0]}"

    print(f"{name:<20} {sz_orig:>12} {tm_score:>8.4f} {tm_r_score:>8.4f} {cn_score:>8.4f} {cn_r_score:>8.4f}")
    print(f"  -> resized         {sz_resz:>12} {'':>8} {'FOUND' if tm_r_score>=0.75 else 'NOT':>8} {'':>8} {'FOUND' if cn_r_score>=0.38 else 'NOT':>8}")

print()
print("TM threshold=0.75 / Canny threshold=0.38")
print("TM_R, Can_R = 200px 리사이즈 후 점수")

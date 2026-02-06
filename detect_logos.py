import cv2
import numpy as np
import sys
import os
import re


def detect_logo(scene_img, template_img, method_name, min_match_count=10, ratio_thresh=0.75):
    """
    Feature matching으로 scene 이미지에서 template 로고를 찾는다.

    Returns:
        found (bool): 로고 검출 여부
        good_matches (int): 유효 매칭 수
        dst_corners (np.ndarray|None): 검출된 영역의 꼭짓점 4개 (scene 좌표)
    """
    # 그레이스케일 변환
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # ORB: 작은 템플릿 업스케일 + CLAHE 전처리
    if method_name == "ORB":
        scale = max(1, 300 // max(tmpl_gray.shape))
        if scale > 1:
            tmpl_gray = cv2.resize(tmpl_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        tmpl_gray = clahe.apply(tmpl_gray)
        scene_gray = clahe.apply(scene_gray)

    # 디텍터 생성
    if method_name == "SIFT":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method_name == "SURF":
        detector = cv2.xfeatures2d.SURF_create(400)
        norm_type = cv2.NORM_L2
    elif method_name == "ORB":
        detector = cv2.ORB_create(
            nfeatures=10000,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=10,
            patchSize=20,
        )
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # 키포인트 & 디스크립터 추출
    kp_tmpl, des_tmpl = detector.detectAndCompute(tmpl_gray, None)
    kp_scene, des_scene = detector.detectAndCompute(scene_gray, None)

    if des_tmpl is None or des_scene is None:
        return False, 0, None

    # 매칭
    if norm_type == cv2.NORM_HAMMING:
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
    else:
        bf = cv2.BFMatcher(norm_type)

    matches = bf.knnMatch(des_tmpl, des_scene, k=2)

    # Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    found = False
    dst_corners = None
    if len(good) >= min_match_count:
        # Homography로 기하학적 검증
        src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            inliers = mask.ravel().sum()
            if inliers >= min_match_count // 2:
                found = True
                h, w = tmpl_gray.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                dst_corners = cv2.perspectiveTransform(corners, M)

    return found, len(good), dst_corners


def detect_logo_template(scene_img, template_img, threshold=0.75):
    """
    멀티스케일 Template Matching으로 scene에서 template 로고를 찾는다.
    픽셀 수준 비교라 유사 로고(DolbyAtmos vs DolbyCinema 등)를 구분할 수 있다.

    Returns:
        found (bool): 로고 검출 여부
        score (float): 최대 매칭 점수 (0~1)
        dst_corners (np.ndarray|None): 검출 영역 꼭짓점 4개
    """
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_loc = None
    best_scale = 1.0
    th, tw = tmpl_gray.shape[:2]
    sh, sw = scene_gray.shape[:2]

    # 템플릿을 다양한 스케일로 리사이즈하며 매칭
    for scale in np.linspace(0.3, 3.0, 60):
        new_w = int(tw * scale)
        new_h = int(th * scale)
        if new_w >= sw or new_h >= sh or new_w < 20 or new_h < 10:
            continue
        resized = cv2.resize(tmpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale

    found = best_score >= threshold
    dst_corners = None
    if found and best_loc is not None:
        x, y = best_loc
        w = int(tw * best_scale)
        h = int(th * best_scale)
        dst_corners = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ]).reshape(-1, 1, 2)

    return found, best_score, dst_corners


def detect_logo_canny(scene_img, template_img, threshold=0.38):
    """
    Canny 엣지 기반 멀티스케일 Template Matching.
    엣지(윤곽)만 비교하므로 색상/톤 변화에 강하다.

    Returns:
        found (bool): 로고 검출 여부
        score (float): 최대 매칭 점수 (0~1)
        dst_corners (np.ndarray|None): 검출 영역 꼭짓점 4개
    """
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # Canny 엣지 추출
    scene_edges = cv2.Canny(scene_gray, 50, 150)
    tmpl_edges = cv2.Canny(tmpl_gray, 50, 150)

    best_score = -1
    best_loc = None
    best_scale = 1.0
    th, tw = tmpl_edges.shape[:2]
    sh, sw = scene_edges.shape[:2]

    for scale in np.linspace(0.3, 3.0, 60):
        new_w = int(tw * scale)
        new_h = int(th * scale)
        if new_w >= sw or new_h >= sh or new_w < 20 or new_h < 10:
            continue
        resized = cv2.resize(tmpl_edges, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(scene_edges, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale

    found = best_score >= threshold
    dst_corners = None
    if found and best_loc is not None:
        x, y = best_loc
        w = int(tw * best_scale)
        h = int(th * best_scale)
        dst_corners = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ]).reshape(-1, 1, 2)

    return found, best_score, dst_corners


def draw_legend(img, results, colors, line_h=30, margin=15):
    """이미지 좌측 상단에 범례를 그린다."""
    x0, y0 = margin, margin
    box_h = line_h * len(results) + margin * 2
    box_w = 380
    # 반투명 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 1)

    for i, (name, found) in enumerate(results):
        cy = y0 + margin + i * line_h
        color = colors[name]
        status = "FOUND" if found else "NOT FOUND"
        # 색상 견본 사각형
        cv2.rectangle(img, (x0 + 10, cy - 8), (x0 + 30, cy + 8), color, -1)
        cv2.rectangle(img, (x0 + 10, cy - 8), (x0 + 30, cy + 8), (0, 0, 0), 1)
        # 텍스트
        label = f"{name}: {status}"
        cv2.putText(img, label, (x0 + 40, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 40, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    # 이미지 로드
    scene = cv2.imread("box_raw.png")
    if scene is None:
        print("[ERROR] box_raw.png 를 로드할 수 없습니다.")
        sys.exit(1)

    # 폴더에서 dolby/hdmi 관련 PNG 자동 스캔 (box_raw, result_ 제외)
    scene_dir = os.path.dirname(os.path.abspath("box_raw.png")) or "."
    pattern = re.compile(r"(dolby|hdmi)", re.IGNORECASE)
    skip = re.compile(r"(box_raw|result_)", re.IGNORECASE)

    png_files = sorted([
        f for f in os.listdir(scene_dir)
        if f.lower().endswith(".png") and pattern.search(f) and not skip.search(f)
    ])

    templates = {}
    for fname in png_files:
        name = os.path.splitext(fname)[0]  # 확장자 제거해서 표시이름으로 사용
        img = cv2.imread(os.path.join(scene_dir, fname))
        if img is None:
            print(f"[WARN] {fname} 로드 실패, 건너뜁니다.")
            continue
        templates[name] = img

    print(f"Scene image : box_raw.png ({scene.shape[1]}x{scene.shape[0]})")
    for name, img in templates.items():
        print(f"  Template  : {name:<14} ({img.shape[1]}x{img.shape[0]})")
    print()

    methods = ["SIFT", "ORB", "TM", "Canny"]

    # SURF 사용 가능 여부 확인
    try:
        _ = cv2.xfeatures2d.SURF_create(400)
        methods.insert(1, "SURF")
    except (AttributeError, cv2.error):
        print("[INFO] SURF 사용 불가 (특허 제한). 건너뜁니다.\n")

    # HSV 색상환에서 균등 분배하여 고유 색상 자동 생성
    colors = {}
    n = len(templates)
    for i, name in enumerate(templates):
        hue = int(180 * i / max(n, 1))  # OpenCV HSV hue: 0~179
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    print("=" * 70)
    print(f"{'Method':<8} {'Logo':<16} {'Matches/Score':>14}  {'Result'}")
    print("=" * 70)

    for method in methods:
        result_img = scene.copy()
        label_offset_y = {}  # 겹침 방지용 라벨 y 오프셋 추적
        results_for_legend = []

        for logo_name, tmpl_img in templates.items():
            try:
                if method == "TM":
                    found, score, dst_corners = detect_logo_template(
                        scene, tmpl_img, threshold=0.75
                    )
                    status = "FOUND" if found else "NOT FOUND"
                    print(f"{method:<8} {logo_name:<16} {score:>14.4f}  {status}")
                elif method == "Canny":
                    found, score, dst_corners = detect_logo_canny(
                        scene, tmpl_img, threshold=0.38
                    )
                    status = "FOUND" if found else "NOT FOUND"
                    print(f"{method:<8} {logo_name:<16} {score:>14.4f}  {status}")
                else:
                    if method == "ORB":
                        r_thresh, m_count = 0.85, 5
                    else:
                        r_thresh, m_count = 0.75, 7
                    found, good_cnt, dst_corners = detect_logo(
                        scene, tmpl_img, method,
                        min_match_count=m_count,
                        ratio_thresh=r_thresh
                    )
                    status = "FOUND" if found else "NOT FOUND"
                    print(f"{method:<8} {logo_name:<16} {good_cnt:>14}  {status}")
                results_for_legend.append((logo_name, found))

                if found and dst_corners is not None:
                    pts = np.int32(dst_corners)
                    color = colors[logo_name]

                    # 박스 중심 계산 (겹침 영역 감지용)
                    cx = int(np.mean(pts[:, 0, 0]))
                    cy = int(np.mean(pts[:, 0, 1]))
                    grid_key = (cx // 150, cy // 150)

                    # 같은 영역에 여러 검출이 겹칠 때 박스를 약간 확장/축소
                    shift = label_offset_y.get(grid_key, 0)
                    label_offset_y[grid_key] = shift + 1

                    # 겹침 시 박스를 shift만큼 바깥으로 팽창
                    if shift > 0:
                        expand = shift * 8
                        pts_shifted = pts.copy()
                        pts_shifted[0, 0] += [-expand, -expand]
                        pts_shifted[1, 0] += [expand, -expand]
                        pts_shifted[2, 0] += [expand, expand]
                        pts_shifted[3, 0] += [-expand, expand]
                    else:
                        pts_shifted = pts

                    cv2.polylines(result_img, [pts_shifted], True, color, 3, cv2.LINE_AA)

                    # 라벨: 겹침 시 위로 단계적 오프셋
                    tx, ty = pts_shifted[0][0]
                    ty_label = ty - 12 - shift * 28
                    # 라벨 배경
                    (tw, th), _ = cv2.getTextSize(logo_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(result_img, (tx - 2, ty_label - th - 4),
                                  (tx + tw + 4, ty_label + 4), (255, 255, 255), -1)
                    cv2.putText(result_img, logo_name, (tx, ty_label),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            except Exception as e:
                print(f"{method:<8} {logo_name:<16} {'ERROR':>8}  {e}")
                results_for_legend.append((logo_name, False))

        # 범례 그리기
        draw_legend(result_img, results_for_legend, colors)

        out_path = f"result_{method}.png"
        cv2.imwrite(out_path, result_img)
        print(f"  -> saved: {out_path}")
        print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()

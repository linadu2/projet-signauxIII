import cv2
import numpy as np
import os

def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p)

def isolate_rotate_resize_debug(
    img_path,
    out_dir="debug_out",
    canvas_size=(800, 400),
    margin_px=20,
    thresh_method="auto"
):
    ensure_dir(out_dir)

    # 1) Load & grayscale
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, "01_gray.png"), gray)

    # 2) Threshold (invert for white background)
    if thresh_method == "auto":
        _, th = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(255 - gray, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(out_dir, "02_thresh.png"), th)

    # 3) Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(os.path.join(out_dir, "03_mask_clean.png"), th_clean)

    # 4) Largest contour + minAreaRect visualization
    contours, _ = cv2.findContours(th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
    c = max(contours, key=cv2.contourArea)

    vis = img.copy()
    cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)  # largest contour in green
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(vis, [box], -1, (0, 0, 255), 2)  # rotated rect in red
    cv2.imwrite(os.path.join(out_dir, "04_contour_vis.png"), vis)

    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle = angle + 90.0

    # 5) Rotate-bound color image
    (h_img, w_img) = img.shape[:2]
    (cX, cY) = (w_img // 2, h_img // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h_img * sin) + (w_img * cos))
    nH = int((h_img * cos) + (w_img * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rot = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    cv2.imwrite(os.path.join(out_dir, "05_rotated.png"), rot)

    # 6) Rotate mask in sync
    rot_mask = cv2.warpAffine(th_clean, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)
    cv2.imwrite(os.path.join(out_dir, "06_rotated_mask.png"), rot_mask)

    # 7) Tight ROI after rotation
    contours_r, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_r:
        raise ValueError("No contours after rotation")
    c_r = max(contours_r, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c_r)
    roi = rot[y:y + h_box, x:x + w_box]
    cv2.imwrite(os.path.join(out_dir, "07_roi.png"), roi)

    # 8) Resize to canvas
    canvas_w, canvas_h = canvas_size
    tgt_w = canvas_w - 2 * margin_px
    tgt_h = canvas_h - 2 * margin_px
    scale = min(tgt_w / w_box, tgt_h / h_box)
    new_w = max(1, int(w_box * scale))
    new_h = max(1, int(h_box * scale))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    cv2.imwrite(os.path.join(out_dir, "08_final_canvas.png"), canvas)

if __name__ == "__main__":
    isolate_rotate_resize_debug(
        img_path="resistance/r1/20251020_092534.jpg",
        out_dir="debug_out",
        canvas_size=(800, 400),
        margin_px=20,
        thresh_method="auto",
    )

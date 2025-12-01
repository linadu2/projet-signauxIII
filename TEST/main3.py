import cv2
import numpy as np
import os
from scipy.signal import find_peaks

# --- IMPORT DU CALCULATEUR ---
try:
    from lib.main import decode_resistor
except ImportError:
    decode_resistor = None


def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p)


# =============================================================================
# 1. ANALYSE COULEUR ADAPTATIVE (TUNED)
# =============================================================================
def get_color_name_adaptive(h, s, v):
    # 1. Special Case: Black/Silver/White (Achromatic)
    if s < 25 and v > 160: return "white"
    if v < 40: return "black"
    if s < 40: return "silver"

    # 2. Adaptive Hue Logic
    if (0 <= h < 9) or (170 <= h <= 180):
        if s > 50: return "red"
        return "brown"
    if 9 <= h < 23:
        if v > 140: return "orange"
        return "brown"
    if 23 <= h < 36:
        if v < 130 or s < 60: return "gold"
        return "yellow"
    if 36 <= h < 85: return "green"
    if 85 <= h < 135: return "blue"
    if 135 <= h < 165: return "violet"
    if 165 <= h < 170: return "grey"
    return "brown"


def analyze_band_color(roi_hsv, rect):
    x, y, w, h = rect
    center_x = x + w // 2
    margin = max(1, w // 4)
    band_crop = roi_hsv[int(h * 0.3):int(h * 0.7), max(0, center_x - margin):min(roi_hsv.shape[1], center_x + margin)]
    if band_crop.size == 0: return "unknown"
    h_val = np.median(band_crop[:, :, 0])
    s_val = np.median(band_crop[:, :, 1])
    v_val = np.median(band_crop[:, :, 2])
    return get_color_name_adaptive(h_val, s_val, v_val)


# =============================================================================
# 2. MASQUES & UTILITAIRES
# =============================================================================
def keep_resistor_body(mask_bin):
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    max_val = np.max(dist)
    if max_val == 0: return mask_bin
    _, body_core = cv2.threshold(dist, 0.4 * max_val, 255, cv2.THRESH_BINARY)
    body_core = body_core.astype(np.uint8)
    contours_core, _ = cv2.findContours(body_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_core: return mask_bin
    largest_core = max(contours_core, key=cv2.contourArea)
    core_mask_single = np.zeros_like(mask_bin)
    cv2.drawContours(core_mask_single, [largest_core], -1, 255, -1)
    dilation_size = int(max_val * 0.45)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size * 2, dilation_size * 2))
    restored_body = cv2.dilate(core_mask_single, kernel, iterations=1)
    return cv2.bitwise_and(restored_body, mask_bin)


def rects_overlap(r1, r2, threshold=0.3):
    x1, _, w1, _ = r1
    x2, _, w2, _ = r2
    start = max(x1, x2)
    end = min(x1 + w1, x2 + w2)
    if end < start: return False
    overlap_w = end - start
    min_w = min(w1, w2)
    return overlap_w > (min_w * threshold)


# =============================================================================
# 3. DÉTECTION SPÉCIALE OR
# =============================================================================
def detect_gold_only(roi_hsv, h, w):
    lower_gold = np.array([10, 50, 60])
    upper_gold = np.array([42, 255, 255])
    mask = cv2.inRange(roi_hsv, lower_gold, upper_gold)
    margin_cut = int(w * 0.06)
    mask[:, 0:margin_cut] = 0
    mask[:, w - margin_cut:w] = 0
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v)
    mask = cv2.dilate(mask, kernel_v, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gold_rects = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh > h * 0.4:
            if bw < w * 0.20 and bw > w * 0.02:
                gold_rects.append((x, 0, bw, h))
    return gold_rects


# =============================================================================
# 4. SEGMENTATION PAR BORDS (EDGE-BASED)
# =============================================================================
def find_bands(roi_img, out_dir=None, debug_prefix="debug"):
    h, w = roi_img.shape[:2]

    # 1. Convert to Grayscale and Blur slightly (Vertical)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (1, 11), 0)

    # --- SAVE STEP: Gray Blur ---
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}_step8_roi_gray_blur.jpg"), gray_blur)

    # 2. Compute Vertical Edges (Sobel X)
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.abs(sobelx)

    # --- SAVE STEP: Sobel ---
    if out_dir:
        # Normalize for saving
        sobel_norm = cv2.normalize(abs_sobelx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}_step9_sobel.jpg"), sobel_norm)

    # 3. Edge Profile
    edge_profile = np.sum(abs_sobelx, axis=0)
    if np.max(edge_profile) > 0:
        edge_profile = edge_profile / np.max(edge_profile)

    # 4. Intensity Profile (Dark Valley Logic)
    intensity_profile = np.mean(gray, axis=0)
    inv_intensity = 255 - intensity_profile
    inv_intensity = inv_intensity / np.max(inv_intensity)

    # 5. Combined Score: Edges + Dark Intensity
    combined_score = (edge_profile * 0.4) + (inv_intensity * 0.6)

    # 6. Find Peaks
    min_dist = w * 0.05
    peaks, _ = find_peaks(combined_score, distance=min_dist, height=0.25, prominence=0.08)

    final_rects = []
    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    for p in peaks:
        band_w = int(w * 0.06)
        x1 = max(0, int(p - band_w / 2))
        x2 = min(w, int(p + band_w / 2))
        rect = (x1, 0, x2 - x1, h)
        c_name = analyze_band_color(roi_hsv, rect)
        final_rects.append({'rect': rect, 'color': c_name, 'source': 'edge_peak'})

    gold_rects = detect_gold_only(roi_hsv, h, w)
    for g_rect in gold_rects:
        merged = False
        for i, item in enumerate(final_rects):
            if rects_overlap(g_rect, item['rect'], 0.3):
                final_rects[i] = {'rect': g_rect, 'color': 'gold', 'source': 'gold_override'}
                merged = True
                break
        if not merged:
            final_rects.append({'rect': g_rect, 'color': 'gold', 'source': 'gold_override'})

    final_rects.sort(key=lambda x: x['rect'][0])
    result_colors = [x['color'] for x in final_rects]
    result_rects = [x['rect'] for x in final_rects]

    if len(result_colors) >= 4 and result_colors[-1] == 'brown':
        result_colors[-1] = 'gold'
    if len(result_colors) > 0 and result_colors[0] == 'gold':
        result_colors[0] = 'brown'

    # --- Visualization ---
    vis = roi_img.copy()
    for i, r in enumerate(result_rects):
        cv2.rectangle(vis, (r[0], 0), (r[0] + r[2], h), (0, 255, 0), 2)
        cv2.putText(vis, result_colors[i], (r[0], 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    points = []
    for x in range(w):
        val = combined_score[x]
        graph_y = int(h - (val * h))
        points.append((x, graph_y))
    if points:
        cv2.polylines(vis, [np.array(points)], False, (255, 255, 255), 1)

    return result_colors, result_rects, vis


# =============================================================================
# 5. MAIN WRAPPER
# =============================================================================
def isolate_rotate_resize_debug_body(img_path, out_dir="debug_out", thresh_method="auto"):
    ensure_dir(out_dir)

    # Create a file prefix based on the input filename
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path)
    if img is None: return "errors"

    # Resize protection
    h_raw, w_raw = img.shape[:2]
    MAX_WIDTH = 1500
    if w_raw > MAX_WIDTH:
        scale = MAX_WIDTH / w_raw
        new_w = int(w_raw * scale)
        new_h = int(h_raw * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --- SAVE STEP: Original Resized ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step0_resized.jpg"), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # --- SAVE STEP: Grayscale ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step1_gray.jpg"), gray)

    if thresh_method == "auto":
        _, th = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(255 - gray, 30, 255, cv2.THRESH_BINARY)

    # --- SAVE STEP: Threshold Raw ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step2_thresh_raw.jpg"), th)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- SAVE STEP: Threshold Clean ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step3_thresh_clean.jpg"), th_clean)

    body_mask = keep_resistor_body(th_clean)

    # --- SAVE STEP: Body Mask ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step4_body_mask.jpg"), body_mask)

    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "errors"

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect

    if rw < rh:
        angle += 90.0
        rw, rh = rh, rw

    (h_img, w_img) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int((h_img * sin) + (w_img * cos))
    nH = int((h_img * cos) + (w_img * sin))
    M[0, 2] += (nW / 2) - w_img // 2
    M[1, 2] += (nH / 2) - h_img // 2

    rot = cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))
    rot_body = cv2.warpAffine(body_mask, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)

    # --- SAVE STEP: Rotated Image & Mask ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step5_rotated.jpg"), rot)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step6_rotated_mask.jpg"), rot_body)

    contours_r, _ = cv2.findContours(rot_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_r: return "errors"

    c_r = max(contours_r, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c_r)
    crop_margin = int(w_box * 0.10)
    x_roi = x + crop_margin
    y_roi = y
    w_roi = w_box - (2 * crop_margin)
    h_roi = h_box

    if w_roi < 10: w_roi = w_box; x_roi = x
    roi = rot[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]

    # --- SAVE STEP: ROI (Cropped Resistor) ---
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_step7_roi.jpg"), roi)

    # Pass base_name to find_bands for prefixed debug files
    colors, rects, debug_vis = find_bands(roi, out_dir=out_dir, debug_prefix=base_name)

    # Final Result
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_final_result.jpg"), debug_vis)

    result_data = {"colors": colors}
    if decode_resistor and len(colors) >= 3:
        best_match, history = decode_resistor(colors)
        if not best_match.get("error"):
            result_data.update({
                "resistance": best_match["ohms"],
                "unit": "Ω",
                "tolerance": f"±{best_match['tolerance_pct']}%"
            })
            return result_data

        rev_colors = list(reversed(colors))
        if decode_resistor:
            match2, _ = decode_resistor(rev_colors)
            if not match2.get("error"):
                result_data.update({
                    "resistance": match2["ohms"],
                    "unit": "Ω",
                    "tolerance": f"±{match2['tolerance_pct']}%"
                })
                return result_data

    return result_data


if __name__ == "__main__":
    print(isolate_rotate_resize_debug_body("dataset/r5/20251020_093406.jpg"))

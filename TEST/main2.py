import cv2
import numpy as np
import os

# --- IMPORT DU CALCULATEUR ---
try:
    from lib.main import decode_resistor
except ImportError:
    print("Attention: Module 'resistor_calculator' introuvable.")
    decode_resistor = None


def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p)

# =============================================================================
# 1. ANALYSE COULEUR (LOGIQUE MODIFIÉE : ECHANGE SILVER <-> GREY)
# =============================================================================

def get_color_name(h, s, v):
    # --- A. LOGIQUE ARGENT / GRIS (Prioritaire - Inversée selon ta demande précédente) ---
    if v > 70 and s < 60:
        return "grey"  # C'est brillant -> Gris (pour le calculateur)
    if s < 40: 
        if v < 50: return "black"
        return "silver" # C'est terne -> Silver (pour le calculateur)
    if v < 50: return "black"

    # --- B. TEINTES ---

    if (0 <= h < 12) or (160 <= h <= 180): 
        if s > 80: 
            return "red"
        else:
            return "grey" # Si pas saturé, c'est du gris/argent
    
    elif 12 <= h < 22:
        if v > 180 and s > 90: return "orange"
        return "brown"
        
    elif 22 <= h < 35:
        if s > 60 and v > 60: return "gold"
        if v < 150: return "brown"
        return "yellow"

    elif 35 <= h < 85: return "green"
    elif 85 <= h < 130: return "blue"
    
    elif 130 <= h < 160: 
        return "violet"
    
    return "brown"


def analyze_band_color(roi_hsv, rect):
    x, y, w, h = rect
    center_x = x + w // 2
    margin = max(1, w // 4)
    if margin < 1: margin = 1
    
    # On prend le centre de la bande
    band_crop = roi_hsv[int(h*0.25):int(h*0.75), max(0, center_x-margin):min(roi_hsv.shape[1], center_x+margin)]
    
    if band_crop.size == 0: return "unknown"
    
    h_val = np.median(band_crop[:,:,0])
    s_val = np.median(band_crop[:,:,1])
    v_val = np.median(band_crop[:,:,2])
    
    return get_color_name(h_val, s_val, v_val)


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


def remove_glare_and_fill(img_bgr, body_color_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255]) 
    mask_sat_white = cv2.inRange(hsv, lower_white, upper_white)
    
    _, _, v = cv2.split(hsv)
    mask_bright = cv2.inRange(v, 240, 255) 
    
    glare_mask = cv2.bitwise_or(mask_sat_white, mask_bright)
    kernel = np.ones((3, 3), np.uint8)
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
    
    result = img_bgr.copy()
    result[glare_mask > 0] = body_color_bgr
    return result


def get_vertical_projection(roi_img, out_dir=None):
    h, w = roi_img.shape[:2]
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)] = 255
    
    b_chan, g_chan, r_chan = cv2.split(roi_img)
    med_b = np.median(b_chan[center_mask > 0])
    med_g = np.median(g_chan[center_mask > 0])
    med_r = np.median(r_chan[center_mask > 0])
    body_bgr = (int(med_b), int(med_g), int(med_r))
    
    clean_img = remove_glare_and_fill(roi_img, body_bgr)
    
    if out_dir:
        glare_path = os.path.join(out_dir, "09a_glare_removed.png")
        cv2.imwrite(glare_path, clean_img)
    
    smooth = cv2.bilateralFilter(clean_img, 5, 75, 75)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    masked_l = l_chan[center_mask > 0]
    masked_a = a_chan[center_mask > 0]
    masked_b = b_chan[center_mask > 0]
    
    if masked_l.size == 0: return np.zeros(w)
    med_l = np.median(masked_l)
    med_a = np.median(masked_a)
    med_b = np.median(masked_b)
    
    diff_l = np.abs(l_chan.astype(np.float32) - med_l) * 1.0
    diff_a = np.abs(a_chan.astype(np.float32) - med_a) * 2.0
    diff_b = np.abs(b_chan.astype(np.float32) - med_b) * 2.0
    total_diff_map = diff_l + diff_a + diff_b
    
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    combined_energy = total_diff_map + (sobelx * 0.6)
    
    row_start, row_end = int(h * 0.2), int(h * 0.8)
    roi_energy = combined_energy[row_start:row_end, :]
    profile = np.mean(roi_energy, axis=0)
    
    if np.max(profile) > 0:
        profile = profile / np.max(profile)
        
    margin_cut = int(w * 0.05)
    profile[0:margin_cut] = 0
    profile[w-margin_cut:w] = 0
    
    return profile


# =============================================================================
# 2. DÉTECTION SPÉCIALE OR
# =============================================================================

def detect_gold_only(roi_img):
    h, w = roi_img.shape[:2]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    lower_gold = np.array([12, 60, 60]) 
    upper_gold = np.array([38, 255, 255])
    
    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    
    margin_cut = int(w * 0.06)
    mask[:, 0:margin_cut] = 0
    mask[:, w-margin_cut:w] = 0
    
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
# 3. FONCTION PRINCIPALE
# =============================================================================

def find_bands(roi_img, out_dir=None):
    h, w = roi_img.shape[:2]

    # --- A. Projection Standard ---
    profile = get_vertical_projection(roi_img, out_dir)
    THRESHOLD = 0.28
    
    band_mask_1d = (profile > THRESHOLD).astype(np.uint8)
    band_mask_1d = band_mask_1d.reshape(1, -1)
    kernel = np.ones((1, 3), np.uint8)
    band_mask_1d = cv2.dilate(band_mask_1d, kernel, iterations=1)
    band_mask_1d = cv2.erode(band_mask_1d, kernel, iterations=1)
    band_mask_1d = band_mask_1d.flatten()

    MIN_BAND_WIDTH = w * 0.03
    standard_rects = []
    in_band = False
    start_x = 0
    
    for x in range(w):
        val = band_mask_1d[x]
        if val > 0 and not in_band:
            in_band = True
            start_x = x
        elif val == 0 and in_band:
            in_band = False
            end_x = x
            width = end_x - start_x
            if width > MIN_BAND_WIDTH: 
                standard_rects.append((start_x, 0, width, h))
                
    if in_band:
        width = w - start_x
        if width > MIN_BAND_WIDTH:
            standard_rects.append((start_x, 0, width, h))

    # --- B. Détection Or Spécifique ---
    gold_rects = detect_gold_only(roi_img)
    
    # --- C. Fusion et Analyse Couleur ---
    final_rects = []
    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # 1. Ajout des standards
    for r in standard_rects:
        c_name = analyze_band_color(roi_hsv, r)
        final_rects.append({'rect': r, 'color': c_name, 'source': 'std'})

    # 2. Fusion des Golds
    for g_rect in gold_rects:
        merged = False
        for i, item in enumerate(final_rects):
            s_rect = item['rect']
            if rects_overlap(g_rect, s_rect, threshold=0.2):
                if item['color'] in ['brown', 'orange', 'unknown']:
                    final_rects[i] = {'rect': g_rect, 'color': 'gold', 'source': 'gold'}
                merged = True
                break
        if not merged:
            final_rects.append({'rect': g_rect, 'color': 'gold', 'source': 'gold'})

    final_rects.sort(key=lambda x: x['rect'][0])
    
    result_colors = [x['color'] for x in final_rects]
    result_rects = [x['rect'] for x in final_rects]

    # --- D. Heuristique de dernier recours ---
    if len(result_colors) == 4 and result_colors[-1] == 'brown':
        result_colors[-1] = 'gold'
        final_rects[-1]['source'] = 'correction'

    # --- VISUALISATION ---
    vis = roi_img.copy()
    for i, rect in enumerate(result_rects):
        x, y, wb, hb = rect
        color_name = result_colors[i]
        
        src = final_rects[i].get('source', 'std')
        if src == 'gold': box_c = (0, 255, 255)
        elif src == 'correction': box_c = (255, 255, 0)
        else: box_c = (0, 255, 0)
        
        cv2.rectangle(vis, (x, 0), (x + wb, h), box_c, 2)
        cv2.putText(vis, color_name, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    points = []
    for x in range(w):
        val = profile[x]
        graph_y = int(h - (val * h))
        points.append((x, graph_y))
    if points:
        cv2.polylines(vis, [np.array(points)], False, (255, 255, 255), 1)
        
    thresh_y = int(h - (THRESHOLD * h))
    cv2.line(vis, (0, thresh_y), (w, thresh_y), (255, 0, 0), 1)
    
    return result_colors, result_rects, vis


def isolate_rotate_resize_debug_body(
        img_path,
        out_dir="debug_out",
        thresh_method="auto"
):
    ensure_dir(out_dir)
    img = cv2.imread(img_path)
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if thresh_method == "auto":
        _, th = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(255 - gray, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    body_mask = keep_resistor_body(th_clean)

    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
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

    contours_r, _ = cv2.findContours(rot_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_r: return
    c_r = max(contours_r, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c_r)

    crop_margin = int(w_box * 0.10)
    x_roi = x + crop_margin
    y_roi = y
    w_roi = w_box - (2 * crop_margin)
    h_roi = h_box
    if w_roi < 10: w_roi = w_box; x_roi = x
    roi = rot[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]
    cv2.imwrite(os.path.join(out_dir, "07_roi_body.png"), roi)

    print("Running Analysis...")
    colors, rects, debug_vis = find_bands(roi, out_dir=out_dir)
    
    #print(f"Found {len(colors)} bands.")
    #print(f"Colors: {colors}")
    cv2.imwrite(os.path.join(out_dir, "09_bands_detected.png"), debug_vis)

    if decode_resistor and len(colors) >= 3:
        #print("\n--- CALCULATEUR ---")
        best_match, history = decode_resistor(colors)
        if not best_match.get("error"):
            #print(f"Résistance: {best_match['ohms']} Ohms")
            #print(f"Tolérance: {best_match['tolerance_pct']}%")
            return {
            "resistance": best_match["ohms"],
            "unit": "Ω",
            "tolerance": f"±{best_match['tolerance_pct']}%",
            "colors": colors
        }
        else:
            #print("Erreur décodage:", best_match.get("reason"))
            rev_colors = list(reversed(colors))
            match2, _ = decode_resistor(rev_colors)
            if not match2.get("error"):
                #print(f" -> (Sens Inverse) {match2['ohms']} Ohms")
                return {
                    "resistance": match2["ohms"],
                    "unit": "Ω",
                    "tolerance": f"±{match2['tolerance_pct']}%",
                    "colors": colors
                }
    else:
        #print("Pas assez de bandes ou calculateur absent.")
        return "errors"


if __name__ == "__main__":
    print(isolate_rotate_resize_debug_body(
        img_path="dataset/r1/20251020_092534.jpg",
        out_dir="debug_out",
    ))
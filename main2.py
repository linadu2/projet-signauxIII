import cv2
import numpy as np
import os

# --- IMPORT DU CALCULATEUR ---
try:
    from resistor_calculator.main import decode_resistor
except ImportError:
    print("Attention: Module 'resistor_calculator' introuvable.")
    decode_resistor = None


def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p)


# --- NOUVELLE FONCTION : Identifier la couleur ---
def get_color_name(h, s, v):
    """Détermine le nom de la couleur selon HSV"""
    # 1. Achromatique (Gris, Noir, Blanc)
    if s < 25: 
        if v < 80: return "black"
        if v > 200: return "white" # ou silver très brillant
        return "silver"
    
    if v < 50: return "black"

    # 2. Teintes (H = 0..179)
    if (0 <= h < 10) or (170 <= h <= 180):
        return "red"
    elif 10 <= h < 22:
        # Zone ambigue Marron / Orange / Or
        if v > 180 and s > 100: return "orange" # Orange vif
        if s > 80 and v < 180: return "brown"   # Marron classique
        return "brown"
    elif 22 <= h < 33:
        # Zone Jaune / Or
        if s > 100 and v < 220: return "gold"   # Or (souvent plus sombre que le jaune pur)
        return "yellow"
    elif 33 <= h < 85:
        return "green"
    elif 85 <= h < 130:
        return "blue"
    elif 130 <= h < 170:
        return "violet"
    
    return "brown" # Fallback


def analyze_band_color(roi_hsv, rect):
    """Extrait la couleur moyenne dans le rectangle de la bande"""
    x, y, w, h = rect
    
    # On regarde au centre de la bande pour éviter les bords flous
    center_x = x + w // 2
    margin = max(1, w // 4)
    
    # Petit crop au centre de la bande
    band_crop = roi_hsv[int(h*0.25):int(h*0.75), center_x-margin:center_x+margin]
    
    if band_crop.size == 0: return "unknown"
    
    # Médiane pour éviter le bruit
    h_val = np.median(band_crop[:,:,0])
    s_val = np.median(band_crop[:,:,1])
    v_val = np.median(band_crop[:,:,2])
    
    return get_color_name(h_val, s_val, v_val)


# --- TON CODE ORIGINAL (Intact) ---

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
    upper_white = np.array([180, 60, 255])
    mask_sat_white = cv2.inRange(hsv, lower_white, upper_white)
    _, _, v = cv2.split(hsv)
    mask_bright = cv2.inRange(v, 235, 255)
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
    if out_dir: cv2.imwrite(os.path.join(out_dir, "09a_glare_removed.png"), clean_img)
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
    x_axis = np.linspace(-1, 1, w)
    gaussian_weight = np.exp(-(x_axis ** 2) / (2 * (0.65 ** 2)))
    profile = profile * gaussian_weight
    if np.max(profile) > 0:
        profile = profile / np.max(profile)
    return profile


def find_bands(roi_img, out_dir=None):
    h, w = roi_img.shape[:2]

    profile = get_vertical_projection(roi_img, out_dir)

    THRESHOLD = 0.20

    band_mask_1d = (profile > THRESHOLD).astype(np.uint8)
    band_mask_1d = band_mask_1d.reshape(1, -1)
    kernel = np.ones((1, 3), np.uint8)
    band_mask_1d = cv2.dilate(band_mask_1d, kernel, iterations=1)
    band_mask_1d = cv2.erode(band_mask_1d, kernel, iterations=1)
    band_mask_1d = band_mask_1d.flatten()

    band_rects = []
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
            if (end_x - start_x) > 2:
                band_rects.append((start_x, 0, end_x - start_x, h))

    if in_band and (w - start_x) > 2:
        band_rects.append((start_x, 0, w - start_x, h))

    # --- MODIFICATION ICI : Analyse des couleurs ---
    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    detected_colors = []

    vis = roi_img.copy()
    for i, rect in enumerate(band_rects):
        x, y, wb, hb = rect
        
        # Trouver la couleur
        color_name = analyze_band_color(roi_hsv, rect)
        detected_colors.append(color_name)

        cv2.rectangle(vis, (x, 0), (x + wb, h), (0, 255, 0), 2)
        # Afficher la couleur sur l'image
        cv2.putText(vis, color_name, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    points = []
    for x in range(w):
        val = profile[x]
        graph_y = int(h - (val * h))
        points.append((x, graph_y))

    cv2.polylines(vis, [np.array(points)], False, (255, 255, 255), 1)
    thresh_y = int(h - (THRESHOLD * h))
    cv2.line(vis, (0, thresh_y), (w, thresh_y), (255, 0, 0), 1)

    # On retourne maintenant les couleurs en plus
    return detected_colors, band_rects, vis


def isolate_rotate_resize_debug_body(
        img_path,
        out_dir="debug_out",
        canvas_size=(800, 400),
        margin_px=20,
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

    # --- CLEARANCE / CROP MODIFICATION ---
    crop_margin = int(w_box * 0.10)

    x_roi = x + crop_margin
    y_roi = y
    w_roi = w_box - (2 * crop_margin)
    h_roi = h_box

    if w_roi < 10: w_roi = w_box; x_roi = x
    roi = rot[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]
    cv2.imwrite(os.path.join(out_dir, "07_roi_body.png"), roi)

    print("Running Vertical Projection...")
    # Récupération des couleurs ici
    colors, rects, debug_vis = find_bands(roi, out_dir=out_dir)
    
    print(f"Found {len(colors)} bands.")
    print(f"Colors: {colors}")
    cv2.imwrite(os.path.join(out_dir, "09_bands_detected.png"), debug_vis)

    # --- ENVOI AU CALCULATEUR ---
    if decode_resistor and len(colors) >= 3:
        print("\n--- CALCULATEUR ---")
        best_match, history = decode_resistor(colors)
        if not best_match.get("error"):
            print(f"Résistance: {best_match['ohms']} Ohms")
            print(f"Tolérance: {best_match['tolerance_pct']}%")
        else:
            print("Erreur décodage:", best_match.get("reason"))
    else:
        print("Pas assez de bandes ou calculateur absent.")


if __name__ == "__main__":
    isolate_rotate_resize_debug_body(
        img_path="resistance/r1/20251020_092534.jpg",
        out_dir="debug_out",
    )
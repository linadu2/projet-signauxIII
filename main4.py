import cv2
import numpy as np
import os
import math

# --- IMPORT CALCULATEUR ---
try:
    from resistor_calculator.main import decode_resistor
except ImportError:
    decode_resistor = None

# --- CONFIG ---
CROP_MARGIN_PERCENT = 0.02
DEBUG_MODE = True

# --- CALIBRAGE LAB (Spécial Corps Rouge - Sans Silver) ---
REF_COLORS_LAB = {
    "black":  (20, 128, 128),
    "white":  (240, 128, 128),
    
    # Gris : On cible un gris moyen/clair. 
    # Tout ce qui brillait comme "Silver" tombera ici ou dans White.
    "grey":   (130, 128, 128),
    
    "brown":  (50, 138, 135),  
    "red":    (100, 170, 150),   
    "orange": (140, 160, 170),
    "yellow": (210, 130, 190),  
    "gold":   (140, 135, 160),  
    "green":  (80, 100, 135),  
    "blue":   (50, 130, 80), 
    "violet": (70, 155, 90), 
}

def ensure_dir(p):
    if p and not os.path.exists(p): os.makedirs(p)

def remove_glare(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_glare = cv2.inRange(hsv, (0, 0, 230), (180, 20, 255))
    mask_glare = cv2.dilate(mask_glare, np.ones((3,3), np.uint8), iterations=1)
    return cv2.inpaint(img, mask_glare, 2, cv2.INPAINT_TELEA)

def get_closest_color(mean_lab):
    l, a, b = mean_lab
    # Chroma = Saturation (distance du centre gris 128,128)
    chroma = math.sqrt((a - 128)**2 + (b - 128)**2)

    # --- 1. GESTION DES NEUTRES (NOIR/GRIS/BLANC) ---
    # Si la couleur est très faible (chroma < 10), c'est un neutre.
    if chroma < 10:
        if l < 40: return "black"
        if l > 210: return "white"
        # Tout ce qui est entre les deux est gris (y compris ce qui était silver)
        return "grey"

    # --- 2. DISTANCE PONDÉRÉE ---
    min_dist = float('inf')
    best_name = "unknown"

    for name, (rl, ra, rb) in REF_COLORS_LAB.items():
        if name in ['black', 'white', 'grey']: continue
        
        # Poids : On privilégie la teinte (a,b)
        dist = math.sqrt(0.8*(l - rl)**2 + 2.0*(a - ra)**2 + 2.0*(b - rb)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_name = name

    # --- 3. CORRECTIFS ---

    # FIX MARRON (Teinte Rougeâtre)
    if best_name == "brown":
        # Si c'est "Marron" mais plus jaune que rouge -> OR
        if b > a: return "gold"
        # Si c'est "Marron" mais très lumineux -> OR
        if l > 75: return "gold"
        # Si c'est "Marron" mais très saturé en rouge -> ROUGE
        if a > 155: return "red"

    # FIX OR
    if best_name == "yellow" and l < 160: return "gold"

    # FIX VIOLET/BLEU
    if best_name == "violet" and a < 140: return "blue"
    if best_name == "blue" and a > 145: return "violet"

    return best_name

def scan_resistor_bands(roi_img, out_dir=None):
    h, w = roi_img.shape[:2]
    
    clean = remove_glare(roi_img)
    if out_dir: 
        cv2.imwrite(os.path.join(out_dir, "05b_glare_removed.png"), clean)

    smooth = cv2.bilateralFilter(clean, 5, 50, 50)
    if out_dir: 
        cv2.imwrite(os.path.join(out_dir, "10_clean_smooth.png"), smooth)

    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    bg_l, bg_a, bg_b = np.median(l), np.median(a), np.median(b)
    
    y_start, y_end = int(h * 0.2), int(h * 0.8)
    diff_signal = []
    
    for x in range(w):
        col_l = np.mean(l[y_start:y_end, x])
        col_a = np.mean(a[y_start:y_end, x])
        col_b = np.mean(b[y_start:y_end, x])
        d = math.sqrt(0.6*(col_l-bg_l)**2 + 1.8*(col_a-bg_a)**2 + 1.8*(col_b-bg_b)**2)
        diff_signal.append(d)
        
    diff_signal = np.array(diff_signal)
    if np.max(diff_signal) > 0: diff_signal /= np.max(diff_signal)

    # Seuil
    mask_bands = (diff_signal > 0.22).astype(np.uint8)
    kernel = np.ones((1, max(3, int(w * 0.02))), np.uint8)
    mask_bands = cv2.morphologyEx(mask_bands.reshape(1, -1), cv2.MORPH_CLOSE, kernel).flatten()

    bands_found = []
    in_band = False
    start_x = 0
    margin = int(w * 0.10)
    
    for x in range(margin, w - margin):
        val = mask_bands[x]
        if val == 1 and not in_band:
            in_band, start_x = True, x
        elif val == 0 and in_band:
            in_band = False
            width = x - start_x
            if width > w * 0.02:
                cx = start_x + width // 2
                sw = max(1, width // 3)
                roi_b = lab[y_start:y_end, cx-sw:cx+sw]
                ml, ma, mb = np.median(roi_b[:,:,0]), np.median(roi_b[:,:,1]), np.median(roi_b[:,:,2])
                c_name = get_closest_color((ml, ma, mb))
                bands_found.append({'color': c_name, 'rect': (start_x, 0, width, h)})

    vis = clean.copy()
    pts = [(x, int(h - (diff_signal[x] * h))) for x in range(w)]
    cv2.polylines(vis, [np.array(pts)], False, (0,0,255), 1)
    
    final_colors, final_rects = [], []
    for b in bands_found:
        x, y, bw, bh = b['rect']
        cv2.rectangle(vis, (x, 0), (x+bw, h), (0, 255, 0), 2)
        cv2.putText(vis, b['color'], (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        final_colors.append(b['color'])
        final_rects.append(b['rect'])

    return final_colors, final_rects, vis

def keep_resistor_body(mask_bin):
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    max_val = np.max(dist)
    if max_val == 0: return mask_bin
    _, core = cv2.threshold(dist, 0.3 * max_val, 255, cv2.THRESH_BINARY)
    core = core.astype(np.uint8)
    ctrs, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs: return mask_bin
    mask_s = np.zeros_like(mask_bin)
    cv2.drawContours(mask_s, [max(ctrs, key=cv2.contourArea)], -1, 255, -1)
    return cv2.bitwise_and(cv2.dilate(mask_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(max_val), int(max_val)))), mask_bin)

def process_resistor_image(img_path, out_dir="output"):
    ensure_dir(out_dir)
    img = cv2.imread(img_path)
    if img is None: return print(f"Err: {img_path}")

    h, w = img.shape[:2]
    if w > 1000: img = cv2.resize(img, (1000, int(h * (1000/w))))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body_mask = keep_resistor_body(cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)))
    cv2.imwrite(os.path.join(out_dir, "02_body_mask.png"), body_mask)

    ctrs, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs: return print("No body")
    
    rect = cv2.minAreaRect(max(ctrs, key=cv2.contourArea))
    if rect[1][0] < rect[1][1]: angle = rect[2] + 90
    else: angle = rect[2]
    
    M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
    rot = cv2.warpAffine(img, M, img.shape[:2][::-1], borderValue=(255,255,255))
    rot_mask = cv2.warpAffine(body_mask, M, img.shape[:2][::-1], flags=cv2.INTER_NEAREST, borderValue=0)
    
    c_r = max(cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    x, y, wb, hb = cv2.boundingRect(c_r)
    roi = rot[y+int(hb*0.1):y+hb-int(hb*0.1), x+int(wb*CROP_MARGIN_PERCENT):x+wb-int(wb*CROP_MARGIN_PERCENT)]
    
    if roi.size == 0: return print("Crop failed")
    cv2.imwrite(os.path.join(out_dir, "05_roi.png"), roi)

    # print("Scanning...")
    colors, rects, vis = scan_resistor_bands(roi, out_dir)
    cv2.imwrite(os.path.join(out_dir, "06_result.png"), vis)
    # print(f"Colors: {colors}")

    if decode_resistor and len(colors) >= 3:
        # print("\n--- CALCULATEUR ---")
        best_match, history = decode_resistor(colors)
        if not best_match.get("error"):
            # print(f"Résistance: {best_match['ohms']} Ohms")
            # print(f"Tolérance: {best_match['tolerance_pct']}%")
            return {
                "resistance": best_match["ohms"],
                "unit": "Ω",
                "tolerance": f"±{best_match['tolerance_pct']}%",
                "colors": colors
            }
        else:
            # print("Erreur décodage:", best_match.get("reason"))
            rev_colors = list(reversed(colors))
            match2, _ = decode_resistor(rev_colors)
            if not match2.get("error"):
                # print(f" -> (Sens Inverse) {match2['ohms']} Ohms")
                return {
                    "resistance": match2["ohms"],
                    "unit": "Ω",
                    "tolerance": f"±{match2['tolerance_pct']}%",
                    "colors": colors
                }
    else:
        # print("Pas assez de bandes ou calculateur absent.")
        return "errors"

if __name__ == "__main__":
    print(process_resistor_image("resistance/r5/20251020_093432.jpg", "debug_out"))
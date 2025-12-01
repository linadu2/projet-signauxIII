import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def rotate_resistor(img_path):
    """
    Rotates and crops a resistor image to be horizontal.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    rect = cv2.minAreaRect(coords)
    (cx, cy), (rw, rh), angle = rect

    if rw < rh:
        angle += 90.0

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    rotated_thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    x, y, w_box, h_box = cv2.boundingRect(rotated_thresh)
    roi = rotated[y:y + h_box, x:x + w_box]

    target_h = 60
    scale = target_h / roi.shape[0]
    resized = cv2.resize(roi, (int(roi.shape[1] * scale), target_h))

    return resized


def scan_middle_line(img, line_height=5, step_px=5):
    """
    Scans a single horizontal line through the middle of the resistor.

    Args:
        img: Rotated resistor image (BGR format)
        line_height: Height of the horizontal strip to scan (default 5px)
        step_px: Step size in pixels for horizontal scanning (default 5)

    Returns:
        Tuple of (raw_scan_data, saturated_scan_data)
    """
    h, w = img.shape[:2]

    # Calculate middle line position
    middle_y = h // 2
    line_top = middle_y - line_height // 2
    line_bottom = line_top + line_height

    print(f"\n=== SCANNING MIDDLE LINE ===")
    print(f"Image size: {w}x{h}px")
    print(f"Scanning horizontal strip from Y={line_top} to Y={line_bottom} (center: {middle_y})")
    print(f"Strip height: {line_height}px")
    print(f"Step size: {step_px}px")

    # Extract the middle horizontal strip
    middle_strip = img[line_top:line_bottom, :]

    raw_scan = []
    saturated_scan = []

    # Scan left to right through the strip
    for x in range(0, w, step_px):
        x_end = min(x + step_px, w)

        # Extract small region from the strip
        region = middle_strip[:, x:x_end]
        pixels = region.reshape(-1, 3)

        # Filter out white/bright background
        mask = np.all(pixels < [220, 220, 220], axis=1)
        valid_pixels = pixels[mask]

        if len(valid_pixels) > 0:
            # Calculate average color (BGR)
            avg_color_bgr = np.mean(valid_pixels, axis=0).astype(np.uint8)
            avg_color_rgb = np.array([avg_color_bgr[2], avg_color_bgr[1], avg_color_bgr[0]])
            raw_scan.append(tuple(avg_color_rgb))

            # Boost saturation in HSV space
            hsv = cv2.cvtColor(np.uint8([[avg_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            h_val, s_val, v_val = hsv[0], hsv[1], hsv[2]

            s_boosted = min(int(s_val * 3.0), 255)
            v_boosted = min(int(v_val * 1.2), 255)

            hsv_boosted = np.uint8([[[h_val, s_boosted, v_boosted]]])
            bgr_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)[0][0]
            rgb_boosted = np.array([bgr_boosted[2], bgr_boosted[1], bgr_boosted[0]])

            saturated_scan.append(tuple(rgb_boosted))
        else:
            # No valid pixels (background)
            raw_scan.append((255, 255, 255))
            saturated_scan.append((255, 255, 255))

    print(f"Scanned {len(raw_scan)} positions")

    return raw_scan, saturated_scan


def detect_legs_and_body(saturated_data):
    """
    Detects resistor legs and finds the body region.

    Returns:
        tuple: (body_start_idx, body_end_idx)
    """
    brightness = [np.mean(rgb) for rgb in saturated_data]
    brightness_arr = np.array(brightness)

    # Find valleys (legs are dark)
    valleys, _ = find_peaks(-brightness_arr, prominence=30, width=3)

    print(f"\n=== LEG DETECTION ===")
    print(f"Detected potential leg positions at steps: {list(valleys)}")

    if len(valleys) >= 2:
        first_leg_end = valleys[0] + 5
        last_leg_start = valleys[-1] - 5

        body_start = first_leg_end
        body_end = last_leg_start

        print(f"Body region: steps {body_start} to {body_end}")
        return body_start, body_end
    else:
        body_start = int(len(saturated_data) * 0.2)
        body_end = int(len(saturated_data) * 0.8)
        print(f"Fallback body region: steps {body_start} to {body_end}")
        return body_start, body_end


def detect_color_bands(saturated_data, body_start, body_end):
    """
    Detects color bands within the body region by finding peaks in color variance.

    Returns:
        tuple: (band_positions, diff_signal)
    """
    body_data = saturated_data[body_start:body_end]
    body_colors = np.array(body_data)
    median_color = np.median(body_colors, axis=0)

    print(f"\n=== BODY ANALYSIS ===")
    print(f"Median body color (RGB): {median_color.astype(int)}")

    diff_signal = np.linalg.norm(body_colors - median_color, axis=1)

    peaks, properties = find_peaks(diff_signal,
                                   height=20,
                                   distance=5,
                                   prominence=10)

    band_positions = [p + body_start for p in peaks]

    print(f"\n=== BAND DETECTION ===")
    print(f"Detected {len(band_positions)} color bands at steps: {band_positions}")

    return band_positions, diff_signal


def classify_color(rgb_tuple):
    """
    Classifies an RGB color into resistor color code names.
    """
    r, g, b = rgb_tuple

    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 50:
        return "Black"

    if s < 50:
        if v > 200:
            return "Silver"
        elif v > 100:
            return "Grey"
        else:
            return "Black"

    if s < 100 and v < 120 and (h < 20 or h > 160):
        return "Brown"

    if h < 10 or h > 170:
        if s > 100 and v > 100:
            return "Red"
        else:
            return "Brown"
    elif 10 <= h < 25:
        return "Orange" if v > 120 else "Brown"
    elif 25 <= h < 40:
        return "Yellow" if v > 140 else "Gold"
    elif 40 <= h < 80:
        return "Green"
    elif 80 <= h < 130:
        return "Blue"
    elif 130 <= h < 160:
        return "Violet"
    else:
        return "Unknown"


def analyze_bands(saturated_data, band_positions):
    """
    Analyzes and classifies detected color bands.
    """
    colors = []

    print(f"\n=== BAND CLASSIFICATION ===")
    for i, pos in enumerate(band_positions):
        rgb = saturated_data[pos]
        color_name = classify_color(rgb)
        colors.append(color_name)

        bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]

        print(f"Band {i + 1} at step {pos}:")
        print(f"  RGB: {rgb}")
        print(f"  HSV: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
        print(f"  Color: {color_name}")

    return colors


def plot_full_analysis(raw_data, saturated_data, body_start, body_end, band_positions, colors):
    """
    Plots comprehensive analysis with all detected features.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    r_sat = [rgb[0] for rgb in saturated_data]
    g_sat = [rgb[1] for rgb in saturated_data]
    b_sat = [rgb[2] for rgb in saturated_data]

    axes[0].plot(r_sat, 'r', label='Red', linewidth=2, alpha=0.7)
    axes[0].plot(g_sat, 'g', label='Green', linewidth=2, alpha=0.7)
    axes[0].plot(b_sat, 'b', label='Blue', linewidth=2, alpha=0.7)

    axes[0].axvspan(body_start, body_end, alpha=0.1, color='yellow', label='Body Region')

    for i, pos in enumerate(band_positions):
        axes[0].axvline(x=pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[0].text(pos, 240, colors[i], rotation=90, va='bottom', ha='right',
                     fontweight='bold', fontsize=9)

    axes[0].set_title("Color Analysis - Middle Line Scan", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Color Value (0-255)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0, 255)

    color_bar_sat = np.zeros((1, len(saturated_data), 3), dtype=np.uint8)
    for i, rgb in enumerate(saturated_data):
        color_bar_sat[0, i] = rgb
    axes[1].imshow(color_bar_sat, aspect='auto')
    axes[1].set_yticks([])
    axes[1].set_ylabel("Saturated")

    for pos in band_positions:
        axes[1].axvline(x=pos, color='red', linestyle='--', linewidth=2, alpha=0.7)

    color_bar_raw = np.zeros((1, len(raw_data), 3), dtype=np.uint8)
    for i, rgb in enumerate(raw_data):
        color_bar_raw[0, i] = rgb
    axes[2].imshow(color_bar_raw, aspect='auto')
    axes[2].set_yticks([])
    axes[2].set_ylabel("Raw")
    axes[2].set_xlabel("Step Number")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = "dataset/r5/20251020_093409.jpg"

    print("=" * 60)
    print("RESISTOR COLOR CODE ANALYZER")
    print("=" * 60)

    # Step 1: Rotate and crop
    result = rotate_resistor(img_path)

    if result is not None:
        # Step 2: Scan single horizontal line through middle
        raw_data, saturated_data = scan_middle_line(result, line_height=5, step_px=5)

        # Step 3: Detect legs and body
        body_start, body_end = detect_legs_and_body(saturated_data)

        # Step 4: Detect color bands
        band_positions, diff_signal = detect_color_bands(saturated_data, body_start, body_end)

        # Step 5: Classify colors
        colors = analyze_bands(saturated_data, band_positions)

        # Step 6: Output result
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULT: {colors}")
        print(f"{'=' * 60}")

        # Step 7: Visualize everything
        plot_full_analysis(raw_data, saturated_data, body_start, body_end, band_positions, colors)
    else:
        print("Error: Could not load image")

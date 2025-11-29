import os
import json
import time
import pandas as pd
from collections import Counter

# Import the analysis function from your optimized script
# Ensure your optimized script is named 'main3.py' and is in the same directory
try:
    from main4 import isolate_rotate_resize_debug_body
except ImportError:
    print("Error: Could not import 'main3.py'. Make sure the file exists and is named correctly.")
    exit(1)

# Configuration
RESISTANCE_DIR = "resistance"
JSON_FILE = os.path.join(RESISTANCE_DIR, "value.json")
OUTPUT_CSV = "efficiency_report.csv"


def load_ground_truth(json_path):
    """Loads the expected color bands from value.json."""
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return {}

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            # Convert lists to simpler expected strings for comparison if needed,
            # but keeping them as lists is fine for exact matching.
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}


def compare_bands(detected, expected):
    """
    Compares detected bands with expected bands.
    Returns (is_exact_match, accuracy_score, mismatch_details).
    """
    if not detected or detected == "errors":
        return False, 0.0, "Detection Failed"

    # We only care about the color list 'colors' from the result dictionary
    detected_colors = detected.get("colors", [])

    # Normalize both lists to lowercase for safer comparison
    d_norm = [c.lower() for c in detected_colors]
    e_norm = [c.lower() for c in expected]

    # Check for exact match (order matters)
    if d_norm == e_norm:
        return True, 1.0, "Match"

    # Check if it's a reversed match (common issue with resistors)
    if d_norm == list(reversed(e_norm)):
        return True, 1.0, "Match (Reversed)"

    # Calculate partial accuracy (SequenceMatcher or simple intersection could be used)
    # Here we use a simple positional match count for strictness,
    # or intersection for leniency. Let's do intersection-based to see if we found the right colors at least.

    common = Counter(d_norm) & Counter(e_norm)
    matches = sum(common.values())
    total_expected = len(e_norm)

    accuracy = matches / total_expected if total_expected > 0 else 0

    # Formatting mismatch string
    return False, accuracy, f"Expected {e_norm} but got {d_norm}"


def main():
    ground_truth = load_ground_truth(JSON_FILE)
    if not ground_truth:
        return

    results = []

    print(f"{'Folder':<10} {'Image':<25} {'Time(s)':<10} {'Status':<15} {'Accuracy':<10}")
    print("-" * 75)

    total_start_time = time.time()
    total_images = 0
    total_matches = 0

    # Iterate through folders r1, r2, ... defined in the JSON keys
    # (or just iterate directories if you prefer)
    sorted_folders = sorted([k for k in ground_truth.keys()])

    for folder_name in sorted_folders:
        folder_path = os.path.join(RESISTANCE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        expected_colors = ground_truth[folder_name]

        # List all jpg images
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            total_images += 1

            # --- Run Detection ---
            t0 = time.time()
            try:
                # Suppress print output from main3 during bulk processing if desired
                detection_result = isolate_rotate_resize_debug_body(
                    img_path=img_path,
                    out_dir=os.path.join("debug_test", folder_name),  # Separate debug folder per resistor
                    thresh_method="auto"
                )
            except Exception as e:
                detection_result = "errors"
                print(f"Exception processing {img_name}: {e}")

            dt = time.time() - t0

            # --- Compare Results ---
            is_match, accuracy, notes = compare_bands(detection_result, expected_colors)

            if is_match:
                total_matches += 1
                status = "OK"
            else:
                status = "FAIL"

            # Print minimal stats to console
            print(f"{folder_name:<10} {img_name:<25} {dt:.4f}     {status:<15} {accuracy:.2%}")

            # Log to results list
            results.append({
                "Folder": folder_name,
                "Image": img_name,
                "Execution Time": dt,
                "Success": is_match,
                "Accuracy": accuracy,
                "Detected": detection_result.get("colors", []) if isinstance(detection_result, dict) else "Error",
                "Expected": expected_colors,
                "Notes": notes
            })

    total_time = time.time() - total_start_time

    # --- Summary ---
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    print(f"Total Images Processed: {total_images}")
    print(f"Total Successful Detections: {total_matches}")
    print(f"Overall Success Rate: {(total_matches / total_images) * 100:.2f}%")
    print(f"Total Time Taken: {total_time:.2f}s")
    print(f"Average Time per Image: {total_time / total_images:.4f}s")

    # Save CSV Report
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed report saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

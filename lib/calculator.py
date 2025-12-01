from math import log10, floor

# IEC 60062 color code: digits, multipliers, tolerances
DIGIT = {
    "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
    "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9
}

MULTIPLIER = {
    "silver": 1e-2, "gold": 1e-1,
    "black": 1, "brown": 1e1, "red": 1e2, "orange": 1e3, "yellow": 1e4,
    "green": 1e5, "blue": 1e6, "violet": 1e7, "grey": 1e8, "white": 1e9
}

TOLERANCE = {
    # none means 20%
    "silver": 10.0, "gold": 5.0, "brown": 1.0, "red": 2.0, "green": 0.5,
    "blue": 0.25, "violet": 0.1, "grey": 0.05, "none": 20.0
}

# Preferred numbers (IEC 60063)
E6  = [1.0, 1.5, 2.2, 3.3, 4.7, 6.8]
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
# E96 list (mantissas in 1–10 decade) from IEC 60063
E96 = [1.00,1.02,1.05,1.07,1.10,1.13,1.15,1.18,1.21,1.24,1.27,1.30,
       1.33,1.37,1.40,1.43,1.47,1.50,1.54,1.58,1.62,1.65,1.69,1.74,
       1.78,1.82,1.87,1.91,1.96,2.00,2.05,2.10,2.15,2.21,2.26,2.32,
       2.37,2.43,2.49,2.55,2.61,2.67,2.74,2.80,2.87,2.94,3.01,3.09,
       3.16,3.24,3.32,3.40,3.48,3.57,3.65,3.74,3.83,3.92,4.02,4.12,
       4.22,4.32,4.42,4.53,4.64,4.75,4.87,4.99,5.11,5.23,5.36,5.49,
       5.62,5.76,5.90,6.04,6.19,6.34,6.49,6.65,6.81,6.98,7.15,7.32,
       7.50,7.68,7.87,8.06,8.25,8.45,8.66,8.87,9.09,9.31,9.53,9.76]

def pick_series_by_tolerance(tol_pct):
    # Typical mapping: E6≈20%, E12≈10%, E24≈5%, E48≈2%, E96≈1%
    if tol_pct is None or tol_pct >= 20.0:
        return "E6", E6
    if tol_pct >= 10.0:
        return "E12", E12
    if tol_pct >= 5.0:
        return "E24", E24
    if tol_pct >= 2.0:
        # Use E48/E96 group; approximate with E96 for PoC
        return "E96", E96
    # <= 1% (1%, 0.5%, 0.25%, 0.1%) -> use E96 as practical default
    return "E96", E96

def normalize_mantissa(value_ohms):
    if value_ohms <= 0:
        return None, None
    decade = floor(log10(value_ohms))
    mant = value_ohms / (10 ** decade)
    # Shift to [1,10)
    if mant < 1:
        mant *= 10
        decade -= 1
    elif mant >= 10:
        mant /= 10
        decade += 1
    return mant, decade

def is_standard_value(value_ohms, series_values):
    mant, _ = normalize_mantissa(value_ohms)
    if mant is None:
        return False
    # Compare within tight epsilon to avoid FP artifacts
    for v in series_values:
        if abs(mant - v) <= 1e-6:
            return True
    return False

def parse_resistor_bands(bands):
    b = [c.strip().lower() for c in bands]
    n = len(b)
    # Determine digits, multiplier, tolerance for 3–6 bands
    if n == 3:
        d1, d2, mul = b[0:3]
        sig = [DIGIT.get(d1), DIGIT.get(d2)]
        tol = 20.0
        m = MULTIPLIER.get(mul)
    elif n == 4:
        d1, d2, mul, tol_c = b[0:4]
        sig = [DIGIT.get(d1), DIGIT.get(d2)]
        m = MULTIPLIER.get(mul)
        tol = TOLERANCE.get(tol_c, None)
    elif n == 5:
        d1, d2, d3, mul, tol_c = b[0:5]
        sig = [DIGIT.get(d1), DIGIT.get(d2), DIGIT.get(d3)]
        m = MULTIPLIER.get(mul)
        tol = TOLERANCE.get(tol_c, None)
    elif n == 6:
        # Ignore 6th band (TCR) for value computation
        d1, d2, d3, mul, tol_c, _tcr = b[0:6]
        sig = [DIGIT.get(d1), DIGIT.get(d2), DIGIT.get(d3)]
        m = MULTIPLIER.get(mul)
        tol = TOLERANCE.get(tol_c, None)
    else:
        raise ValueError("Support only 3 to 6 bands")

    if any(v is None for v in sig) or m is None:
        raise ValueError("Invalid color sequence for digits/multiplier")

    # Build numeric from significant digits
    val = 0
    for d in sig:
        val = val * 10 + d
    ohms = val * m
    return ohms, tol

def decode_resistor(bands):
    """
    Try forward orientation; verify 'standard' against E-series
    implied by tolerance; if not standard, reverse and retry.
    """
    tried = []
    for orientation, seq in [("forward", bands), ("reversed", list(reversed(bands)))]:
        try:
            ohms, tol = parse_resistor_bands(seq)
            series_name, series_vals = pick_series_by_tolerance(tol)
            standard = is_standard_value(ohms, series_vals)
            tried.append({
                "orientation": orientation,
                "ohms": ohms,
                "tolerance_pct": tol,
                "series": series_name,
                "is_standard": standard,
                "bands": seq
            })
            # Return immediately if standard found
            if standard:
                return tried[-1], tried
        except Exception:
            tried.append({
                "orientation": orientation,
                "error": True,
                "bands": seq
            })
            continue
    # If neither orientation produced a standard value, return best forward result
    # or an error if both failed
    for r in tried:
        if not r.get("error"):
            return r, tried
    return {"error": True, "reason": "Could not parse in either orientation"}, tried

# --- Examples ---
if __name__ == "__main__":
    examples = [
        ["red","violet","orange","gold"],            # 27k ±5% (E24)
        ["gold","orange","violet","red"],           # reversed of above
        ["brown","black","black","red","brown"],    # 10k ±1% (E96)
        ["yellow","violet","brown","gold"],         # 470 Ω ±5% (E24)
        ["green","blue","black","black","brown"],    # 560 Ω ±1% (E96)
        ["yellow", "violet", "black", "brown", "brown"]
    ]
    for ex in examples:
        best, history = decode_resistor(ex)
        print(f"Input={ex} => {best}")

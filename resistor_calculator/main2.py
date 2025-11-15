def calculate_resistor_value(colors, num_bands):
    # Mapping of color names to numeric values
    color_mapping = {
        'black': 0, 'brown': 1, 'red': 2, 'orange': 3,
        'yellow': 4, 'green': 5, 'blue': 6, 'violet': 7,
        'gray': 8, 'white': 9
    }

    # Validate the number of colors provided matches the selected number of bands
    if len(colors) != num_bands:
        raise ValueError(f"Expected {num_bands} colors, but got {len(colors)} colors.")

    # Resistor calculation logic based on the number of bands
    if num_bands == 4:
        significant_digits = color_mapping[colors[0]] * 10 + color_mapping[colors[1]]
        multiplier = 10 ** color_mapping[colors[2]]
        tolerance = None

        if len(colors) == 4:
            tolerance_mapping = {'gold': 5, 'silver': 10}
            tolerance = tolerance_mapping.get(colors[3])

        resistance_value = significant_digits * multiplier

    elif num_bands == 5:
        significant_digits = color_mapping[colors[0]] * 100 + color_mapping[colors[1]] * 10 + color_mapping[colors[2]]
        multiplier = 10 ** color_mapping[colors[3]]
        tolerance = None

        if len(colors) == 5:
            tolerance_mapping = {'gold': 5, 'silver': 10}
            tolerance = tolerance_mapping.get(colors[4])

        resistance_value = significant_digits * multiplier

    elif num_bands == 6:
        significant_digits = color_mapping[colors[0]] * 100 + color_mapping[colors[1]] * 10 + color_mapping[colors[2]]
        multiplier = 10 ** color_mapping[colors[3]]
        tolerance = None

        if len(colors) == 6:
            temperature_coefficient_mapping = {'brown': 100, 'red': 50, 'orange': 15, 'yellow': 25}
            temperature_coefficient = temperature_coefficient_mapping.get(colors[4])
            tolerance_mapping = {'gold': 5, 'silver': 10}
            tolerance = tolerance_mapping.get(colors[5])

        resistance_value = significant_digits * multiplier

    return resistance_value, tolerance

print(calculate_resistor_value(["yellow", "violet", "black", "brown", "brown"], 5))
print(calculate_resistor_value(["brown", "brown", "black", "violet", "yellow"], 5))
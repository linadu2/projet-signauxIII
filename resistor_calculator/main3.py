def calculate_resistor_value(band: list) -> float:
    num_band = len(band)
    if num_band < 3 or num_band > 6:
        return 0.0

    digit = {
        "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
        "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9
    }

    multiplier = {
        "silver": 1e-2, "gold": 1e-1,
        "black": 1, "brown": 1e1, "red": 1e2, "orange": 1e3, "yellow": 1e4,
        "green": 1e5, "blue": 1e6, "violet": 1e7, "grey": 1e8, "white": 1e9
    }

    tolerance = {
        # none means 20%
        "silver": 10.0, "gold": 5.0, "brown": 1.0, "red": 2.0, "green": 0.5,
        "blue": 0.25, "violet": 0.1, "grey": 0.05, "none": 20.0
    }

    if num_band == 3:
        if band[0] == "black" or band[0] == "silver" or band[0] == "gold":
            return calculate_resistor_value(band[::-1])
        return (digit[band[0]] * 10 + digit[band[1]]) * multiplier[band[2]]

    if num_band == 4:
        if band[0] == "black" or band[3] == "black":
            return 0.0
        if band[0] == "silver" or band[0] == "gold":
            return calculate_resistor_value(band[::-1])
        if band[3] == "orange" or band[3] == "yellow" or band[3] == "white":
            return calculate_resistor_value(band[::-1])
        return (digit[band[0]] * 10 + digit[band[1]]) * multiplier[band[2]]

    if num_band == 5:
        if band[4] == "black" or band[4] == "orange" or band[4] == "yellow" or band[4] == "white":
            return calculate_resistor_value(band[::-1])
        if band[0] == "silver" or band[0] == "gold":
            return calculate_resistor_value(band[::-1])
        return (digit[band[0]] * 100 + digit[band[1]] * 10 + digit[band[2]]) * multiplier[band[3]]

    if num_band == 6:
        if band[4] == "black" or band[4] == "orange" or band[4] == "yellow" or band[4] == "white":
            return calculate_resistor_value(band[::-1])
        if band[0] == "silver" or band[0] == "gold":
            return calculate_resistor_value(band[::-1])

        return (digit[band[0]] * 100 + digit[band[1]] * 10 + digit[band[2]]) * multiplier[band[3]]





    return 0.0

print(calculate_resistor_value(["black", "brown", "brown", "red", "test", "test"]))
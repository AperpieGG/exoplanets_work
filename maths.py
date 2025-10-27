import math


def calculate_ratio(Rb_to_Rstar, a_to_Rstar):
    """
    Calculate the ratio (R_star + R_b) / a_b.
    """
    numerator = 1 + Rb_to_Rstar  # R_star + R_b = R_star * (1 + Rb/R_star)
    denominator = a_to_Rstar  # a = a_to_Rstar * R_star
    ratio = numerator / denominator
    return ratio


def propagate_ratio_error(Rb_to_Rstar, Rb_err_low, Rb_err_high, a_to_Rstar, a_err_low, a_err_high):
    """
    Propagate errors for the ratio (R_star + R_b) / a_b.
    """
    # Calculate the central ratio
    ratio_central = calculate_ratio(Rb_to_Rstar, a_to_Rstar)

    # Partial derivatives
    d_ratio_d_Rb = 1 / a_to_Rstar
    d_ratio_d_a = -(1 + Rb_to_Rstar) / (a_to_Rstar ** 2)

    # Propagate asymmetric errors
    ratio_err_low = math.sqrt((d_ratio_d_Rb * Rb_err_low) ** 2 + (d_ratio_d_a * a_err_low) ** 2)
    ratio_err_high = math.sqrt((d_ratio_d_Rb * Rb_err_high) ** 2 + (d_ratio_d_a * a_err_high) ** 2)

    return ratio_central, ratio_err_low, ratio_err_high


# Given values
Rb_to_Rstar = 0.09446  # Rb / R_star
Rb_to_Rstar_upper_error = 0.00067  # Upper error
Rb_to_Rstar_lower_error = -0.00067  # Lower error

# Given values for a / R_*
a_to_Rstar = 1 / 0.01119
a_to_Rstar_upper_error = 0.00021  # Upper error
a_to_Rstar_lower_error = -0.00021  # Lower error

# Calculate the ratio and its errors
ratio, ratio_err_low, ratio_err_high = propagate_ratio_error(
    Rb_to_Rstar, Rb_to_Rstar_lower_error, Rb_to_Rstar_upper_error,
    a_to_Rstar, a_to_Rstar_lower_error, a_to_Rstar_upper_error
)

# Calculate cosine of the angle
angle_in_degrees = 89.37
upper_error = 0.5
lower_error = -1.07
cosine_value = math.cos(math.radians(angle_in_degrees))
# Calculate upper and lower errors for cos(i)
cosine_upper_error = math.cos(math.radians(angle_in_degrees - upper_error)) - cosine_value
cosine_lower_error = cosine_value - math.cos(math.radians(angle_in_degrees + lower_error))

# Print the results
print(f"The ratio (R_star + R_b) / a_b is approximately: {ratio:.6f}, "
      f"lower limit {ratio_err_low:.6f}, upper limit {ratio_err_high:.6f} ")
print(f"The cos(i) of {angle_in_degrees} is: {cosine_value:.6f}, "
      f"lower limit: {cosine_lower_error:.6f}, upper limit: {cosine_upper_error:.6f}")
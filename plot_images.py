import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import median_abs_deviation


def plot_images():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times'
    plt.rcParams['font.size'] = 12

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


def bin_time_flux_error(time, flux, error, bin_fact):
    """
    Use reshape to bin light curve data, clip under filled bins
    Works with 2D arrays of flux and errors

    Note: under filled bins are clipped off the end of the series

    Parameters
    ----------
    time : array         of times to bin
    flux : array         of flux values to bin
    error : array         of error values to bin
    bin_fact : int
        Number of measurements to combine

    Returns
    -------
    times_b : array
        Binned times
    flux_b : array
        Binned fluxes
    error_b : array
        Binned errors

    Raises
    ------
    None
    """
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    # determine if 1 or 2d flux/err inputs
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        # assumed 2d with 1 row per star
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def load_json(file_path):
    """
    Load JSON file and return data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def bin_by_time_interval(time, flux, error, interval_minutes=5):
    """
    Bin data dynamically based on a specified time interval.

    Parameters
    ----------
    time :
        Array of time values.
    flux :
        Array of flux values corresponding to the time array.
    error :
        Array of error values corresponding to the flux array.
    interval_minutes : int, optional
        Time interval for binning in minutes (default is 5).

    Returns
    -------
    binned_time : array
        Binned time values (averages within each bin).
    binned_flux : array
        Binned flux values (averages within each bin).
    binned_error : array
        Binned error values (propagated within each bin).
    """
    interval_days = interval_minutes / (24 * 60)  # Convert minutes to days
    binned_time, binned_flux, binned_error = [], [], []

    start_idx = 0
    while start_idx < len(time):
        # Find the end index where the time difference exceeds the interval
        end_idx = start_idx
        while end_idx < len(time) and (time[end_idx] - time[start_idx]) < interval_days:
            end_idx += 1

        # Bin the data in the current interval
        binned_time.append(np.mean(time[start_idx:end_idx]))
        binned_flux.append(np.mean(flux[start_idx:end_idx]))
        binned_error.append(
            np.sqrt(np.sum(error[start_idx:end_idx] ** 2)) / (end_idx - start_idx)
        )

        # Move to the next interval
        start_idx = end_idx

    return np.array(binned_time), np.array(binned_flux), np.array(binned_error)


def remove_outliers(time, flux, flux_err, VALUE,  air_mass=None, zero_point=None):
    """
    Remove massive outliers in 3 rounds of clipping.

    Parameters
    ----------
    time : array
        Time values.
    flux : array
        Flux values.
    flux_err : array
        Flux error values.
    air_mass : array, optional
        Air mass values. Default is None.
    zero_point : array, optional
        Zero point values. Default is None.
    VALUE : float
        Threshold multiplier for median absolute deviation.

    Returns
    -------
    n_time : array
        Non-outlier time values.
    n_flux : array
        Non-outlier flux values.
    n_flux_err : array
        Non-outlier flux error values.
    n_air_mass : array or None
        Non-outlier air mass values, or None if air_mass was not provided.
    n_zero_point : array or None
        Non-outlier zero point values, or None if zero_point was not provided.
    """
    n_time = np.copy(time)
    n_flux = np.copy(flux)
    n_flux_err = np.copy(flux_err)
    n_air_mass = np.copy(air_mass) if air_mass is not None else None
    n_zero_point = np.copy(zero_point) if zero_point is not None else None

    for _ in range(3):
        mad = median_abs_deviation(n_flux)
        loc = np.where((n_flux < np.median(n_flux) + VALUE * mad) & (n_flux > np.median(n_flux) - VALUE * mad))[0]

        # Update time, flux, and flux_err with non-outlier values
        n_time = n_time[loc]
        n_flux = n_flux[loc]
        n_flux_err = n_flux_err[loc]
        if n_air_mass is not None:
            n_air_mass = n_air_mass[loc]
        if n_zero_point is not None:
            n_zero_point = n_zero_point[loc]

        # If no outliers were removed in this round, return the filtered arrays
        if len(n_time) == len(time):
            return n_time, n_flux, n_flux_err, n_air_mass, n_zero_point

    return n_time, n_flux, n_flux_err, n_air_mass, n_zero_point

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from plot_images import *

plot_images()

# --- Settings ---
path = '/Users/u5500483/Downloads/TIC-453147896_NGTS/'
filenames = glob.glob(path + '*.fits')
print(filenames)

all_time = []
all_flux = []
all_err = []

# --- Load all FITS files ---
for fname in filenames:
    hdul = fits.open(fname)
    data = hdul[4].data
    # print the avaliable columns
    print(f"Columns in {fname}: {data.columns.names}")
    t = data['BJD'] - 2457000
    f = data['FLUX_BSPROC']
    e = data['FLUX_BSPROC_ERR']
    hdul.close()

    all_time.append(t)
    all_flux.append(f)
    all_err.append(e)

# Concatenate
time = np.concatenate(all_time)
flux = np.concatenate(all_flux)
flux_err = np.concatenate(all_err)

# Remove NaNs
good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
time = time[good]
flux = flux[good]
flux_err = flux_err[good]

# --- Sort by time ---
order = np.argsort(time)
time = time[order]
flux = flux[order]
flux_err = flux_err[order]

# --- Identify observing segments ---
time_diff = np.diff(time)
gap_threshold = 2 / 24  # 2 hours in days
segment_breaks = np.where(time_diff > gap_threshold)[0]
n_segments = len(segment_breaks) + 1
print(f"Number of observing segments (nights): {n_segments}")

segment_edges = np.concatenate(([0], segment_breaks + 1, [len(time)]))


# --- Bin function with errors ---
def bin_segment(t, f, e, bin_minutes=5):
    bin_days = bin_minutes / 1440  # convert minutes to days
    bins = np.arange(t[0], t[-1] + bin_days, bin_days)
    digitized = np.digitize(t, bins)

    binned_time = []
    binned_flux = []
    binned_err = []

    for i in range(1, len(bins)):
        mask = digitized == i
        if np.any(mask):
            binned_time.append(np.mean(t[mask]))
            binned_flux.append(np.mean(f[mask]))
            # Combine errors in quadrature
            binned_err.append(np.sqrt(np.sum(e[mask]**2)) / np.sum(mask))

    return np.array(binned_time), np.array(binned_flux), np.array(binned_err)


# --- Bin each segment separately ---
binned_time_all = []
binned_flux_all = []
binned_err_all = []

for i in range(n_segments):
    start, end = segment_edges[i], segment_edges[i + 1]
    t_seg = time[start:end]
    f_seg = flux[start:end]
    e_seg = flux_err[start:end]

    t_b, f_b, e_b = bin_segment(t_seg, f_seg, e_seg, bin_minutes=5)

    binned_time_all.append(t_b)
    binned_flux_all.append(f_b)
    binned_err_all.append(e_b)

# Concatenate all binned segments
binned_time = np.concatenate(binned_time_all)
binned_flux = np.concatenate(binned_flux_all)
binned_err = np.concatenate(binned_err_all)

# -------------------------------------------------
# Transit ephemeris
# -------------------------------------------------
t0 = 2460098.8315131348 - 2457000
period = 58.204721

t_min, t_max = np.nanmin(time), np.nanmax(time)
n_before = int(np.floor((t_min - t0) / period))
n_after = int(np.ceil((t_max - t0) / period))
transit_times = t0 + np.arange(n_before, n_after + 1) * period

# --- Plot binned light curve with error bars ---
# --- Remove outliers from the binned light curve ---
VALUE = 10  # threshold in MAD units, adjust if needed
binned_time, binned_flux, binned_err, _, _ = remove_outliers(
    binned_time, binned_flux, binned_err, VALUE
)

plt.figure(figsize=(12, 4))
plt.errorbar(
    binned_time, binned_flux, yerr=binned_err,
    fmt='.', color='blue', alpha=0.6)

# Plot transit markers
tolerance = 0.01  # ~14.4 minutes
for tt in transit_times:
    has_data = np.any(np.abs(binned_time - tt) < tolerance)
    color = 'red' if has_data else 'black'
    marker = 'v'
    plt.plot(tt, 0.9825, marker=marker, color=color, markersize=8, alpha=0.8)


plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Relative Flux")
plt.ylim(0.982, 1.019)
plt.xlim(3190, 3350)
plt.savefig(path + 'ngts_sectors.pdf', dpi=100, bbox_inches='tight')
plt.show()


t0 = 2460098.8315131348 - 2457000  # reference transit
period = 58.204721                  # days
# Find last observed time
t_max = np.nanmax(time)  # assuming 'time' is your array of observations

# Compute number of periods after t_max
n_next = 10  # number of future transits to print
n_start = int(np.ceil((t_max - t0) / period))  # first transit after last observation

# Next 10 transits
next_transits = t0 + (n_start + np.arange(n_next)) * period
print("Next 10 transit times (BJD - 2457000):")
for i, tt in enumerate(next_transits, 1):
    print(f"{i}: {tt:.5f}")
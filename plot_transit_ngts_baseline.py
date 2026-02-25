import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from plot_images import *
import batman
import matplotlib.gridspec as gridspec


plot_images()

# --- Define transit parameters ---
params = batman.TransitParams()
params.t0 = 2460098.8315131348 - 2457000   # reference mid-transit
params.per = 58.204721                      # orbital period (days)
params.rp = 0.09675                          # Rp/Rs
params.a = 57.26                             # a/Rs
params.inc = 88.968                          # inclination (deg)
params.ecc = 0.370                           # eccentricity
params.w = 97.5                              # longitude of periastron (deg)
params.limb_dark = "quadratic"
params.u = [0.336, 0.246]

# --- Settings ---
path = '/Users/u5500483/Downloads/TIC-453147896_NGTS/old_detrending/'
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

t0 = 2459225.760694092 - 2457000  # reference transit
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


# --- Create a fine time array spanning your whole dataset ---
time_model = np.linspace(np.min(binned_time), np.max(binned_time), 50000)

# --- Initialize batman model for full time array ---
m_full = batman.TransitModel(params, time_model)
model_flux_full = m_full.light_curve(params)

# --- Plot binned data ---
plt.figure(figsize=(12, 4))
plt.errorbar(
    binned_time,
    binned_flux,
    yerr=binned_err,
    fmt='.',
    color='blue',
    alpha=0.5,
    label='NGTS binned data'
)

# --- Overplot full transit model ---
plt.plot(
    time_model,
    model_flux_full,
    color='red',
    lw=2,
    label='Transit model'
)

# Optional: mark predicted transit centers
n_before = int(np.floor((np.min(binned_time) - params.t0) / params.per))
n_after = int(np.ceil((np.max(binned_time) - params.t0) / params.per))
transit_times = params.t0 + np.arange(n_before, n_after + 1) * params.per

# Plot transit markers
tolerance = 0.01  # ~14.4 minutes
for tt in transit_times:
    has_data = np.any(np.abs(binned_time - tt) < tolerance)
    color = 'red' if has_data else 'black'
    marker = 'v'
    plt.plot(tt, 0.9825, marker=marker, color=color, markersize=8, alpha=0.8)


plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Relative Flux")
plt.ylim(39300, 40500)
plt.xlim(3190, 3350)
# plt.legend()
plt.show()

# Compute predicted transit times
n_before = int(np.floor((np.min(binned_time) - params.t0) / params.per))
n_after  = int(np.ceil((np.max(binned_time) - params.t0) / params.per))
transit_times = params.t0 + np.arange(n_before, n_after + 1) * params.per

# Keep only transits that actually have data
tolerance = 0.01
transits_with_data = [tt for tt in transit_times if np.any(np.abs(binned_time - tt) < tolerance)]
transits_to_plot = transits_with_data[:3]  # first three transits

# --- Create figure with 2 rows, 3 columns ---
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1], hspace=0.2, wspace=0.3)

# --- Top row: full lightcurve (span all 3 columns) ---
ax0 = fig.add_subplot(gs[0, :])
ax0.errorbar(binned_time, binned_flux, yerr=binned_err, fmt='.', color='blue', alpha=0.5, label='NGTS binned data')
ax0.plot(time_model, model_flux_full, color='red', lw=2, label='Transit model')

# Mark transit centers
for tt in transit_times:
    has_data = np.any(np.abs(binned_time - tt) < tolerance)
    color = 'red' if has_data else 'black'
    ax0.plot(tt, 0.9825, marker='v', color=color, markersize=8, alpha=0.8)

ax0.set_xlabel("Time (BJD - 2457000)")
ax0.set_ylabel("Relative Flux")
ax0.set_xlim(3190, 3350)
ax0.set_ylim(0.982, 1.019)
# ax0.legend()

# --- Bottom row: 3 columns for zoomed transits ---
ax_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]  # bottom row, 3 columns

# Manually specified x-limits for each subplot
xlims = [(3213.24, 3217.24), (3271.45, 3275.45), (3329.65, 3333.65)]

for i, ax in enumerate(ax_bottom):
    # Plot full dataset and model
    ax.errorbar(binned_time, binned_flux, yerr=binned_err, fmt='.', color='blue', alpha=0.5)
    ax.plot(time_model, model_flux_full, color='red', lw=2)

    # Mark transit centers
    # Mark transit centers
    for tt in transit_times:
        if i < 2:   # first two subplots
            marker_color = 'black'
        else:       # third subplot
            marker_color = 'red'
        ax.plot(tt, 0.9835, marker='v', color=marker_color, markersize=8, alpha=0.8)

    # Set y-axis limits
    ax.set_ylim(0.982, 1.019)

    # Set manual x-limits
    ax.set_xlim(xlims[i])
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Relative Flux")

plt.show()



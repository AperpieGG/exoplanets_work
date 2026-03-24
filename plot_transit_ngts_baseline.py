import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from plot_images import *
import batman
import matplotlib.gridspec as gridspec
from astropy.time import Time


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
all_bkg = []  # store background for filtering
all_zp = []  # store ZERO_POINT for filtering


# --- Load all FITS files ---
updated_path = '/Users/u5500483/Downloads/TIC-453147896_NGTS/'

for fname in filenames:

    base_name = fname.split('/')[-1]
    updated_file = updated_path + base_name

    # Open OLD file (photometry)
    hdul_old = fits.open(fname)
    data_old = hdul_old[3].data

    # Open UPDATED file (contains CLEAN column)
    hdul_new = fits.open(updated_file)
    data_new = hdul_new[4].data

    # CLEAN mask
    clean_mask = data_new['CLEAN'] == 1

    # Extract data using CLEAN mask
    t = data_old['BJD'][clean_mask] - 2457000
    f = data_old['FLUX_BSPROC'][clean_mask]
    e = data_old['FLUX_BSPROC_ERR'][clean_mask]
    bkg = data_old['FLUX_BKG'][clean_mask]  # background
    # Open OLD file (photometry)
    hdul_zp = fits.open(fname)
    hdul_zp = hdul_zp[2].data
    zp = hdul_zp['ZERO_POINT'][clean_mask]  # zero point

    hdul_old.close()
    hdul_new.close()

    all_time.append(t)
    all_flux.append(f)
    all_err.append(e)
    all_bkg.append(bkg)
    all_zp.append(zp)

# Concatenate all arrays
time = np.concatenate(all_time)
flux = np.concatenate(all_flux)
flux_err = np.concatenate(all_err)
bkg = np.concatenate(all_bkg)
zp = np.concatenate(all_zp)


# Remove NaNs
good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & np.isfinite(bkg) & np.isfinite(zp)
time = time[good]
flux = flux[good]
flux_err = flux_err[good]
bkg = bkg[good]
zp = zp[good]

# --- Linear detrending based on flux vs background ---
slope = -8.081030e-08
intercept = 1.001081

# Compute the background trend
flux_trend = slope * bkg + intercept

# Detrend the flux
flux = flux / flux_trend

# Optionally, adjust flux_err too (divide by same trend)
flux_err = flux_err / flux_trend

# --- Apply filters ---
# supposed to filter bkg but will not do since slope fix this
mask_bkg = bkg <= 400000
mask_zp = (zp > -0.06) & (zp < 0.2)  # keep ZERO_POINT in (-0.06, 0.2)

# Combine both masks
final_mask = mask_bkg & mask_zp

time = time[final_mask]
flux = flux[final_mask]
flux_err = flux_err[final_mask]
bkg = bkg[final_mask]
zp = zp[final_mask]
# Then remove NaNs
good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

time = time[good]
flux = flux[good]
flux_err = flux_err[good]

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

t0 = 2459225.760694092 - 2457000  # reference transit
period = 58.204721                  # days
# Find last observed time
t_max = np.nanmax(time)  # assuming 'time' is your array of observations

# Compute number of periods after t_max
n_next = 15  # number of future transits to print
n_start = int(np.ceil((t_max - t0) / period))  # first transit after last observation

# Next 10 transits
next_transits = t0 + (n_start + np.arange(n_next)) * period
# print("Next 10 transit times (BJD - 2457000):")
# for i, tt in enumerate(next_transits, 1):
#     print(f"{i}: {tt:.5f}")


# --- Create a fine time array spanning your whole dataset ---
time_model = np.linspace(np.min(binned_time), np.max(binned_time), 50000)

# --- Initialize batman model for full time array ---
m_full = batman.TransitModel(params, time_model)
model_flux_full = m_full.light_curve(params)


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
ax0.set_ylim(0.987, 1.008)
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
    ax.set_ylim(0.987, 1.008)

    # Set manual x-limits
    ax.set_xlim(xlims[i])
    ax.set_xlabel("Time (BJD - 2457000)")
    # only label y-label for the first plot
    if i == 0:
        ax.set_ylabel("Relative Flux")

plt.savefig("ngts_sectors.pdf", bbox_inches='tight')
plt.show()

# Convert to full BJD
next_transits_full_bjd = next_transits + 2457000

print("Next 10 transit times:")
for i, tt in enumerate(next_transits_full_bjd, 0):

    # Create astropy Time object (BJD is basically JD in TDB scale)
    t_obj = Time(tt, format='jd', scale='tdb')

    # Convert to UTC calendar date
    utc_time = t_obj.utc

    print(f"{i}: BJD = {tt:.5f}  |  UTC = {utc_time.iso}")

print("Previous 10 transit times:")
for i in range(0, 12):
    prev_tt = t0 - i * period  # go backwards in time

    # Convert to full BJD
    prev_tt_full = prev_tt + 2457000

    t_obj = Time(prev_tt_full, format='jd', scale='tdb')
    utc_time = t_obj.utc

    print(f"-{i}: BJD = {prev_tt_full:.5f}  |  UTC = {utc_time.iso}")


print("\nTransits that are covered by the data:")

tolerance = 0.02  # ~30 minutes (adjust if needed)

for tt in transit_times:
    # Check if this transit is within your data
    has_data = np.any(np.abs(binned_time - tt) < tolerance)

    if has_data:
        tt_full = tt + 2457000  # convert to full BJD

        t_obj = Time(tt_full, format='jd', scale='tdb')
        utc_time = t_obj.utc

        print(f"BJD = {tt_full:.10f}  |  UTC = {utc_time.iso}")
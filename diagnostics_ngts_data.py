"""
Script that loads all available NGTS data for TIC 453147896
and performs diagnostic analysis, including:

- Identifying individual observing segments
- Binning the time-series data
- Plotting the light curve with transit markers

The goal is to inspect data quality and visualize the transit events.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from plot_images import *
import batman
import matplotlib.gridspec as gridspec


plot_images()


# --- Settings ---
path = '/Users/u5500483/Downloads/TIC-453147896_NGTS/old_detrending/'
filenames = glob.glob(path + '*.fits')
print(filenames)

# --- Storage for concatenated arrays ---
all_time = []
all_flux = []
all_err = []
all_bkg = []
all_airmass = []
all_zp = []

# --- Load all FITS files ---
updated_path = '/Users/u5500483/Downloads/TIC-453147896_NGTS/'  # updated directory

for fname in filenames:

    base_name = fname.split('/')[-1]
    updated_file = updated_path + base_name

    # --- Open OLD file ---
    hdul_old = fits.open(fname)
    data_old = hdul_old[3].data  # main photometry table

    # --- Open UPDATED file (contains CLEAN column) ---
    hdul_new = fits.open(updated_file)
    data_new = hdul_new[4].data
    print(f"Columns in {fname}: {data_old.columns.names}")

    # --- CLEAN mask ---
    clean_mask = data_new['CLEAN'] == 1

    # --- Extract all columns from OLD table using CLEAN mask ---
    t = data_old['BJD'][clean_mask] - 2457000
    f = data_old['FLUX_BSPROC'][clean_mask]
    e = data_old['FLUX_BSPROC_ERR'][clean_mask]
    bkg = data_old['FLUX_BKG'][clean_mask]
    hdul_air = fits.open(fname)
    data_air = hdul_air[2].data
    print(f"Columns in {fname}: {data_air.columns.names}")
    airmass_array = data_air['AIRMASS'][clean_mask]
    zero_point_array = data_air['ZERO_POINT'][clean_mask]
    hdul_air.close()

    # --- Append to master lists ---
    all_time.append(t)
    all_flux.append(f)
    all_err.append(e)
    all_bkg.append(bkg)
    all_airmass.append(airmass_array)
    all_zp.append(zero_point_array)

    # --- Close HDUs ---
    hdul_old.close()
    hdul_new.close()

# --- Concatenate all segments ---
time = np.concatenate(all_time)
flux = np.concatenate(all_flux)
flux_err = np.concatenate(all_err)
bkg = np.concatenate(all_bkg)
airmass = np.concatenate(all_airmass)
zero_point = np.concatenate(all_zp)

# --- Remove NaNs (safety) ---
mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & np.isfinite(bkg) & np.isfinite(airmass) & np.isfinite(zero_point)
time = time[mask]
flux = flux[mask]
flux_err = flux_err[mask]
bkg = bkg[mask]
airmass = airmass[mask]

# ===============================
# 1️⃣ Flux vs Time
# ===============================
plt.figure()
plt.scatter(time, flux, s=5, color='blue')
plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Flux")
plt.title("Flux vs Time (CLEAN = 1)")
plt.show()

# ===============================
# 2️⃣ Flux Error vs Time
# ===============================
plt.figure()
plt.scatter(time, flux_err, s=5, color='blue')
plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Flux Error")
plt.title("Flux Error vs Time (CLEAN = 1)")
# plt.xlim(3320, 3340)
plt.show()

# ===============================
# 3️⃣ Sky Background vs Time
# ===============================
plt.figure()
plt.scatter(time, bkg, s=5, color='blue')
plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Sky Background (FLUX_BKG)")
plt.title("Sky Background vs Time (CLEAN = 1)")
plt.show()

# ===============================
# 4️⃣ Airmass vs Time
# ===============================
plt.figure()
plt.scatter(time, airmass, s=5, color='blue')
plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Airmass")
plt.title("Airmass vs Time (CLEAN = 1)")
# plt.xlim(3320, 3340)
plt.show()

# ===============================
# 5️⃣ Zero Point vs Time
# ===============================
plt.figure()
plt.scatter(time, zero_point, s=5, color='blue')
plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Zero Point (ZP)")
plt.title("Zero Point vs Time (CLEAN = 1)")
plt.ylim(-0.06, 0.2)
plt.show()


# ===============================
# 6️⃣ Flux vs Sky Background
# ===============================
plt.figure()
plt.scatter(bkg, flux, s=5, color='blue')
plt.xlabel("Sky Background (FLUX_BKG)")
plt.ylabel("Flux")
plt.title("Flux vs Sky Background (FLUX_BKG) (CLEAN = 1)")
plt.show()


# ===============================
# 6️⃣ Flux vs Sky Background + Linear Fit
# ===============================
plt.figure()
plt.scatter(bkg, flux, s=5, color='blue', label='Data')

# --- Linear fit ---
# y = m*x + c
slope, intercept = np.polyfit(bkg, flux, 1)
print(f"Slope of flux vs background: {slope:.6e}")
print(f"Intercept of flux vs background: {intercept:.6e}")

# --- Plot fitted line ---
x_fit = np.linspace(np.min(bkg), np.max(bkg), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color='red', lw=2, label=f'Linear fit: slope={slope:.2e}')

plt.xlabel("Sky Background (FLUX_BKG)")
plt.ylabel("Flux")
plt.title("Flux vs Sky Background (FLUX_BKG) (CLEAN = 1)")
plt.legend()
plt.show()
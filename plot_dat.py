import pandas as pd
import matplotlib.pyplot as plt
from plot_images import plot_images, bin_by_time_interval, remove_outliers, bin_time_flux_error
import numpy as np
import batman

interval = 2 # in minutes
params = batman.TransitParams()
params.t0 = 2460971.902088  # time of inferior conjunction (BJD)
params.per = 58.20462   # orbital period (days)
params.rp = 0.09446  # planet radius (in units of stellar radii)
params.a = 87.19777823858355  # semi-major axis (in units of stellar radii)
params.inc = 89.636  # orbital inclination (degrees)
params.ecc = 0.33  # eccentricity
params.w = 0  # longitude of periastron (degrees)
params.u = [0.3443385582043821, 0.3820056981486999]  # limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  # limb darkening model            # generate model flux


plot_images()
# # Path to your .dat file
path = "/Users/u5500483/Documents/GitHub/exoplanets_work/data/"
# filename = path + "TIC-453147896_NGTS_2025-10-22_master_apers_bsproc_lc.dat"
#
# # Step 1: Find header line automatically (allowing leading # or spaces)
# header_line = None
# with open(filename) as f:
#     for i, line in enumerate(f):
#         clean = line.strip().lstrip("#").strip()  # remove any leading '#' or spaces
#         if clean.startswith("ActionID"):
#             header_line = i
#             break
#
# if header_line is None:
#     raise ValueError("Couldn't find the header line (no line containing 'ActionID')")
#
# # Step 2: Read file using the detected header
# df = pd.read_csv(filename, delim_whitespace=True, skiprows=header_line)
#
# # Step 3: Check loaded columns
# # print("Detected columns:", list(df.columns))
# # print(df.head())
#
# time = np.array(df["BJD"], dtype=float)
# flux = np.array(df["FluxNorm"], dtype=float)
# flux_err = np.array(df["FluxNormErr"], dtype=float)
#
# # remove outliers
# time_ngts, flux_ngts, flux_err_ngts, _, _ = remove_outliers(time, flux, flux_err, air_mass=None, zero_point=None)
#
# time_binned_ngts, flux_binned_ngts, flux_binned_err_ngts = bin_by_time_interval(time_ngts, flux_ngts, flux_err_ngts, interval)
#
# mean_first5 = np.mean(flux_binned_ngts[:4])
# flux_binned_ngts /= mean_first5
# # # Step 4: Plot normalized flux vs BJD
# # plt.figure()
# # # plt.errorbar(df["BJD"], df["FluxNorm"], yerr=df["FluxNormErr"], fmt="ro", markersize=2)
# # plt.errorbar(time_ngts, flux_ngts, yerr=flux_err_ngts, fmt='ro', markersize=1, alpha=0.2)
# # plt.plot(time_binned_ngts, flux_binned_ngts, 'bo')
# # plt.xlabel("BJD")
# # plt.ylabel("Normalized Flux")
# # plt.title("Light Curve: TIC-453147896")
# # plt.tight_layout()
# # plt.show()

filename_LOOK = path + "TIC453147896.01_20251023_LO_OSC (rgb)_measurement.tbl"

# Read file with no header
df_look = pd.read_csv(filename_LOOK, delim_whitespace=True, header=None, comment="#")

# Assign proper column names
df_look.columns = ["BJD_TDB", "Rel_flux_T1", "err_rel_flux_T1", "dX", "dY", "FWHM", "Sky", "Airmass"]

# Keep only the two columns you want
df_look = df_look[["BJD_TDB", "Rel_flux_T1", "err_rel_flux_T1"]]

# Convert to numpy arrays
time_look = df_look["BJD_TDB"].astype(float).to_numpy()
flux_look = df_look["Rel_flux_T1"].astype(float).to_numpy()
flux_err_look = df_look["err_rel_flux_T1"].astype(float).to_numpy()

# remove outliers
time_look, flux_look, flux_err_look, _, _ = remove_outliers(time_look, flux_look, flux_err_look, 5, air_mass=None, zero_point=None)

time_binned_look, flux_binned_look, flux_binned_err_look = bin_by_time_interval(time_look, flux_look, flux_err_look, interval)


mean_first5_look = np.mean(flux_binned_look[-4:])
flux_binned_look /= mean_first5_look

filenames = [
    path + "tic453147896_20251023_ulmt_rp_bjd-flux-err-am-detrended.dat",
    path + "TIC453147896-11_20251023_LCO-CTIO-0m35_rp_bjd-flux-err-am-detrended.dat",
    path + "TIC453147896-11_UT20251023_Acton-Sky-Portal-0.36m_rp_bjd-flux-err-am-detrended.dat",
    path + "TIC453147896.11_20251023_OACC-CAO_i2_KC_bjd-flux-err-xCen_detrended.dat",
    path + "TIC453147896-11_20251023_KeplerCam_ip.dat"
]

colors = ['r', 'g', 'b', 'y', 'm']
labels = ['ULMT', 'LCO', 'ACTON', 'OACC-CAO', 'KeplerCam']

special_keplercam = path + "TIC453147896-11_20251023_KeplerCam_ip.dat"

plt.figure()

# -------------------------------------------------------------------
# Combine all times for batman model
# -------------------------------------------------------------------
all_times = np.concatenate([time_binned_look, time_binned_look])

for file in filenames:
    df = pd.read_csv(file, delim_whitespace=True)
    df.columns = df.columns.str.strip().str.lstrip('#')

    # SPECIAL CASE — KeplerCam file
    if file == special_keplercam:
        time = df["BJD_TDB"].to_numpy()
    else:
        time = df["BJD_TDB"].to_numpy()

    all_times = np.concatenate([all_times, time])

all_times = np.sort(all_times)

m = batman.TransitModel(params, all_times)
flux_model = m.light_curve(params)

# -------------------------------------------------------------------
# PLOT ALL DATASETS
# -------------------------------------------------------------------
for file, color, label in zip(filenames, colors, labels):
    df = pd.read_csv(file, delim_whitespace=True)
    df.columns = df.columns.str.strip().str.lstrip('#')

    # SPECIAL CASE FOR KEPLERCAM
    if file == special_keplercam:
        time = df["BJD_TDB"].astype(float).to_numpy()
        flux = df["rel_flux_T1_n"].astype(float).to_numpy()
        flux_err = df["rel_flux_err_T1_n"].astype(float).to_numpy()

        # Detrend / normalize using the LAST 5 points
        if len(flux) >= 5:
            norm_factor = np.mean(flux[-14:])
        else:
            norm_factor = np.mean(flux)  # fallback for very small datasets

        flux = flux / norm_factor
        flux_err = flux_err / norm_factor
    else:
        time = df["BJD_TDB"].astype(float).to_numpy()
        flux = df["rel_flux_T1_dfn"].astype(float).to_numpy()
        flux_err = df["rel_flux_err_T1_dfn"].astype(float).to_numpy()

    time, flux, flux_err, _, _ = remove_outliers(time, flux, flux_err, 7, air_mass=None, zero_point=None)

    time_binned, flux_binned, flux_binned_err = bin_by_time_interval(time, flux, flux_err, interval_minutes=interval)

    plt.errorbar(time_binned, flux_binned, yerr=flux_binned_err,
                 fmt='o', color=color, alpha=1, markersize=6,
                 label=f'{label} binned')

plt.errorbar(time_binned_look, flux_binned_look, yerr=flux_binned_err_look,
             fmt='o', color='purple', alpha=1, markersize=6,
             label='LOOK binned')

plt.plot(all_times, flux_model, '-', color='black', linewidth=2, label='Batman model')
plt.ylim(0.975, 1.015)
plt.xlabel("BJD")
plt.ylabel("Detrended Flux")
plt.legend()
plt.tight_layout()
plt.show()


# --- Save all binned data together ---
all_binned_time = []
all_binned_flux = []
all_binned_err = []

# Include LOOK data first
all_binned_time.extend(time_binned_look)
all_binned_flux.extend(flux_binned_look)
all_binned_err.extend(flux_binned_err_look)

for file, color, label in zip(filenames, colors, labels):
    df = pd.read_csv(file, delim_whitespace=True)
    df.columns = df.columns.str.strip().str.lstrip('#')

    # --- Special handling for KeplerCam file ---
    if "KeplerCam" in file:
        time = df["BJD_TDB"].astype(float).to_numpy()
        flux = df["rel_flux_T1_n"].astype(float).to_numpy()
        flux_err = df["rel_flux_err_T1_n"].astype(float).to_numpy()

        # Normalise using mean of last 5 points
        if len(flux) >= 5:
            norm_factor = np.mean(flux[-14:])
        else:
            norm_factor = np.mean(flux)
        flux = flux / norm_factor
        flux_err = flux_err / norm_factor
    else:
        time = df["BJD_TDB"].astype(float).to_numpy()
        flux = df["rel_flux_T1_dfn"].astype(float).to_numpy()
        flux_err = df["rel_flux_err_T1_dfn"].astype(float).to_numpy()

    # remove outliers
    time, flux, flux_err, _, _ = remove_outliers(time, flux, flux_err, 7, air_mass=None, zero_point=None)

    # bin the data
    time_binned, flux_binned, flux_binned_err = bin_by_time_interval(time, flux, flux_err, interval_minutes=interval)

    # append to combined arrays
    all_binned_time.extend(time_binned)
    all_binned_flux.extend(flux_binned)
    all_binned_err.extend(flux_binned_err)

# Convert to numpy arrays and sort by time
all_binned_time = np.array(all_binned_time)
all_binned_flux = np.array(all_binned_flux)
all_binned_err = np.array(all_binned_err)

sorted_indices = np.argsort(all_binned_time)
final_time = all_binned_time[sorted_indices]
final_flux = all_binned_flux[sorted_indices]
final_err = all_binned_err[sorted_indices]

# Save to CSV file
output_df = pd.DataFrame({
    '# time': final_time,
    'flux': final_flux,
    'flux_err': final_err
})

output_df.to_csv("sg1.csv", index=False)
print("✅ Saved combined binned data to sg1.csv")
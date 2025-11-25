import pandas as pd
import matplotlib.pyplot as plt
from plot_images import plot_images, bin_by_time_interval, remove_outliers, bin_time_flux_error
import numpy as np
import batman


interval = 5  # in minutes
plot_images()
# Path to your .dat file
path = "/Users/u5500483/Documents/GitHub/exoplanets_work/data/"
filename = path + "TIC-453147896_NGTS_2025-10-22_master_apers_bsproc_lc.dat"

# Step 1: Find header line automatically (allowing leading # or spaces)
header_line = None
with open(filename) as f:
    for i, line in enumerate(f):
        clean = line.strip().lstrip("#").strip()  # remove any leading '#' or spaces
        if clean.startswith("ActionID"):
            header_line = i
            break

if header_line is None:
    raise ValueError("Couldn't find the header line (no line containing 'ActionID')")

# Step 2: Read file using the detected header
df = pd.read_csv(filename, delim_whitespace=True, skiprows=header_line)

time = np.array(df["BJD"], dtype=float)
flux = np.array(df["FluxNorm"], dtype=float)
flux_err = np.array(df["FluxNormErr"], dtype=float)

# Check unique ActionIDs
unique_actions = df["ActionID"].unique()
print("Unique ActionIDs:", unique_actions)

# Plot each action separately
plt.figure()

for action in unique_actions:
    # Select the rows corresponding to this ActionID
    mask = df["ActionID"] == action
    time_action = df.loc[mask, "BJD"].values
    flux_action = df.loc[mask, "FluxNorm"].values
    flux_err_action = df.loc[mask, "FluxNormErr"].values
    print("Action:", action, "Number of points:", len(time_action))
    time_action, flux_action, flux_err_action, _, _ = remove_outliers(time_action, flux_action, flux_err_action)
    print("After removing outliers, points:", len(time_action))
    flux_action_mean = np.mean(flux_action[:150])
    flux_action /= flux_action_mean

    time_action_binned, flux_action_binned, flux_err_action_binned = bin_by_time_interval(time_action, flux_action,
                                                                                          flux_err_action, interval)

    # # Step 4: Plot normalized flux vs
    # Plot with error bars
    plt.errorbar(time_action_binned, flux_action_binned, flux_err_action_binned, fmt="o", label=action)


plt.xlabel("BJD")
plt.ylabel("Normalized Flux")
plt.title("Flux vs Time split by ActionID")
plt.legend()
plt.ylim(0.98, 1.01)
plt.tight_layout()
plt.show()


all_times = []
all_fluxes = []
all_errors = []

# --- Collect all binned points from each ActionID ---
for action in unique_actions:
    mask = df["ActionID"] == action
    time_action = df.loc[mask, "BJD"].values
    flux_action = df.loc[mask, "FluxNorm"].values
    flux_err_action = df.loc[mask, "FluxNormErr"].values

    # remove outliers
    time_action, flux_action, flux_err_action, _, _ = remove_outliers(time_action, flux_action, flux_err_action)

    # 5-min binning
    time_action_binned, flux_action_binned, flux_err_action_binned = bin_by_time_interval(
        time_action, flux_action, flux_err_action, interval_minutes=interval
    )

    # find all data points with time < 2460971.80
    mask = time_action_binned < 2460971.79

    # compute mean flux only from those points
    mean_flux_before = np.mean(flux_action_binned[mask])

    # normalize the flux
    flux_action_binned /= mean_flux_before

    all_times.append(time_action_binned)
    all_fluxes.append(flux_action_binned)
    all_errors.append(flux_err_action_binned)

# --- Concatenate everything ---
all_times = np.concatenate(all_times)
all_fluxes = np.concatenate(all_fluxes)
all_errors = np.concatenate(all_errors)

# --- Sort by time first ---
sort_idx = np.argsort(all_times)
all_times = all_times[sort_idx]
all_fluxes = all_fluxes[sort_idx]
all_errors = all_errors[sort_idx]

# --- Bin every 6 points (since 6 telescopes per timestamp) ---
N_bin = 6

# Ensure the total length is a multiple of 6
num_bins = len(all_times) // N_bin

final_time = []
final_flux = []
final_err = []

for i in range(num_bins):
    start = i * N_bin
    end = start + N_bin

    time_chunk = all_times[start:end]
    flux_chunk = all_fluxes[start:end]
    err_chunk = all_errors[start:end]

    # Average quantities
    final_time.append(np.mean(time_chunk))
    final_flux.append(np.mean(flux_chunk))

    # Combine errors in quadrature, divided by sqrt(N)
    combined_err = np.sqrt(np.sum(err_chunk**2)) / N_bin
    final_err.append(combined_err)

# Convert to numpy arrays
final_time = np.array(final_time)
final_flux = np.array(final_flux)
final_err = np.array(final_err)

print(f"Binned {len(all_times)} points into {len(final_time)} bins of {N_bin} points each.")

# --- Plot final stacked light curve ---
plt.figure()
plt.errorbar(final_time, final_flux, yerr=final_err, fmt='o', color='orange')
plt.xlabel("BJD")
plt.ylabel("Normalized Flux")
plt.title("Final Stacked Light Curve Across All Actions")
plt.ylim(0.98, 1.01)
plt.tight_layout()
plt.show()

# save this in a csv file as well
# output_df = pd.DataFrame({
#     '# time': final_time,
#     'flux': final_flux,
#     'flux_err': final_err
# })
# output_df.to_csv(path + "NGTS_6.csv", index=False)

# # Data of NGTS TIC-453147896 planet candidate from Jan 2024
# filename_2024 = path + "NGTS_1.csv"
# df_2024 = pd.read_csv(filename_2024)
# df_2024.columns = df_2024.columns.str.strip().str.replace('# ', '')
# # Now columns become: ['time', 'flux', 'flux_err']
#
# time_2024 = np.array(df_2024["time"], dtype=float)
# flux_2024 = np.array(df_2024["flux"], dtype=float)
# flux_err_2024 = np.array(df_2024["flux_err"], dtype=float)
#
# # time_2024, flux_2024, flux_err_2024, _, _ = remove_outliers(time_2024, flux_2024, flux_err_2024, air_mass=None, zero_point=None)
# time_binned_2024, flux_binned_2024, flux_binned_err_2024 = bin_by_time_interval(time_2024, flux_2024, flux_err_2024, interval)
# mean_first5_2024 = np.mean(flux_binned_2024[:5])
# flux_binned_2024 /= mean_first5_2024
# plt.figure()
# plt.errorbar(time_binned_2024, flux_binned_2024, yerr=flux_binned_err_2024, fmt='go', label='NGTS Jan 2024')
# plt.xlabel("BJD")
# plt.ylabel("Normalized Flux")
# plt.title("NGTS TIC-453147896 Light Curve Jan 2024")
# plt.ylim(0.98, 1.01)
# plt.tight_layout()
# plt.show()
#
#



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from plot_images import plot_images, bin_by_time_interval

plot_images()

# --- Read CSVs ---
filename = 'data/TIC-453147896_SPOC_FLUX_NORMALISED.csv'
filename_ngts = 'data/NGTS_1.csv'

df = pd.read_csv(filename, header=None, names=['time', 'flux', 'flux_err'], comment='#')
df_ngts = pd.read_csv(filename_ngts, header=None, names=['time', 'flux', 'flux_err'], comment='#')

time = df['time'].to_numpy()
flux = df['flux'].to_numpy()
flux_err = df['flux_err'].to_numpy()

time_ngts = df_ngts['time'].to_numpy()
flux_ngts = df_ngts['flux'].to_numpy()
flux_err_ngts = df_ngts['flux_err'].to_numpy()


# --- Handle BJD offset ---
def guess_xlabels(time):
    jd_moon_landing = 2440422.5
    if np.min(time) < jd_moon_landing:
        return time, 'Time [days]'
    else:
        return time - 2457000, 'Time (BJD$_\mathrm{TDB}$ - 2,459,000)'

# Save raw (unbinned) time for gap analysis
time_ngts, flux_ngts, flux_err_ngts = bin_by_time_interval(time_ngts, flux_ngts, flux_err_ngts, 30)
time, flux, flux_err = bin_by_time_interval(time, flux, flux_err, 30)

time_plot, xlabel = guess_xlabels(time)
time_plot_ngts, xlabel_ngts = guess_xlabels(time_ngts)


def plot_tess(time, y, yerr=None, sharey=False, **kwargs):
    """
    Plot TESS light curve in separate subplots per sector,
    marking expected transits.

    Parameters
    ----------
    time : array-like
        BJD_TDB time stamps.
    y : array-like
        Flux values.
    yerr : array-like, optional
        Flux uncertainties.
    sharey : bool
        Share y-axis between subplots.
    kwargs : dict
        Additional plotting kwargs (color, markersize, etc.)
    """

    # --- Sanity check ---
    jd_moon_landing = 2440422.5
    if np.nanmin(time) < jd_moon_landing:
        raise ValueError("Time array does not seem to be in BJD_TDB units.")

    # --- Load TESS orbit data ---
    csv_path = '/Users/u5500483/Downloads/Paper_IV/TESS_orbit_times.csv'
    df = pd.read_csv(csv_path, comment='#')

    sectors_to_plot = []
    sector_bounds = []

    for s in sorted(df['Sector'].unique()):
        rows = df[df['Sector'] == s]
        if len(rows) < 2:
            print(f"Skipping sector {s}: incomplete rows")
            continue

        t_start = Time(rows['Start of Orbit'].min()).jd
        t_end = Time(rows['End of Orbit'].max()).jd

        ind = np.where((time >= t_start) & (time <= t_end))[0]

        if len(ind) > 0:
            sectors_to_plot.append(s)
            sector_bounds.append((t_start, t_end))

    # --- Shift time by 2457000 for plotting ---
    time_shifted = time - 2457000

    # --- Transit information ---
    t0 = 2460680.87881  # first observed transit (BJD)
    period = 58.20462  # days

    # Generate all expected transit times within observed time range
    t_min, t_max = np.nanmin(time), np.nanmax(time)
    n_before = int(np.floor((t_min - t0) / period))
    n_after = int(np.ceil((t_max - t0) / period))
    transit_times = t0 + np.arange(n_before, n_after + 1) * period

    # --- Shift transit times to BJD - 2457000 for plotting ---
    transit_times_shifted = transit_times - 2457000

    # --- Create subplots ---
    N = len(sectors_to_plot)
    fig, axes = plt.subplots(N, 1, figsize=(12, 3 * N), sharey=sharey, tight_layout=True)

    if N == 1:
        axes = [axes]

    # --- Plot each sector ---
    for ax, s, bounds in zip(axes, sectors_to_plot, sector_bounds):
        t0_sec, t1_sec = bounds
        ind = np.where((time >= t0_sec) & (time <= t1_sec))[0]

        t0_shifted, t1_shifted = t0_sec - 2457000, t1_sec - 2457000

        # Plot error bars
        if yerr is not None:
            ax.errorbar(time_shifted[ind], y[ind], yerr=yerr[ind], fmt='none', alpha=0.2, color='blue')

        # Plot flux points
        ax.scatter(time_shifted[ind], y[ind], s=40, alpha=1, color='cornflowerblue', label=f'Sector {s}', **kwargs)

        # Plot expected transit lines
        # Plot short red line at each predicted transit time
        for tt in transit_times_shifted:
            if t0_shifted <= tt <= t1_shifted:
                ax.plot([tt, tt], [0.987, 0.9885], color='red', lw=9)

        ax.set_xlim(t0_shifted - 0.5, t1_shifted + 0.5)
        ax.set_ylim(0.987, 1.008)
        ax.set_ylabel("Relative Flux")
        ax.legend(loc='lower left')

    axes[-1].set_xlabel("Time (BJD - 2457000)")
    # plt.savefig("tess_sectors.pdf", dpi=300, bbox_inches='tight')

    plt.show()


axes = plot_tess(
    time=time,
    y=flux,
    yerr=flux_err,
    sharey=True  # optional, share y-axis for all sectors
)

# Show the plot (if tessplot does not call plt.show() internally)
#save the figure
# -----------------------------------------------------------
#  Compute per-sector TESS observing time (using raw timestamps)
# -----------------------------------------------------------

# Load TESS orbit schedule
csv_path = '/Users/u5500483/Downloads/Paper_IV/TESS_orbit_times.csv'
df_orbits = pd.read_csv(csv_path, comment='#')

# Use *raw* non-binned timestamps
time_raw = df['time'].to_numpy()  # IMPORTANT: before binning
flux_raw = df['flux'].to_numpy()

# TESS exptime: infer from median time difference
# (or hardcode: 120 s for 2-min cadence, 600 s for 10-min FFI depending on your file)
# Assign exposure time per sector (seconds)
exptime_per_sector = {
    7: 1800,   # 30 min
    33: 600,   # 10 min
    71: 200,   # 200 s
    72: 200,   # 200 s
    87: 200,   # 200 s
}

sector_times = {}

for s in sorted(df_orbits['Sector'].unique()):
    rows = df_orbits[df_orbits['Sector'] == s]
    if len(rows) < 2:
        continue

    t_start = Time(rows['Start of Orbit'].min()).jd
    t_end = Time(rows['End of Orbit'].max()).jd

    # indices of raw TESS timestamps inside this sector
    ind = np.where((time_raw >= t_start) & (time_raw <= t_end))[0]

    if len(ind) == 0:
        continue

    N_images = len(ind)
    # Use correct exposure time for this sector
    exptime = exptime_per_sector.get(s, np.nanmedian(np.diff(time_raw)) * 86400)  # fallback to dt
    total_time_days = (exptime * N_images) / 86400

    sector_times[s] = {
        "N_images": N_images,
        "Total_time_days": total_time_days,
        "Span_days": t_end - t_start,
    }

# Print summary
print("\n=== TESS Observing Time Per Sector ===")
for s, info in sector_times.items():
    print(f"Sector {s}:")
    print(f"    Valid images (Q=0): {info['N_images']}")
    print(f"    Total integration time: {info['Total_time_days']:.2f} days")
    print(f"    Sector span (start→end): {info['Span_days']:.2f} days")
    print()
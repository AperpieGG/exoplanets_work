import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from plot_images import plot_images, bin_by_time_interval
import batman

plot_images()

# --------------------------------------------------
# Define transit parameters
# --------------------------------------------------

params = batman.TransitParams()

params.t0 = 2460098.8315131348  # mid-transit time (BJD)
params.per = 58.204721  # orbital period (days)
params.rp = 0.09675  # Rp/Rs
params.a = 57.26  # a/Rs
params.inc = 88.968  # inclination (deg)
params.ecc = 0.370  # eccentricity
params.w = 97.5  # longitude of periastron (deg)

params.limb_dark = "quadratic"
params.u = [0.336, 0.246]

# --------------------------------------------------
# Read TESS CSV
# --------------------------------------------------

filename = 'data/TIC-453147896_SPOC_FLUX_NORMALISED.csv'

df = pd.read_csv(filename, header=None,
                 names=['time', 'flux', 'flux_err'], comment='#')

time_raw = df['time'].to_numpy()
flux_raw = df['flux'].to_numpy()
flux_err_raw = df['flux_err'].to_numpy()

# --------------------------------------------------
# Bin data (30 min bins for plotting)
# --------------------------------------------------

time, flux, flux_err = bin_by_time_interval(
    time_raw, flux_raw, flux_err_raw, 30)

# --------------------------------------------------
# Create smooth batman model
# --------------------------------------------------

time_model = np.linspace(np.min(time_raw), np.max(time_raw), 100000)

m = batman.TransitModel(params, time_model)
model_flux = m.light_curve(params)


# --------------------------------------------------
# Handle BJD offset for plotting
# --------------------------------------------------

def guess_xlabels(time):
    jd_moon_landing = 2440422.5

    if np.min(time) < jd_moon_landing:
        return time, 'Time [days]'
    else:
        return time - 2457000, 'Time (BJD - 2457000)'


time_plot, xlabel = guess_xlabels(time)
time_model_plot, _ = guess_xlabels(time_model)


# --------------------------------------------------
# Plot TESS sectors
# --------------------------------------------------

def plot_tess(time, y, yerr=None, sharey=False):
    jd_moon_landing = 2440422.5
    if np.nanmin(time) < jd_moon_landing:
        raise ValueError("Time array does not seem to be BJD.")

    # Load orbit schedule
    csv_path = '/Users/u5500483/Downloads/Paper_IV/TESS_orbit_times.csv'
    df_orbits = pd.read_csv(csv_path, comment='#')

    sectors_to_plot = []
    sector_bounds = []

    for s in sorted(df_orbits['Sector'].unique()):

        rows = df_orbits[df_orbits['Sector'] == s]

        if len(rows) < 2:
            continue

        t_start = Time(rows['Start of Orbit'].min()).jd
        t_end = Time(rows['End of Orbit'].max()).jd

        ind = np.where((time >= t_start) & (time <= t_end))[0]

        if len(ind) > 0:
            sectors_to_plot.append(s)
            sector_bounds.append((t_start, t_end))

    # Shift time for plotting
    time_shifted = time - 2457000

    # Compute expected transit times
    t0 = params.t0
    period = params.per

    t_min = np.nanmin(time)
    t_max = np.nanmax(time)

    n_before = int(np.floor((t_min - t0) / period))
    n_after = int(np.ceil((t_max - t0) / period))

    transit_times = t0 + np.arange(n_before, n_after + 1) * period
    transit_times_shifted = transit_times - 2457000

    # Create subplots
    N = len(sectors_to_plot)

    fig, axes = plt.subplots(
        N, 1,
        figsize=(12, 3 * N),
        sharey=sharey,
        tight_layout=True
    )

    if N == 1:
        axes = [axes]

    # Plot each sector
    for ax, s, bounds in zip(axes, sectors_to_plot, sector_bounds):

        t0_sec, t1_sec = bounds

        ind = np.where((time >= t0_sec) & (time <= t1_sec))[0]

        t0_shifted = t0_sec - 2457000
        t1_shifted = t1_sec - 2457000

        # Error bars
        if yerr is not None:
            ax.errorbar(
                time_shifted[ind],
                y[ind],
                yerr=yerr[ind],
                fmt='none',
                alpha=0.25,
                color='blue'
            )

        # Data points
        ax.scatter(
            time_shifted[ind],
            y[ind],
            s=35,
            color='cornflowerblue',
            label=f'Sector {s}'
        )

        # Plot transit model
        ax.plot(
            time_model_plot,
            model_flux,
            color='red',
            lw=2,
            # label='Transit model'
        )

        # # Plot expected transit markers
        # for tt in transit_times_shifted:
        #
        #     if t0_shifted <= tt <= t1_shifted:
        #         ax.plot([tt, tt], [0.987, 0.9885],
        #                 color='red', lw=8)

        ax.set_xlim(t0_shifted - 0.5, t1_shifted + 0.5)
        ax.set_ylim(0.987, 1.008)

        ax.set_ylabel("Relative Flux")

        ax.legend(loc='lower left')

    axes[-1].set_xlabel(xlabel)
    path = '/Users/u5500483/Downloads/'
    # plt.savefig(path + "tess_sectors.pdf", dpi=300, bbox_inches='tight')

    plt.show()


plot_tess(
    time=time,
    y=flux,
    yerr=flux_err,
    sharey=True
)

# --------------------------------------------------
# Compute per-sector observing time
# --------------------------------------------------

csv_path = '/Users/u5500483/Downloads/Paper_IV/TESS_orbit_times.csv'
df_orbits = pd.read_csv(csv_path, comment='#')

exptime_per_sector = {
    7: 1800,
    33: 600,
    71: 200,
    72: 200,
    87: 200,
}

sector_times = {}

for s in sorted(df_orbits['Sector'].unique()):

    rows = df_orbits[df_orbits['Sector'] == s]

    if len(rows) < 2:
        continue

    t_start = Time(rows['Start of Orbit'].min()).jd
    t_end = Time(rows['End of Orbit'].max()).jd

    ind = np.where(
        (time_raw >= t_start) &
        (time_raw <= t_end)
    )[0]

    if len(ind) == 0:
        continue

    N_images = len(ind)

    exptime = exptime_per_sector.get(
        s,
        np.nanmedian(np.diff(time_raw)) * 86400
    )

    total_time_days = (exptime * N_images) / 86400

    sector_times[s] = {
        "N_images": N_images,
        "Total_time_days": total_time_days,
        "Span_days": t_end - t_start
    }

print("\n=== TESS Observing Time Per Sector ===")

for s, info in sector_times.items():
    print(f"Sector {s}:")
    print(f"    Valid images: {info['N_images']}")
    print(f"    Total integration time: {info['Total_time_days']:.2f} days")
    print(f"    Sector span: {info['Span_days']:.2f} days\n")

import allesfitter
from allesfitter.computer import calculate_baseline, calculate_yerr_w
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk

from plot_dat import label
from plot_images import plot_images

plot_images()

dirname = 'data/1901'
alles = allesfitter.allesclass(dirname)

posterior_params_median = alles.posterior_params_median
epoch = posterior_params_median['b_epoch']
period = posterior_params_median['b_period']

data_path = '/Users/u5500483/Documents/GitHub/exoplanets_work/data/1901/'

datasets = [
    data_path + "ACTON.csv",
    data_path + "KeplerCam.csv",
    data_path + "LOOK.csv",
    data_path + "LCO.csv",
    data_path + "OACC-CAO.csv",
    data_path + "ULMT.csv"
]

colors = ["black", "orange", "green", "purple", "brown", "teal"]

# Vertical offsets for stacking the fluxes
offsets = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
markers = ['o', 's', '^', 'D', 'v', 'P']  # circle, square, triangle_up, diamond, triangle_down, plus


# transit_ylim = (0.976, 1.12)  # extended for offsets
transit_res_ylim = [-0.005, 0.02]  # residuals fixed

T0 = posterior_params_median['b_epoch']
P = posterior_params_median['b_period']

fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
fig.subplots_adjust(hspace=0)
fig.supxlabel('Time (T - T$_0$) [hrs]')

# Prepare full model across ±5 hours
model_times_full = np.linspace(-5, 5, 100000) / 24 * P + T0
model_hr_full = ((model_times_full - T0 + 0.5 * P) % P - 0.5 * P) * 24
model_flux_full = alles.get_posterior_median_model(inst='sg1', key='flux', xx=model_times_full)

for dataset_file, color, offset, marker in zip(datasets, colors, offsets, markers):
    df = pd.read_csv(dataset_file)
    df.columns = df.columns.str.strip()
    time = df['#time'].to_numpy()
    flux = df['flux'].to_numpy()
    flux_err = df['flux_err'].to_numpy()
    label = dataset_file.split('/')[-1].replace('.csv','')

    # Center times around transit
    N_cycle = int(round((np.mean(time) - T0) / P))
    epoch_dataset = T0 + N_cycle * P
    time_hr = ((time - epoch_dataset + 0.5 * P) % P - 0.5 * P) * 24

    # Bin data
    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
    lc = lc.bin(time_bin_size=1 / 6 / 24)
    lc_time_hr = ((lc.time.jd - epoch_dataset + 0.5 * P) % P - 0.5 * P) * 24

    # Plot binned flux with vertical offset
    ax[0].errorbar(lc_time_hr[:-1], lc.flux[:-1] + offset, yerr=lc.flux_err[:-1],
                   fmt=marker, color=color, alpha=0.8, markersize=4, label=label)

    # Plot full SG1 model with the same vertical offset
    ax[0].plot(model_hr_full, model_flux_full + offset, color='red', lw=1)

    # Residuals without offset
    sg1_model_data = alles.get_posterior_median_model(inst='sg1', key='flux', xx=lc.time.jd[:-1])
    ax[1].errorbar(lc_time_hr[:-1], lc.flux[:-1] - sg1_model_data, yerr=lc.flux_err[:-1],
                   fmt=marker, color=color, alpha=0.8, markersize=4)

# Draw a red horizontal line at 0 in residuals
ax[1].axhline(0, color='red', lw=1)

# Axes formatting
ax[0].set(ylabel='Normalized flux + offset')
ax[1].set(ylabel='Residuals', ylim=transit_res_ylim)
ax[0].set(xlim=(-4, 4))
ax[1].set(ylim=(-0.005, 0.005))
ax[0].set(ylim=(0.984, 1.12))
ax[0].legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False
)
path = '/Users/u5500483/Downloads/'
fig.savefig(path + '/sg1_LC_splited.pdf', bbox_inches='tight')
plt.show()
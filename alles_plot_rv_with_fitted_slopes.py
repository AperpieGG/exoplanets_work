import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import allesfitter

from plot_images import plot_images

RV_SCALE = 1000.0  # km/s -> m/s
DATA_DIR = "data/0205"
RV_INSTRUMENTS = ["HARPS", "CORALIE_1", "CORALIE_2"]

DISPLAY_NAMES = {
    "HARPS": "HARPS",
    "CORALIE_1": "CORALIE14",
    "CORALIE_2": "CORALIE24",
}

MARKERS = {
    "HARPS": "o",
    "CORALIE_1": "^",
    "CORALIE_2": "s",
}

COLORS = {
    "HARPS": "blue",
    "CORALIE_1": "orange",
    "CORALIE_2": "green",
}


def get_rv_jitter(inst, posterior_params_median):
    key = "jitter_rv_" + inst
    if key in posterior_params_median:
        return posterior_params_median[key]

    ln_key = "ln_jitter_rv_" + inst
    if ln_key in posterior_params_median:
        return np.exp(posterior_params_median[ln_key])

    return 0.0


def rv_baseline(alles, posterior_params_median, inst, time):
    """
    Return the fitted allesfitter RV baseline in m/s.

    allesfitter RV quantities are assumed to be in km/s, so we multiply by 1000.
    """
    data_time = np.asarray(alles.data[inst]["time"], dtype=float)
    time = np.asarray(time, dtype=float)

    mode = alles.settings[f"baseline_rv_{inst}"]
    offset = posterior_params_median[f"baseline_offset_rv_{inst}"] * RV_SCALE

    if mode == "sample_offset":
        return offset * np.ones_like(time, dtype=float)

    if mode == "sample_linear":
        slope = posterior_params_median[f"baseline_slope_rv_{inst}"] * RV_SCALE
        time_norm = (time - data_time[0]) / (data_time[-1] - data_time[0])
        return offset + slope * time_norm

    raise ValueError(f"Unsupported RV baseline mode for {inst}: {mode}")


def rv_baseline_no_offset(alles, posterior_params_median, inst, time):
    """
    Return only the fitted linear part of the RV baseline, excluding the offset.

    This is useful for the time-domain plot where we subtract the offset from
    the data but keep the fitted linear slope visible.
    """
    data_time = np.asarray(alles.data[inst]["time"], dtype=float)
    time = np.asarray(time, dtype=float)

    mode = alles.settings[f"baseline_rv_{inst}"]

    if mode == "sample_offset":
        return np.zeros_like(time, dtype=float)

    if mode == "sample_linear":
        slope = posterior_params_median[f"baseline_slope_rv_{inst}"] * RV_SCALE
        time_norm = (time - data_time[0]) / (data_time[-1] - data_time[0])
        return slope * time_norm

    raise ValueError(f"Unsupported RV baseline mode for {inst}: {mode}")


def rv_offset(posterior_params_median, inst):
    return posterior_params_median[f"baseline_offset_rv_{inst}"] * RV_SCALE


def planet_model(alles, inst, time):
    return alles.get_posterior_median_model(inst=inst, key="rv", xx=time) * RV_SCALE


def to_phase(period, epoch, time):
    return ((time - epoch) / period) % 1


def save_time_plot(alles, posterior_params_median, dirname):
    all_times = np.hstack([alles.data[inst]["time"] for inst in RV_INSTRUMENTS])
    t0 = int(all_times.min())
    time_grid = np.linspace(all_times.min() - 5, all_times.max() + 5, 2500)

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        dpi=120,
        figsize=(7, 5)
    )

    # Use HARPS only to draw a representative red model line.
    # This includes the planet model plus the HARPS fitted linear baseline,
    # but with the HARPS offset removed.
    harps_offset = rv_offset(posterior_params_median, "HARPS")
    harps_model_grid = (
        planet_model(alles, "HARPS", time_grid)
        + rv_baseline(alles, posterior_params_median, "HARPS", time_grid)
        - harps_offset
    )

    ax[0].plot(
        time_grid - t0,
        harps_model_grid,
        "r-",
        lw=1.4,
        label="Planet + fitted RV baseline"
    )

    for inst in RV_INSTRUMENTS:
        data = alles.data[inst]

        time = np.asarray(data["time"], dtype=float)
        rv = np.asarray(data["rv"], dtype=float) * RV_SCALE
        rv_err = np.asarray(data["rv_err"], dtype=float) * RV_SCALE

        offset = rv_offset(posterior_params_median, inst)

        full_model = (
            planet_model(alles, inst, time)
            + rv_baseline(alles, posterior_params_median, inst, time)
        )

        # Time plot:
        # subtract only the instrumental offset from the data,
        # so the fitted linear baseline remains visible.
        ax[0].errorbar(
            time - t0,
            rv - offset,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
            label=DISPLAY_NAMES[inst],
        )

        # Residuals:
        # subtract the full fitted model: planet + offset + slope.
        ax[1].errorbar(
            time - t0,
            rv - full_model,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
        )

    ax[1].axhline(0, color="r", lw=1)

    # ax[0].legend(loc="best", fontsize=9)
    ax[0].set_ylabel("RV - offset (m s$^{-1}$)")
    ax[1].set_ylabel("O-C (m s$^{-1}$)")
    ax[1].set_xlabel(f"Time [BJD - {t0}]")

    fig.tight_layout()

    pdf_path = os.path.join(dirname, "TIC_RV_TIME.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path


def save_phase_plot(alles, posterior_params_median, period, epoch, dirname):
    phase_grid = np.linspace(0, 1, 2000)
    time_grid = epoch + phase_grid * period

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        dpi=120,
        figsize=(7, 5)
    )

    # Phase-folded model: only the Keplerian planet model.
    # The fitted RV baselines are removed from the data.
    ax[0].plot(
        phase_grid,
        planet_model(alles, "HARPS", time_grid),
        "r-",
        lw=1.4,
        label="Planet model"
    )

    for inst in RV_INSTRUMENTS:
        data = alles.data[inst]

        time = np.asarray(data["time"], dtype=float)
        rv = np.asarray(data["rv"], dtype=float) * RV_SCALE
        rv_err = np.asarray(data["rv_err"], dtype=float) * RV_SCALE

        phase = to_phase(period, epoch, time)

        baseline = rv_baseline(alles, posterior_params_median, inst, time)
        model = planet_model(alles, inst, time)

        # Remove the full fitted baseline before phase-folding.
        ax[0].errorbar(
            phase,
            rv - baseline,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
            label=DISPLAY_NAMES[inst],
        )

        # Residuals after removing baseline and planet model.
        ax[1].errorbar(
            phase,
            rv - baseline - model,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
        )

    ax[1].axhline(0, color="r", lw=1)

    # ax[0].legend(loc="best", fontsize=9)
    ax[0].set_ylabel("RV - baseline (m s$^{-1}$)")
    ax[1].set_ylabel("O-C (m s$^{-1}$)")
    ax[1].set_xlabel("Orbital Phase")

    fig.tight_layout()

    pdf_path = os.path.join(dirname, "TIC_RV_PHASE.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path


def print_fitted_rv_baselines(alles, posterior_params_median, posterior_samples):
    print("\nFitted RV baselines extracted directly from allesfitter:")
    print("-" * 80)

    for inst in RV_INSTRUMENTS:
        mode = alles.settings[f"baseline_rv_{inst}"]

        offset_key = f"baseline_offset_rv_{inst}"
        slope_key = f"baseline_slope_rv_{inst}"

        offset_kms = posterior_params_median[offset_key]
        offset_ms = offset_kms * RV_SCALE

        print(f"{DISPLAY_NAMES[inst]}")
        print(f"  baseline mode                  : {mode}")
        print(f"  offset parameter               : {offset_key}")
        print(f"  offset                         : {offset_ms:.6f} m/s")
        print(f"  offset                         : {offset_kms:.9f} km/s")

        if mode == "sample_linear":
            slope_kms = posterior_params_median[slope_key]
            slope_ms = slope_kms * RV_SCALE

            time = np.asarray(alles.data[inst]["time"], dtype=float)
            dt = time[-1] - time[0]

            slope_per_day_ms = slope_ms / dt
            slope_per_year_ms = slope_per_day_ms * 365.25

            slope_per_day_kms = slope_kms / dt
            slope_per_year_kms = slope_per_day_kms * 365.25

            if slope_key in posterior_samples:
                slope_sigma_kms = np.nanstd(posterior_samples[slope_key])
                slope_sigma_ms = slope_sigma_kms * RV_SCALE

                slope_per_day_sigma_ms = slope_sigma_ms / dt
                slope_per_year_sigma_ms = slope_per_day_sigma_ms * 365.25

                slope_per_day_sigma_kms = slope_sigma_kms / dt
                slope_per_year_sigma_kms = slope_per_day_sigma_kms * 365.25
            else:
                slope_sigma_kms = np.nan
                slope_sigma_ms = np.nan

                slope_per_day_sigma_ms = np.nan
                slope_per_year_sigma_ms = np.nan

                slope_per_day_sigma_kms = np.nan
                slope_per_year_sigma_kms = np.nan

            print(f"  slope parameter                : {slope_key}")
            print(f"  slope                          : {slope_ms:.6f} m/s over instrument baseline")
            print(f"  slope                          : {slope_kms:.9f} km/s over instrument baseline")
            print(f"  time baseline                  : {dt:.6f} days")
            print(f"  slope per day                  : {slope_per_day_ms:.6f} m/s/day")
            print(f"  slope per year                 : {slope_per_year_ms:.6f} m/s/year")
            print(f"  slope per day                  : {slope_per_day_kms:.9f} km/s/day")
            print(f"  slope per year                 : {slope_per_year_kms:.9f} km/s/year")
            print(f"  slope uncertainty              : {slope_sigma_ms:.6f} m/s over instrument baseline")
            print(f"  slope uncertainty              : {slope_sigma_kms:.9f} km/s over instrument baseline")
            print(f"  slope-per-day uncertainty      : {slope_per_day_sigma_ms:.6f} m/s/day")
            print(f"  slope-per-year uncertainty     : {slope_per_year_sigma_ms:.6f} m/s/year")
            print(f"  slope-per-day uncertainty      : {slope_per_day_sigma_kms:.9f} km/s/day")
            print(f"  slope-per-year uncertainty     : {slope_per_year_sigma_kms:.9f} km/s/year")

        print("-" * 80)


def main():
    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=1.8,
        color_codes=True
    )
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    sns.set_context(rc={"lines.markeredgewidth": 1})

    plot_images()

    alles = allesfitter.allesclass(DATA_DIR)

    posterior_params_median = alles.posterior_params_median
    posterior_samples = alles.posterior_params
    print("\nAvailable posterior sample keys containing 'slope':")
    for key in posterior_samples.keys():
        if "slope" in key:
            print(key)

    print("\nAvailable posterior median keys containing 'slope':")
    for key in posterior_params_median.keys():
        if "slope" in key:
            print(key)

    period = posterior_params_median["b_period"]
    epoch = posterior_params_median["b_epoch"]

    # Recalculate RV uncertainties including fitted jitter.
    for inst in RV_INSTRUMENTS:
        data = alles.data[inst]
        data["rv_err"] = np.sqrt(
            data["white_noise_rv"] ** 2
            + get_rv_jitter(inst, posterior_params_median) ** 2
        )

    print_fitted_rv_baselines(
        alles,
        posterior_params_median,
        posterior_samples
    )

    output_paths = []
    output_paths.append(
        save_time_plot(
            alles,
            posterior_params_median,
            DATA_DIR
        )
    )
    output_paths.append(
        save_phase_plot(
            alles,
            posterior_params_median,
            period,
            epoch,
            DATA_DIR
        )
    )

    print("\nSaved plots:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()